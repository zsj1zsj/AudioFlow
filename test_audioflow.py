import unittest
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import audioflow
from audioflow import (
    DownloaderDispatcher,
    DownloadResult,
    EpisodeDocument,
    FasterWhisperTranscriber,
    Segment,
    SpotifyDownloader,
    Transcript,
    XiaoyuzhouDownloader,
    build_markdown_note,
    normalize_audio,
    open_database,
    parse_feed,
    chunk_transcript,
    load_config,
    llm_complete,
    finish_pipeline,
    read_hotwords,
    read_urls,
    resolve_feed_url,
    retry_command,
    safe_filename,
    save_source_metadata,
    stage_cache_hit,
    stage_fingerprint,
    record_stage,
    transcribe_audio,
    upsert_episode,
    summarize_transcript,
    validate_transcript,
    write_markdown_note,
)


class AudioFlowTest(unittest.TestCase):
    def test_dispatcher_selects_supported_sites(self):
        downloaders = DownloaderDispatcher().downloaders
        self.assertTrue(downloaders[0].supports("https://youtu.be/abc"))
        self.assertTrue(downloaders[1].supports("https://www.bilibili.com/video/BV1"))
        self.assertTrue(downloaders[2].supports("https://www.xiaoyuzhoufm.com/episode/1"))
        self.assertTrue(downloaders[3].supports("https://open.spotify.com/episode/1"))

    def test_download_result_metadata_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "episode.mp3"
            audio.touch()
            result = DownloadResult("标题", "作者", "Youtube", "https://youtu.be/1", 12.5, audio)
            with patch.object(audioflow, "METADATA_DIR", root / "metadata"):
                metadata = save_source_metadata(result)
            payload = audioflow.json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual((payload["title"], payload["audio_path"]), ("标题", str(audio)))

    def test_xiaoyuzhou_audio_priority_fallbacks(self):
        parser = XiaoyuzhouDownloader._audio_url
        self.assertEqual(parser('<meta property="og:audio" content="https://a.test/a.m4a">'), "https://a.test/a.m4a")
        self.assertEqual(
            parser('<script type="application/ld+json">{"associatedMedia":{"contentUrl":"https://a.test/b.m4a"}}</script>'),
            "https://a.test/b.m4a",
        )
        self.assertEqual(parser('x https://media.xyzcdn.net/path/c.m4a"'), "https://media.xyzcdn.net/path/c.m4a")

    def test_spotify_only_accepts_public_audio(self):
        self.assertEqual(
            SpotifyDownloader._audio_url('<meta property="og:audio" content="https://cdn.test/a.mp3">'),
            "https://cdn.test/a.mp3",
        )
        with self.assertRaises(RuntimeError):
            SpotifyDownloader._audio_url("<html></html>")

    def test_resolve_apple_podcast_feed(self):
        response = type(
            "Response",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self: {"results": [{"feedUrl": "https://feed.test/rss"}]}},
        )()
        with patch("audioflow.requests.get", return_value=response):
            resolved = resolve_feed_url("https://podcasts.apple.com/us/podcast/show/id123456")
        self.assertEqual(resolved, "https://feed.test/rss")

    def test_input_helpers(self):
        self.assertEqual(read_urls("https://youtu.be/abc"), ["https://youtu.be/abc"])
        self.assertEqual(safe_filename(' bad:/name. '), "bad_name")

    def test_toml_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audioflow.toml"
            path.write_text('[transcriber]\nmodel = "small"\n', encoding="utf-8")
            self.assertEqual(load_config(path)["transcriber"]["model"], "small")

    def test_normalize_audio_builds_16k_mono_wav(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp3"
            source.touch()

            def fake_run(command, **_kwargs):
                self.assertEqual(command[command.index("-ar") + 1], "16000")
                self.assertEqual(command[command.index("-ac") + 1], "1")
                Path(command[-1]).touch()
                return type("Result", (), {"returncode": 0, "stderr": ""})()

            with patch.object(audioflow, "NORMALIZED_AUDIO_DIR", root / "normalized"), patch(
                "audioflow.shutil.which", return_value="/usr/bin/ffmpeg"
            ), patch("audioflow.subprocess.run", side_effect=fake_run), patch.object(
                audioflow, "DATABASE_PATH", root / "audioflow.db"
            ):
                output = normalize_audio(source)

            self.assertEqual(output, root / "normalized/input.wav")
            self.assertTrue(output.exists())

    def test_transcribe_audio_writes_timestamped_text(self):
        class FakeTranscriber:
            def transcribe(self, *_args, **_kwargs):
                return Transcript(
                    "zh",
                    0.99,
                    (Segment(1.2, 2.0, "第一句"), Segment(65.0, 67.0, "第二句")),
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            source.touch()
            with patch.object(audioflow, "TRANSCRIPTS_DIR", root / "transcripts"), patch.object(
                audioflow, "METADATA_DIR", root / "metadata"
            ), patch.object(
                audioflow, "DATABASE_PATH", root / "audioflow.db"
            ):
                output = transcribe_audio(source, FakeTranscriber(), language="zh")
            self.assertEqual(output.read_text(encoding="utf-8"), "[00:00:01] 第一句\n[00:01:05] 第二句\n")

    def test_faster_whisper_adapter_returns_transcript_contract(self):
        raw = type(
            "RawSegment",
            (),
            {"start": 1, "end": 2, "text": " 内容 ", "avg_logprob": -0.2,
             "compression_ratio": 1.1, "no_speech_prob": 0.01},
        )()

        class Model:
            def transcribe(self, *_args, **_kwargs):
                return iter([raw]), type("Info", (), {"language": "zh", "language_probability": 0.98})()

        transcript = FasterWhisperTranscriber(Model()).transcribe(Path("audio.wav"), language=None, hotwords=None)
        self.assertEqual((transcript.language, transcript.full_text), ("zh", "内容"))

    def test_validate_flags_low_confidence_segments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "sample.segments.json"
            metadata.write_text(
                '{"segments": [{"start": 3, "text": "疑似错字", "avg_logprob": -0.9, '
                '"compression_ratio": 1.0, "no_speech_prob": 0.1}]}',
                encoding="utf-8",
            )
            with patch.object(audioflow, "METADATA_DIR", root), patch.object(
                audioflow, "DATABASE_PATH", root / "audioflow.db"
            ):
                report, count = validate_transcript(metadata)
            self.assertEqual(count, 1)
            self.assertIn("疑似错字", report.read_text(encoding="utf-8"))

    def test_validate_failure_is_recorded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "broken.segments.json"
            metadata.write_text("not json", encoding="utf-8")
            with patch.object(audioflow, "DATABASE_PATH", root / "audioflow.db"):
                with self.assertRaises(audioflow.json.JSONDecodeError):
                    validate_transcript(metadata)
                with open_database() as connection:
                    row = connection.execute(
                        "SELECT status FROM stage_runs WHERE episode_key='broken' AND stage='validate'"
                    ).fetchone()
            self.assertEqual(row["status"], "FAILED")

    def test_hotwords_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "words.txt"
            path.write_text("习近平\n淄博\n", encoding="utf-8")
            self.assertEqual(read_hotwords(str(path)), "习近平, 淄博")

    def test_chunk_and_summarize_transcript(self):
        text = "[00:00:00] 第一行\n[00:00:10] 第二行\n"
        self.assertEqual("\n".join(chunk_transcript(text, 1000)), text.strip())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.txt"
            source.write_text(text, encoding="utf-8")
            prompts = []

            def complete(prompt):
                prompts.append(prompt)
                return "## 一句话总结\n测试摘要"

            with patch.object(audioflow, "SUMMARIES_DIR", root / "summaries"), patch.object(
                audioflow, "DATABASE_PATH", root / "audioflow.db"
            ):
                output = summarize_transcript(source, complete)
            self.assertEqual(len(prompts), 2)
            self.assertIn("# episode", output.read_text(encoding="utf-8"))

    def test_openai_compatible_llm_response(self):
        response = type(
            "Response",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"choices": [{"message": {"content": "摘要"}}]},
            },
        )()
        with patch("audioflow.requests.post", return_value=response) as post:
            result = llm_complete("内容", api_key="key", base_url="https://api.test/v1", model="model")
        self.assertEqual(result, "摘要")
        self.assertEqual(post.call_args.args[0], "https://api.test/v1/chat/completions")

    def test_build_markdown_without_llm_summary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "episode.txt"
            transcript.write_text("[00:00:00] 正文", encoding="utf-8")
            with patch.object(audioflow, "NOTES_DIR", root / "notes"), patch.object(
                audioflow, "METADATA_DIR", root / "metadata"
            ), patch.object(audioflow, "SUMMARIES_DIR", root / "summaries"), patch.object(
                audioflow, "DATABASE_PATH", root / "audioflow.db"
            ):
                output = build_markdown_note(transcript)
            note = output.read_text(encoding="utf-8")
            self.assertIn("## 一句话总结\n\n> 待生成", note)
            self.assertIn("## 原始 Transcript\n\n[00:00:00] 正文", note)

    def test_markdown_writer_only_needs_episode_document(self):
        document = EpisodeDocument(
            stem="episode",
            title="标题",
            source="Youtube",
            url="https://youtu.be/1",
            published_at="2026-01-01",
            duration_seconds=65,
            language="zh",
            summary="",
            review="",
            transcript="[00:00:00] 正文",
        )
        with TemporaryDirectory() as directory, patch.object(audioflow, "NOTES_DIR", Path(directory)):
            note = write_markdown_note(document).read_text(encoding="utf-8")
        self.assertIn("- 时长：00:01:05", note)
        self.assertIn("- URL：https://youtu.be/1", note)

    def test_database_upsert_is_idempotent(self):
        with TemporaryDirectory() as directory:
            connection = open_database(Path(directory) / "audioflow.db")
            record = {
                "stem": "episode",
                "title": "第一版",
                "platform": "Youtube",
                "source_url": "https://example.com/1",
                "published_at": None,
                "duration_seconds": 10,
                "status": "transcribed",
                "raw_audio_path": None,
                "normalized_audio_path": None,
                "transcript_path": "episode.txt",
                "summary_path": None,
                "review_path": None,
                "note_path": None,
            }
            upsert_episode(connection, record)
            record["title"] = "第二版"
            upsert_episode(connection, record)
            row = connection.execute("SELECT COUNT(*) AS count, title FROM episodes").fetchone()
            self.assertEqual((row["count"], row["title"]), (1, "第二版"))
            connection.close()

    def test_stage_cache_requires_matching_fingerprint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            artifact = root / "output.txt"
            source.write_text("v1", encoding="utf-8")
            artifact.write_text("done", encoding="utf-8")
            with patch.object(audioflow, "DATABASE_PATH", root / "audioflow.db"):
                fingerprint = stage_fingerprint("test:v1", [source])
                record_stage("episode", "test", "SUCCESS", fingerprint=fingerprint, artifact=artifact)
                self.assertTrue(stage_cache_hit("episode", "test", fingerprint, artifact))
                source.write_text("v2", encoding="utf-8")
                self.assertFalse(stage_cache_hit("episode", "test", stage_fingerprint("test:v1", [source]), artifact))

    def test_retry_failed_markdown_stage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "episode.txt"
            transcript.write_text("[00:00:00] 正文", encoding="utf-8")
            with patch.object(audioflow, "DATABASE_PATH", root / "audioflow.db"), patch.object(
                audioflow, "NOTES_DIR", root / "notes"
            ), patch.object(audioflow, "METADATA_DIR", root / "metadata"), patch.object(
                audioflow, "SUMMARIES_DIR", root / "summaries"
            ):
                record_stage("episode", "markdown", "FAILED", input_ref=transcript, error="test")
                retry_command("episode", "markdown")
                with open_database() as connection:
                    status = connection.execute(
                        "SELECT status FROM stage_runs WHERE episode_key='episode' AND stage='markdown'"
                    ).fetchone()["status"]
            self.assertEqual(status, "SUCCESS")
            self.assertTrue((root / "notes/episode.md").exists())

    def test_parse_rss_and_atom_enclosures(self):
        rss = """<rss><channel><title>测试播客</title><item><title>第一期</title>
        <guid>one</guid><pubDate>today</pubDate><enclosure url="https://a.test/one.mp3" type="audio/mpeg"/>
        </item></channel></rss>"""
        title, entries = parse_feed(rss)
        self.assertEqual((title, entries[0]["guid"], entries[0]["audio_url"]),
                         ("测试播客", "one", "https://a.test/one.mp3"))

        atom = """<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom 播客</title><entry>
        <title>第二期</title><id>two</id><link rel="enclosure" href="https://a.test/two.m4a"/>
        </entry></feed>"""
        title, entries = parse_feed(atom)
        self.assertEqual((title, entries[0]["guid"]), ("Atom 播客", "two"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_offline_end_to_end_pipeline(self):
        class FakeTranscriber:
            cache_key = "fake:v1"

            def transcribe(self, *_args, **_kwargs):
                return Transcript("zh", 1.0, (Segment(0, 0.2, "端到端测试"),))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw = raw_dir / "episode.wav"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=0.2", str(raw)],
                check=True,
            )
            with patch.object(audioflow, "RAW_AUDIO_DIR", raw_dir), patch.object(
                audioflow, "NORMALIZED_AUDIO_DIR", root / "normalized"
            ), patch.object(audioflow, "TRANSCRIPTS_DIR", root / "transcripts"), patch.object(
                audioflow, "METADATA_DIR", root / "metadata"
            ), patch.object(audioflow, "SUMMARIES_DIR", root / "summaries"), patch.object(
                audioflow, "NOTES_DIR", root / "notes"
            ), patch.object(audioflow, "DATABASE_PATH", root / "audioflow.db"):
                normalized = normalize_audio(raw)
                transcript = finish_pipeline(normalized, FakeTranscriber(), language="zh", hotwords=None)
                with open_database() as connection:
                    count = connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            self.assertTrue(transcript.exists())
            self.assertTrue((root / "notes/episode.md").exists())
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
