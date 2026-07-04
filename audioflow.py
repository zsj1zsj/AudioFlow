"""AudioFlow v1.0: turn podcast and video sources into indexed knowledge notes."""

import json
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import requests
import typer
from rich.console import Console


def load_config(path: Path = Path("audioflow.toml")) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)


CONFIG = load_config()


def config_value(section: str, key: str, default: object) -> object:
    value: object = CONFIG
    for part in section.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part, {})
    return value.get(key, default) if isinstance(value, dict) else default

app = typer.Typer(no_args_is_help=True)
rss_app = typer.Typer(no_args_is_help=True, help="管理播客 RSS 订阅")
app.add_typer(rss_app, name="rss")
console = Console()
RAW_AUDIO_DIR = Path("data/raw_audio")
NORMALIZED_AUDIO_DIR = Path("data/normalized_audio")
TRANSCRIPTS_DIR = Path("data/transcripts")
METADATA_DIR = Path("data/metadata")
SUMMARIES_DIR = Path("data/summaries")
NOTES_DIR = Path("data/notes")
DATABASE_PATH = Path(str(config_value("storage", "sqlite", "data/audioflow.db")))
USER_AGENT = "Mozilla/5.0 (AudioFlow/1.0)"
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
TRANSCRIBER_PROVIDER = str(config_value("transcriber", "provider", "faster-whisper"))
TRANSCRIBER_MODEL = str(config_value("transcriber", "model", "turbo"))
TRANSCRIBER_DEVICE = str(config_value("transcriber", "device", "cpu"))
TRANSCRIBER_COMPUTE_TYPE = str(config_value("transcriber", "compute_type", "int8"))
TRANSCRIBER_LANGUAGE = str(config_value("transcriber", "language", "")) or None


@dataclass(frozen=True)
class DownloadResult:
    title: str
    author: str | None
    source: str
    url: str
    duration_seconds: float | None
    audio_path: Path
    cover_url: str | None = None
    published_at: datetime | None = None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class Transcript:
    language: str | None
    language_probability: float | None
    segments: tuple[Segment, ...]

    @property
    def full_text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


@dataclass(frozen=True)
class EpisodeDocument:
    stem: str
    title: str
    source: str | None
    url: str | None
    published_at: str | None
    duration_seconds: float | None
    language: str | None
    summary: str
    review: str
    transcript: str


class Transcriber(Protocol):
    cache_key: str

    def transcribe(self, audio: Path, *, language: str | None, hotwords: str | None) -> Transcript:
        ...


class FasterWhisperTranscriber:
    def __init__(self, model: object, model_name: str = "unknown") -> None:
        self.model = model
        self.cache_key = f"faster-whisper:{model_name}"

    def transcribe(self, audio: Path, *, language: str | None, hotwords: str | None) -> Transcript:
        raw_segments, info = self.model.transcribe(
            str(audio), language=language, hotwords=hotwords, vad_filter=True, beam_size=5
        )
        segments = tuple(
            Segment(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=segment.text.strip(),
                avg_logprob=round(float(segment.avg_logprob), 4),
                compression_ratio=round(float(segment.compression_ratio), 4),
                no_speech_prob=round(float(segment.no_speech_prob), 4),
            )
            for segment in raw_segments
            if segment.text.strip()
        )
        return Transcript(
            language=getattr(info, "language", language),
            language_probability=getattr(info, "language_probability", None),
            segments=segments,
        )


@app.callback()
def main() -> None:
    """Download audio from supported podcast and video sites."""


def safe_filename(name: str) -> str:
    """Return a filesystem-safe, reasonably short filename stem."""
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" .")[:120] or "audio"


def hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


class PageMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.json_scripts: list[str] = []
        self._in_json_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") and attributes.get("content"):
            self.meta[str(attributes["property"])] = str(attributes["content"])
        self._in_json_script = tag == "script" and attributes.get("type") == "application/ld+json"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_json_script = False

    def handle_data(self, data: str) -> None:
        if self._in_json_script:
            self.json_scripts.append(data)


def parse_metadata(html: str) -> PageMetadataParser:
    parser = PageMetadataParser()
    parser.feed(html)
    return parser


class BaseDownloader(ABC):
    @staticmethod
    @abstractmethod
    def supports(url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def download(self, url: str) -> DownloadResult:
        raise NotImplementedError


DOWNLOADER_REGISTRY: list[type[BaseDownloader]] = []


def register_downloader(cls: type[BaseDownloader]) -> type[BaseDownloader]:
    DOWNLOADER_REGISTRY.append(cls)
    return cls


class YtDlpDownloader(BaseDownloader):
    executable = str(config_value("download.youtube", "executable", "yt-dlp"))
    source = "Youtube"
    extra_args: tuple[str, ...] = ()

    @staticmethod
    def supports(url: str) -> bool:
        return hostname(url) in {"youtube.com", "youtu.be", "m.youtube.com"}

    def download(self, url: str) -> DownloadResult:
        executable = shutil.which(self.executable)
        local_executable = Path.home() / ".local/bin" / self.executable
        if not executable and local_executable.is_file():
            executable = str(local_executable)
        if not executable:
            raise RuntimeError(f"未找到 {self.executable}，请先安装 yt-dlp")

        RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "-x",
            "--audio-format",
            "mp3",
            "--embed-metadata",
            *self.extra_args,
            "--print",
            "after_move:filepath",
            "-o",
            str(RAW_AUDIO_DIR / "%(title).120s [%(id)s].%(ext)s"),
            url,
        ]
        console.print(f"[cyan]使用 {self.executable} 下载：[/cyan]{url}")
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"{self.executable} 下载失败")
        output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not output_lines:
            raise RuntimeError("yt-dlp 下载完成，但没有返回输出文件")
        output = Path(output_lines[-1])
        if not output.exists():
            raise RuntimeError(f"yt-dlp 返回的文件不存在：{output}")
        metadata = embedded_audio_metadata(output.stem)
        published = datetime.fromisoformat(metadata["published_at"]) if metadata.get("published_at") else None
        return DownloadResult(
            metadata.get("title", output.stem),
            None,
            self.source,
            url,
            metadata.get("duration"),
            output,
            published_at=published,
        )


@register_downloader
class YoutubeDownloader(YtDlpDownloader):
    pass


@register_downloader
class BilibiliDownloader(YtDlpDownloader):
    executable = str(config_value("download.bilibili", "executable", "yt-dlp-nightly"))
    source = "Bilibili"
    extra_args = (
        "--proxy",
        str(config_value("download.bilibili", "proxy", "")),
        "--cookies-from-browser",
        str(config_value("download.bilibili", "cookies_from_browser", "chrome")),
    )

    @staticmethod
    def supports(url: str) -> bool:
        return hostname(url) in {"bilibili.com", "m.bilibili.com", "b23.tv"}


@register_downloader
class XiaoyuzhouDownloader(BaseDownloader):
    source = "Xiaoyuzhou"
    fallback_title = "xiaoyuzhou"

    @staticmethod
    def supports(url: str) -> bool:
        host = hostname(url)
        return host == "xiaoyuzhoufm.com" or host.endswith(".xiaoyuzhoufm.com")

    @staticmethod
    def _audio_url(html: str) -> str:
        metadata = parse_metadata(html)
        if metadata.meta.get("og:audio"):
            return metadata.meta["og:audio"]

        for script in metadata.json_scripts:
            try:
                payload = json.loads(script)
            except json.JSONDecodeError:
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                media = item.get("associatedMedia")
                media_items = media if isinstance(media, list) else [media]
                for candidate in media_items:
                    if isinstance(candidate, dict) and candidate.get("contentUrl"):
                        return str(candidate["contentUrl"])

        match = re.search(r'https?://media\.xyzcdn\.net/[^"\'<>\\\s]+', html.replace(r"\/", "/"))
        if match:
            return match.group(0).replace("\\u002F", "/")
        raise RuntimeError("页面中未找到小宇宙音频地址")

    def download(self, url: str) -> DownloadResult:
        headers = {"User-Agent": USER_AGENT}
        console.print(f"[cyan]解析 {self.source} 页面：[/cyan]{url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        metadata = parse_metadata(response.text)
        title = metadata.meta.get("og:title", self.fallback_title)
        audio_url = self._audio_url(response.text)
        suffix = Path(urlparse(audio_url).path).suffix or ".m4a"
        output = RAW_AUDIO_DIR / f"{safe_filename(title)}{suffix}"
        RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        with requests.get(audio_url, stream=True, headers=headers, timeout=(30, 120)) as audio:
            audio.raise_for_status()
            with output.open("wb") as file:
                for chunk in audio.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        return DownloadResult(title, None, self.source, url, None, output)


@register_downloader
class SpotifyDownloader(XiaoyuzhouDownloader):
    source = "Spotify"
    fallback_title = "spotify"

    @staticmethod
    def supports(url: str) -> bool:
        return hostname(url) == "open.spotify.com"

    @staticmethod
    def _audio_url(html: str) -> str:
        metadata = parse_metadata(html)
        for key in ("og:audio", "og:audio:url"):
            if metadata.meta.get(key):
                return metadata.meta[key]
        raise RuntimeError("该 Spotify 页面未公开音频 URL，无法下载")


class DownloaderDispatcher:
    def __init__(self, downloaders: list[BaseDownloader] | None = None) -> None:
        self.downloaders = downloaders or [downloader() for downloader in DOWNLOADER_REGISTRY]

    def download(self, url: str) -> DownloadResult:
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError(f"无效 URL：{url}")
        for downloader in self.downloaders:
            if downloader.supports(url):
                result = downloader.download(url)
                save_source_metadata(result)
                return result
        raise ValueError(f"暂不支持该 URL：{url}")


def save_source_metadata(result: DownloadResult) -> Path:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output = METADATA_DIR / f"{result.audio_path.stem}.source.json"
    payload = asdict(result)
    payload["audio_path"] = str(result.audio_path)
    payload["published_at"] = result.published_at.isoformat() if result.published_at else None
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def download_stage_key(url: str) -> str:
    return f"url:{hashlib.sha1(url.encode()).hexdigest()[:16]}"


def download_fingerprint(url: str) -> str:
    return stage_fingerprint(
        "download:v1",
        [],
        {
            "url": url,
            "youtube": config_value("download.youtube", "executable", "yt-dlp"),
            "bilibili": config_value("download.bilibili", "executable", "yt-dlp-nightly"),
        },
    )


def cached_download_result(url: str) -> DownloadResult | None:
    key = download_stage_key(url)
    fingerprint = download_fingerprint(url)
    with open_database() as connection:
        row = connection.execute(
            "SELECT artifact_path FROM stage_runs WHERE episode_key=? AND stage='download'",
            (key,),
        ).fetchone()
    if not row or not row["artifact_path"]:
        return None
    audio = Path(row["artifact_path"])
    if not stage_cache_hit(key, "download", fingerprint, audio):
        return None
    metadata_path = METADATA_DIR / f"{audio.stem}.source.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    published = datetime.fromisoformat(payload["published_at"]) if payload.get("published_at") else None
    return DownloadResult(
        payload.get("title", audio.stem),
        payload.get("author"),
        payload.get("source", "unknown"),
        payload.get("url", url),
        payload.get("duration_seconds"),
        audio,
        payload.get("cover_url"),
        published,
    )


def read_urls(source: str) -> list[str]:
    path = Path(source)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else [source]
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def normalize_audio(source: Path, *, overwrite: bool = False) -> Path:
    """Convert one audio file to a 16 kHz mono WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先运行 brew install ffmpeg")
    if not source.is_file():
        raise ValueError(f"音频文件不存在：{source}")

    NORMALIZED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output = NORMALIZED_AUDIO_DIR / f"{source.stem}.wav"
    fingerprint = stage_fingerprint("normalize:v1", [source], {"sample_rate": 16000, "channels": 1})
    if not overwrite and stage_cache_hit(source.stem, "normalize", fingerprint, output):
        return output

    temporary = output.with_suffix(".tmp.wav")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(temporary),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        record_stage(source.stem, "normalize", "FAILED", fingerprint=fingerprint, input_ref=source, error=result.stderr.strip())
        raise RuntimeError(result.stderr.strip() or f"ffmpeg 转换失败：{source}")
    temporary.replace(output)
    record_stage(source.stem, "normalize", "SUCCESS", fingerprint=fingerprint, input_ref=source, artifact=output)
    return output


def raw_audio_files() -> list[Path]:
    if not RAW_AUDIO_DIR.is_dir():
        return []
    return sorted(path for path in RAW_AUDIO_DIR.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_audio(
    source: Path,
    transcriber: Transcriber,
    *,
    language: str | None = None,
    hotwords: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcribe one normalized WAV through the provider-independent contract."""
    if not source.is_file():
        raise ValueError(f"音频文件不存在：{source}")
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output = TRANSCRIPTS_DIR / f"{source.stem}.txt"
    metadata_output = METADATA_DIR / f"{source.stem}.segments.json"
    fingerprint = stage_fingerprint(
        "transcribe:v1",
        [source],
        {"provider": getattr(transcriber, "cache_key", type(transcriber).__name__), "language": language, "hotwords": hotwords},
    )
    if metadata_output.exists() and not overwrite and stage_cache_hit(source.stem, "transcribe", fingerprint, output):
        return output

    try:
        transcript = transcriber.transcribe(source, language=language, hotwords=hotwords)
    except Exception as error:
        record_stage(source.stem, "transcribe", "FAILED", fingerprint=fingerprint, input_ref=source, error=str(error))
        raise
    temporary = output.with_suffix(".tmp.txt")
    metadata_temporary = metadata_output.with_suffix(".tmp.json")
    records = []
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for segment in transcript.segments:
                file.write(f"[{format_timestamp(segment.start)}] {segment.text}\n")
                records.append(asdict(segment))
        metadata_temporary.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "language": transcript.language,
                    "language_probability": transcript.language_probability,
                    "segments": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(output)
        metadata_temporary.replace(metadata_output)
    except Exception:
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        record_stage(source.stem, "transcribe", "FAILED", fingerprint=fingerprint, input_ref=source, error="写入转录产物失败")
        raise
    record_stage(source.stem, "transcribe", "SUCCESS", fingerprint=fingerprint, input_ref=source, artifact=output)
    return output


def read_hotwords(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return ", ".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return value


def _validate_transcript(metadata_path: Path) -> tuple[Path, int]:
    """Write a review report for segments whose Whisper metrics indicate uncertainty."""
    fingerprint = stage_fingerprint("validate:v1", [metadata_path], {"avg_logprob": -0.7, "compression": 2.4})
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    suspicious = []
    for segment in payload.get("segments", []):
        reasons = []
        if segment["avg_logprob"] < -0.7:
            reasons.append(f"低置信度 {segment['avg_logprob']}")
        if segment["compression_ratio"] > 2.4:
            reasons.append(f"疑似重复 {segment['compression_ratio']}")
        if segment["no_speech_prob"] > 0.6:
            reasons.append(f"疑似非语音 {segment['no_speech_prob']}")
        if reasons:
            suspicious.append((segment, "；".join(reasons)))

    output = METADATA_DIR / metadata_path.name.replace(".segments.json", ".review.md")
    episode_key = metadata_path.name.removesuffix(".segments.json")
    if stage_cache_hit(episode_key, "validate", fingerprint, output):
        return output, len(suspicious)
    lines = [
        f"# 转录质检：{metadata_path.name.removesuffix('.segments.json')}",
        "",
        f"- 总段数：{len(payload.get('segments', []))}",
        f"- 待复核：{len(suspicious)}",
        "- 说明：置信度正常不代表文字一定正确，专有名词仍需人工核对。",
        "",
        "## 待复核片段",
        "",
    ]
    if suspicious:
        for segment, reason in suspicious:
            lines.append(f"- [{format_timestamp(segment['start'])}] {segment['text']}  `[{reason}]`")
    else:
        lines.append("未发现超过阈值的低置信度片段。")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_stage(episode_key, "validate", "SUCCESS", fingerprint=fingerprint, input_ref=metadata_path, artifact=output)
    return output, len(suspicious)


def validate_transcript(metadata_path: Path) -> tuple[Path, int]:
    try:
        return _validate_transcript(metadata_path)
    except Exception as error:
        episode_key = metadata_path.name.removesuffix(".segments.json")
        record_stage(episode_key, "validate", "FAILED", input_ref=metadata_path, error=str(error))
        raise


def normalized_audio_files() -> list[Path]:
    if not NORMALIZED_AUDIO_DIR.is_dir():
        return []
    return sorted(NORMALIZED_AUDIO_DIR.glob("*.wav"))


def chunk_transcript(text: str, max_chars: int = 12000) -> list[str]:
    """Split a transcript on line boundaries without losing content."""
    if max_chars < 1000:
        raise ValueError("分块长度不能小于 1000 字符")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in text.splitlines(keepends=True):
        if current and current_size + len(line) > max_chars:
            chunks.append("".join(current).strip())
            current, current_size = [], 0
        current.append(line)
        current_size += len(line)
    if current:
        chunks.append("".join(current).strip())
    return chunks


def llm_complete(prompt: str, *, api_key: str, base_url: str, model: str) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint."""
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你是严谨的中文播客笔记编辑。只依据提供的转录内容，不补充未知事实。"},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    try:
        return str(response.json()["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM 返回格式不符合 Chat Completions 规范") from error


def summarize_transcript(
    source: Path,
    complete: object,
    *,
    max_chars: int = 12000,
    cache_key: str = "llm",
    overwrite: bool = False,
) -> Path:
    """Create chunk summaries and merge them into one Markdown summary."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    output = SUMMARIES_DIR / f"{source.stem}.md"
    fingerprint = stage_fingerprint(
        "summarize:v1", [source], {"chunk_size": max_chars, "provider": cache_key}
    )
    if not overwrite and stage_cache_hit(source.stem, "summarize", fingerprint, output):
        return output
    chunks = chunk_transcript(source.read_text(encoding="utf-8"), max_chars)
    if not chunks:
        raise ValueError(f"转录文件为空：{source}")

    chunk_summaries = []
    for index, chunk in enumerate(chunks, 1):
        chunk_summaries.append(
            complete(
                f"总结以下转录片段（第 {index}/{len(chunks)} 段）。保留关键事实、人物、观点和时间戳；"
                "指出语义明显不通、可能由转录错误造成的内容，不要自行修正为未经证实的事实。\n\n"
                f"{chunk}"
            )
        )

    merged = complete(
        "将以下分段摘要合并为一份中文 Markdown 笔记。删除重复内容，保留时间戳。"
        "必须依次包含：## 一句话总结、## 核心观点、## 详细笔记、## 人物、## 概念、## 时间轴。"
        "不得添加分段摘要中不存在的事实。\n\n" + "\n\n---\n\n".join(chunk_summaries)
    )
    temporary = output.with_suffix(".tmp.md")
    temporary.write_text(f"# {source.stem}\n\n{merged.strip()}\n", encoding="utf-8")
    temporary.replace(output)
    record_stage(source.stem, "summarize", "SUCCESS", fingerprint=fingerprint, input_ref=source, artifact=output)
    return output


SUMMARY_HEADINGS = ["一句话总结", "核心观点", "详细笔记", "人物", "概念", "时间轴"]


def embedded_audio_metadata(stem: str) -> dict:
    """Read metadata embedded by yt-dlp for notes created from older downloads."""
    ffprobe = shutil.which("ffprobe")
    source = next((path for path in raw_audio_files() if path.stem == stem), None)
    if not ffprobe or not source:
        return {}
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration:format_tags=title,date,purl,comment", "-of", "json", str(source)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return {}
    payload = json.loads(result.stdout).get("format", {})
    tags = payload.get("tags", {})
    url = tags.get("purl") or tags.get("comment")
    date = tags.get("date")
    if date and re.fullmatch(r"\d{8}", date):
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    host = hostname(url) if url else ""
    platform = "Youtube" if host in {"youtube.com", "youtu.be"} else "未记录"
    return {
        "title": tags.get("title") or stem,
        "source": platform,
        "url": url,
        "published_at": date,
        "duration": float(payload["duration"]) if payload.get("duration") else None,
    }


def load_episode_document(transcript: Path) -> EpisodeDocument:
    """Load all available artifacts into the writer-independent contract."""
    source_path = METADATA_DIR / f"{transcript.stem}.source.json"
    segments_path = METADATA_DIR / f"{transcript.stem}.segments.json"
    review_path = METADATA_DIR / f"{transcript.stem}.review.md"
    summary_path = SUMMARIES_DIR / f"{transcript.stem}.md"
    source = embedded_audio_metadata(transcript.stem)
    stored_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    source.update({key: value for key, value in stored_source.items() if value is not None})
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else {}
    segment_items = segments.get("segments", [])
    duration_seconds = segment_items[-1]["end"] if segment_items else source.get("duration_seconds", source.get("duration"))
    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
    review = review_path.read_text(encoding="utf-8").strip() if review_path.exists() else ""
    return EpisodeDocument(
        stem=transcript.stem,
        title=source.get("title", transcript.stem),
        source=source.get("source"),
        url=source.get("url"),
        published_at=source.get("published_at"),
        duration_seconds=duration_seconds,
        language=segments.get("language"),
        summary=summary,
        review=review,
        transcript=transcript.read_text(encoding="utf-8").strip(),
    )


def write_markdown_note(document: EpisodeDocument, *, overwrite: bool = False) -> Path:
    """Render an EpisodeDocument as an Obsidian-ready Markdown note."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    output = NOTES_DIR / f"{document.stem}.md"
    if output.exists() and not overwrite:
        return output

    summary = document.summary
    if summary.startswith("# "):
        summary = summary.split("\n", 1)[1].lstrip() if "\n" in summary else ""
    for heading in SUMMARY_HEADINGS:
        if f"## {heading}" not in summary:
            summary += f"\n\n## {heading}\n\n> 待生成"

    review = document.review
    if review.startswith("# "):
        review = review.split("\n", 1)[1].lstrip() if "\n" in review else ""
    review = review or "> 尚未运行转录质检"

    duration = format_timestamp(document.duration_seconds) if document.duration_seconds is not None else "未记录"
    content = f"""# {document.title}

## 基本信息

- 来源：{document.source or '未记录'}
- 发布时间：{document.published_at or '未记录'}
- 时长：{duration}
- URL：{document.url or '未记录'}
- 转录语言：{document.language or '未记录'}

---

{summary.strip()}

---

## 转录质检

{review}

---

## 原始 Transcript

{document.transcript}
"""
    temporary = output.with_suffix(".tmp.md")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    return output


def _build_markdown_note(transcript: Path, *, overwrite: bool = False) -> Path:
    optional_inputs = [
        METADATA_DIR / f"{transcript.stem}.source.json",
        METADATA_DIR / f"{transcript.stem}.segments.json",
        METADATA_DIR / f"{transcript.stem}.review.md",
        SUMMARIES_DIR / f"{transcript.stem}.md",
    ]
    fingerprint = stage_fingerprint(
        "markdown:v1", [transcript, *(path for path in optional_inputs if path.exists())]
    )
    output = NOTES_DIR / f"{transcript.stem}.md"
    if not overwrite and stage_cache_hit(transcript.stem, "markdown", fingerprint, output):
        return output
    output = write_markdown_note(load_episode_document(transcript), overwrite=True)
    record_stage(transcript.stem, "markdown", "SUCCESS", fingerprint=fingerprint, input_ref=transcript, artifact=output)
    return output


def build_markdown_note(transcript: Path, *, overwrite: bool = False) -> Path:
    try:
        return _build_markdown_note(transcript, overwrite=overwrite)
    except Exception as error:
        record_stage(transcript.stem, "markdown", "FAILED", input_ref=transcript, error=str(error))
        raise


def open_database(path: Path | None = None) -> sqlite3.Connection:
    path = path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            stem TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            platform TEXT,
            source_url TEXT,
            published_at TEXT,
            duration_seconds REAL,
            status TEXT NOT NULL,
            raw_audio_path TEXT,
            normalized_audio_path TEXT,
            transcript_path TEXT,
            summary_path TEXT,
            review_path TEXT,
            note_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feeds (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            last_synced_at TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feed_items (
            id INTEGER PRIMARY KEY,
            feed_id INTEGER NOT NULL REFERENCES feeds(id),
            guid TEXT NOT NULL,
            title TEXT NOT NULL,
            published_at TEXT,
            audio_url TEXT NOT NULL,
            raw_audio_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(feed_id, guid)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_runs (
            id INTEGER PRIMARY KEY,
            episode_key TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('SUCCESS', 'FAILED', 'SKIPPED')),
            fingerprint TEXT,
            input_ref TEXT,
            artifact_path TEXT,
            error TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(episode_key, stage)
        )
        """
    )
    stage_columns = {row[1] for row in connection.execute("PRAGMA table_info(stage_runs)")}
    if "input_ref" not in stage_columns:
        connection.execute("ALTER TABLE stage_runs ADD COLUMN input_ref TEXT")
    return connection


def stage_fingerprint(stage: str, inputs: list[Path], options: dict | None = None) -> str:
    digest = hashlib.sha256(stage.encode())
    digest.update(json.dumps(options or {}, sort_keys=True, ensure_ascii=False).encode())
    for path in inputs:
        digest.update(path.name.encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def record_stage(
    episode_key: str,
    stage: str,
    status: str,
    *,
    fingerprint: str | None = None,
    input_ref: str | Path | None = None,
    artifact: Path | None = None,
    error: str | None = None,
) -> None:
    with open_database() as connection:
        connection.execute(
            "INSERT INTO stage_runs (episode_key, stage, status, fingerprint, input_ref, artifact_path, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(episode_key, stage) DO UPDATE SET "
            "status=excluded.status, fingerprint=excluded.fingerprint, "
            "input_ref=COALESCE(excluded.input_ref, stage_runs.input_ref), "
            "artifact_path=excluded.artifact_path, error=excluded.error, updated_at=CURRENT_TIMESTAMP",
            (
                episode_key,
                stage,
                status,
                fingerprint,
                str(input_ref) if input_ref is not None else None,
                str(artifact) if artifact else None,
                error,
            ),
        )
        connection.commit()


def stage_cache_hit(episode_key: str, stage: str, fingerprint: str, artifact: Path) -> bool:
    if not artifact.exists():
        return False
    with open_database() as connection:
        row = connection.execute(
            "SELECT status, fingerprint FROM stage_runs WHERE episode_key=? AND stage=?",
            (episode_key, stage),
        ).fetchone()
    if row and row["status"] in {"SUCCESS", "SKIPPED"} and row["fingerprint"] == fingerprint:
        record_stage(episode_key, stage, "SKIPPED", fingerprint=fingerprint, artifact=artifact)
        return True
    return False


def episode_record(transcript: Path) -> dict:
    stem = transcript.stem
    source_path = METADATA_DIR / f"{stem}.source.json"
    source = embedded_audio_metadata(stem)
    if source_path.exists():
        stored = json.loads(source_path.read_text(encoding="utf-8"))
        source.update({key: value for key, value in stored.items() if value is not None})
    segments_path = METADATA_DIR / f"{stem}.segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else {}
    items = segments.get("segments", [])
    duration = items[-1]["end"] if items else source.get("duration_seconds", source.get("duration"))
    raw = next((path for path in raw_audio_files() if path.stem == stem), None)
    normalized = NORMALIZED_AUDIO_DIR / f"{stem}.wav"
    summary = SUMMARIES_DIR / f"{stem}.md"
    review = METADATA_DIR / f"{stem}.review.md"
    note = NOTES_DIR / f"{stem}.md"
    status = "summarized" if summary.exists() else "validated" if review.exists() else "transcribed"
    return {
        "stem": stem,
        "title": source.get("title") or stem,
        "platform": source.get("source"),
        "source_url": source.get("url"),
        "published_at": source.get("published_at"),
        "duration_seconds": duration,
        "status": status,
        "raw_audio_path": str(raw) if raw else None,
        "normalized_audio_path": str(normalized) if normalized.exists() else None,
        "transcript_path": str(transcript),
        "summary_path": str(summary) if summary.exists() else None,
        "review_path": str(review) if review.exists() else None,
        "note_path": str(note) if note.exists() else None,
    }


def upsert_episode(connection: sqlite3.Connection, record: dict) -> None:
    columns = tuple(record)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "stem")
    connection.execute(
        f"INSERT INTO episodes ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(stem) DO UPDATE SET {updates}, updated_at=CURRENT_TIMESTAMP",
        tuple(record[column] for column in columns),
    )
    connection.commit()


def xml_text(element: ET.Element, name: str) -> str | None:
    child = next((item for item in element if item.tag.rsplit("}", 1)[-1] == name), None)
    return child.text.strip() if child is not None and child.text else None


def parse_feed(xml: str) -> tuple[str, list[dict]]:
    """Parse podcast entries from RSS 2.0 or Atom XML."""
    root = ET.fromstring(xml)
    channel = next((item for item in root if item.tag.rsplit("}", 1)[-1] == "channel"), root)
    title = xml_text(channel, "title") or "未命名订阅"
    entry_elements = [
        item for item in channel.iter() if item.tag.rsplit("}", 1)[-1] in {"item", "entry"}
    ]
    entries = []
    for item in entry_elements:
        audio_url = None
        for child in item:
            local_name = child.tag.rsplit("}", 1)[-1]
            if local_name == "enclosure" and child.attrib.get("url"):
                audio_url = child.attrib["url"]
                break
            if local_name == "link" and child.attrib.get("rel") == "enclosure":
                audio_url = child.attrib.get("href")
                break
        if not audio_url:
            continue
        entry_title = xml_text(item, "title") or "未命名节目"
        guid = xml_text(item, "guid") or xml_text(item, "id") or audio_url
        published = xml_text(item, "pubDate") or xml_text(item, "published") or xml_text(item, "updated")
        entries.append({"guid": guid, "title": entry_title, "published_at": published, "audio_url": audio_url})
    return title, entries


def resolve_feed_url(url: str) -> str:
    """Resolve Apple Podcasts show URLs to their publisher RSS feed."""
    if hostname(url) != "podcasts.apple.com":
        return url
    match = re.search(r"/id(\d+)", urlparse(url).path)
    if not match:
        raise ValueError("Apple Podcasts URL 中没有节目 ID")
    response = requests.get(
        "https://itunes.apple.com/lookup",
        params={"id": match.group(1), "entity": "podcast"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    feed_url = next((item.get("feedUrl") for item in results if item.get("feedUrl")), None)
    if not feed_url:
        raise RuntimeError("Apple Podcasts 未返回公开 RSS Feed")
    return str(feed_url)


def fetch_feed(url: str) -> tuple[str, list[dict]]:
    response = requests.get(resolve_feed_url(url), headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return parse_feed(response.text)


def download_feed_audio(entry: dict, feed_title: str) -> Path:
    url = entry["audio_url"]
    suffix = Path(urlparse(url).path).suffix
    if not suffix or len(suffix) > 6:
        suffix = ".mp3"
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    item_id = hashlib.sha1(entry["guid"].encode()).hexdigest()[:8]
    output = RAW_AUDIO_DIR / f"{safe_filename(entry['title'])} [{item_id}]{suffix}"
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=(30, 180)) as response:
        response.raise_for_status()
        temporary = output.with_suffix(f".tmp{suffix}")
        try:
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    metadata = save_source_metadata(
        DownloadResult(entry["title"], None, f"RSS: {feed_title}", url, None, output)
    )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["published_at"] = entry.get("published_at")
    metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


@app.command("download")
def download_command(source: str = typer.Argument(..., help="URL 或包含 URL 的文本文件")) -> None:
    """Download one URL or every URL in a text file."""
    dispatcher = DownloaderDispatcher()
    urls = read_urls(source)
    if not urls:
        raise typer.BadParameter("输入中没有 URL")

    failures = 0
    for url in urls:
        try:
            result = cached_download_result(url)
            if result is None:
                result = dispatcher.download(url)
                record_stage(
                    download_stage_key(url),
                    "download",
                    "SUCCESS",
                    fingerprint=download_fingerprint(url),
                    input_ref=url,
                    artifact=result.audio_path,
                )
            console.print(f"[green]完成：[/green]{result.audio_path}")
        except (ValueError, RuntimeError, requests.RequestException, subprocess.CalledProcessError) as error:
            failures += 1
            record_stage(
                download_stage_key(url),
                "download",
                "FAILED",
                fingerprint=download_fingerprint(url),
                input_ref=url,
                error=str(error),
            )
            console.print(f"[red]失败：[/red]{url}\n{error}")
    if failures:
        raise typer.Exit(code=1)


@app.command("normalize")
def normalize_command(
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="重新生成已有 WAV"),
) -> None:
    """Normalize all downloaded audio to 16 kHz mono WAV."""
    files = raw_audio_files()
    if not files:
        console.print("[yellow]data/raw_audio 中没有音频文件[/yellow]")
        return

    failures = 0
    for source in files:
        try:
            output = normalize_audio(source, overwrite=overwrite)
            console.print(f"[green]标准化完成：[/green]{output}")
        except (ValueError, RuntimeError) as error:
            failures += 1
            console.print(f"[red]标准化失败：[/red]{source}\n{error}")
    if failures:
        raise typer.Exit(code=1)


@app.command("transcribe")
def transcribe_command(
    model_name: str = typer.Option(TRANSCRIBER_MODEL, "--model", "-m", help="Whisper 模型名称或本地路径"),
    language: str | None = typer.Option(TRANSCRIBER_LANGUAGE, "--language", "-l", help="语言代码，如 zh；默认自动检测"),
    hotwords: str | None = typer.Option(None, "--hotwords", help="重点词语，或每行一个词的文本文件"),
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="重新生成已有转录"),
) -> None:
    """Transcribe all normalized WAV files with faster-whisper."""
    if TRANSCRIBER_PROVIDER != "faster-whisper":
        raise RuntimeError(f"尚未实现转录 Provider：{TRANSCRIBER_PROVIDER}")
    files = normalized_audio_files()
    if not files:
        console.print("[yellow]data/normalized_audio 中没有 WAV 文件[/yellow]")
        return
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("未安装 faster-whisper，请运行 python3 -m pip install faster-whisper") from error

    console.print(f"[cyan]加载 Whisper 模型：[/cyan]{model_name}")
    transcriber = FasterWhisperTranscriber(
        WhisperModel(model_name, device=TRANSCRIBER_DEVICE, compute_type=TRANSCRIBER_COMPUTE_TYPE),
        model_name,
    )
    failures = 0
    for source in files:
        try:
            output = transcribe_audio(
                source, transcriber, language=language, hotwords=read_hotwords(hotwords), overwrite=overwrite
            )
            console.print(f"[green]转录完成：[/green]{output}")
        except (ValueError, RuntimeError) as error:
            failures += 1
            console.print(f"[red]转录失败：[/red]{source}\n{error}")
    if failures:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_command() -> None:
    """Generate review reports from Whisper segment confidence metrics."""
    files = sorted(METADATA_DIR.glob("*.segments.json")) if METADATA_DIR.is_dir() else []
    if not files:
        console.print("[yellow]没有分段指标；请使用 --overwrite 重新转录旧文件[/yellow]")
        return
    for metadata_path in files:
        output, count = validate_transcript(metadata_path)
        console.print(f"[green]质检完成：[/green]{output}（待复核 {count} 段）")


@app.command("summarize")
def summarize_command(
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="重新生成已有摘要"),
    max_chars: int = typer.Option(int(config_value("llm", "chunk_size", 12000)), "--chunk-size", help="每个转录分块的最大字符数"),
) -> None:
    """Summarize transcripts with an OpenAI-compatible API."""
    api_key = os.getenv("AUDIOFLOW_LLM_API_KEY", str(config_value("llm", "api_key", ""))).strip()
    base_url = os.getenv("AUDIOFLOW_LLM_BASE_URL", str(config_value("llm", "base_url", ""))).strip()
    model = os.getenv("AUDIOFLOW_LLM_MODEL", str(config_value("llm", "model", ""))).strip()
    if not all((api_key, base_url, model)):
        console.print("[yellow]LLM 尚未配置，跳过总结；下载、转录和质检不受影响[/yellow]")
        return

    files = sorted(TRANSCRIPTS_DIR.glob("*.txt")) if TRANSCRIPTS_DIR.is_dir() else []
    if not files:
        console.print("[yellow]data/transcripts 中没有转录文件[/yellow]")
        return
    complete = lambda prompt: llm_complete(
        prompt, api_key=api_key, base_url=base_url, model=model
    )
    failures = 0
    for source in files:
        try:
            output = summarize_transcript(
                source,
                complete,
                max_chars=max_chars,
                cache_key=f"{config_value('llm', 'provider', 'openai-compatible')}:{base_url}:{model}",
                overwrite=overwrite,
            )
            console.print(f"[green]总结完成：[/green]{output}")
        except (ValueError, RuntimeError, requests.RequestException) as error:
            failures += 1
            record_stage(source.stem, "summarize", "FAILED", input_ref=source, error=str(error))
            console.print(f"[red]总结失败：[/red]{source}\n{error}")
    if failures:
        raise typer.Exit(code=1)


@app.command("markdown")
def markdown_command(
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="重新生成已有笔记"),
) -> None:
    """Build Obsidian-ready Markdown notes from available artifacts."""
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt")) if TRANSCRIPTS_DIR.is_dir() else []
    if not files:
        console.print("[yellow]data/transcripts 中没有转录文件[/yellow]")
        return
    for transcript in files:
        output = build_markdown_note(transcript, overwrite=overwrite)
        console.print(f"[green]Markdown 完成：[/green]{output}")


@app.command("index")
def index_command() -> None:
    """Index existing episodes in SQLite."""
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt")) if TRANSCRIPTS_DIR.is_dir() else []
    with open_database() as connection:
        for transcript in files:
            upsert_episode(connection, episode_record(transcript))
    console.print(f"[green]SQLite 索引完成：[/green]{DATABASE_PATH}（{len(files)} 条）")


@app.command("list")
def list_command() -> None:
    """List indexed episodes."""
    with open_database() as connection:
        rows = connection.execute(
            "SELECT id, title, platform, status, duration_seconds FROM episodes ORDER BY updated_at DESC"
        ).fetchall()
    if not rows:
        console.print("[yellow]数据库中没有节目[/yellow]")
        return
    for row in rows:
        duration = format_timestamp(row["duration_seconds"]) if row["duration_seconds"] is not None else "--:--:--"
        console.print(f"{row['id']:>3}  {duration}  {row['status']:<11}  {row['platform'] or '-':<10}  {row['title']}")


@rss_app.command("add")
def rss_add_command(url: str) -> None:
    """Subscribe to a podcast RSS or Atom feed."""
    title, _entries = fetch_feed(url)
    with open_database() as connection:
        connection.execute(
            "INSERT INTO feeds (url, title) VALUES (?, ?) "
            "ON CONFLICT(url) DO UPDATE SET title=excluded.title",
            (url, title),
        )
        connection.commit()
    console.print(f"[green]订阅成功：[/green]{title}")


@rss_app.command("list")
def rss_list_command() -> None:
    """List podcast feed subscriptions."""
    with open_database() as connection:
        rows = connection.execute("SELECT id, title, url, last_synced_at FROM feeds ORDER BY id").fetchall()
    if not rows:
        console.print("[yellow]没有 RSS 订阅[/yellow]")
        return
    for row in rows:
        console.print(f"{row['id']:>3}  {row['title']}\n     {row['url']}  上次同步：{row['last_synced_at'] or '-'}")


@rss_app.command("sync")
def rss_sync_command(
    limit: int = typer.Option(3, "--limit", min=1, help="每个订阅最多下载的新节目数"),
    process: bool = typer.Option(False, "--process", help="下载后继续标准化、转录和生成笔记"),
) -> None:
    """Download new episodes from every subscribed feed."""
    with open_database() as connection:
        feeds = connection.execute("SELECT id, title, url FROM feeds ORDER BY id").fetchall()
        if not feeds:
            console.print("[yellow]没有 RSS 订阅[/yellow]")
            return
        downloaded = 0
        transcriber = None
        for feed in feeds:
            try:
                title, entries = fetch_feed(feed["url"])
                new_entries = [
                    entry
                    for entry in entries
                    if connection.execute(
                        "SELECT 1 FROM feed_items WHERE feed_id=? AND guid=?", (feed["id"], entry["guid"])
                    ).fetchone()
                    is None
                ][:limit]
                for entry in reversed(new_entries):
                    output = download_feed_audio(entry, title)
                    connection.execute(
                        "INSERT INTO feed_items (feed_id, guid, title, published_at, audio_url, raw_audio_path) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (feed["id"], entry["guid"], entry["title"], entry["published_at"], entry["audio_url"], str(output)),
                    )
                    connection.commit()
                    downloaded += 1
                    console.print(f"[green]RSS 下载完成：[/green]{output}")
                    if process:
                        normalized = normalize_audio(output)
                        if transcriber is None:
                            from faster_whisper import WhisperModel

                            transcriber = FasterWhisperTranscriber(
                                WhisperModel(
                                    TRANSCRIBER_MODEL,
                                    device=TRANSCRIBER_DEVICE,
                                    compute_type=TRANSCRIBER_COMPUTE_TYPE,
                                ),
                                TRANSCRIBER_MODEL,
                            )
                        finish_pipeline(
                            normalized,
                            transcriber,
                            language=TRANSCRIBER_LANGUAGE,
                            hotwords=None,
                        )
                connection.execute(
                    "UPDATE feeds SET title=?, last_synced_at=CURRENT_TIMESTAMP WHERE id=?", (title, feed["id"])
                )
                connection.commit()
            except (ET.ParseError, requests.RequestException, RuntimeError, ValueError) as error:
                console.print(f"[red]RSS 同步失败：[/red]{feed['url']}\n{error}")
        console.print(f"[green]RSS 同步完成：[/green]下载 {downloaded} 个新节目")


def llm_settings() -> tuple[str, str, str] | None:
    api_key = os.getenv("AUDIOFLOW_LLM_API_KEY", str(config_value("llm", "api_key", ""))).strip()
    base_url = os.getenv("AUDIOFLOW_LLM_BASE_URL", str(config_value("llm", "base_url", ""))).strip()
    model = os.getenv("AUDIOFLOW_LLM_MODEL", str(config_value("llm", "model", ""))).strip()
    return (api_key, base_url, model) if all((api_key, base_url, model)) else None


def summarize_one(source: Path, *, overwrite: bool = False) -> Path | None:
    settings = llm_settings()
    if settings is None:
        return None
    api_key, base_url, model = settings
    complete = lambda prompt: llm_complete(prompt, api_key=api_key, base_url=base_url, model=model)
    try:
        return summarize_transcript(
            source,
            complete,
            max_chars=int(config_value("llm", "chunk_size", 12000)),
            cache_key=f"{config_value('llm', 'provider', 'openai-compatible')}:{base_url}:{model}",
            overwrite=overwrite,
        )
    except Exception as error:
        record_stage(source.stem, "summarize", "FAILED", input_ref=source, error=str(error))
        raise


def finish_pipeline(
    normalized: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    hotwords: str | None,
) -> Path:
    transcript = transcribe_audio(normalized, transcriber, language=language, hotwords=hotwords)
    console.print(f"[green]转录完成：[/green]{transcript}")
    metadata = METADATA_DIR / f"{normalized.stem}.segments.json"
    report, count = validate_transcript(metadata)
    console.print(f"[green]质检完成：[/green]{report}（待复核 {count} 段）")
    summary = summarize_one(transcript)
    if summary:
        console.print(f"[green]总结完成：[/green]{summary}")
    else:
        console.print("[yellow]LLM 未配置，跳过总结[/yellow]")
    note = build_markdown_note(transcript, overwrite=True)
    console.print(f"[green]Markdown 完成：[/green]{note}")
    with open_database() as connection:
        upsert_episode(connection, episode_record(transcript))
    console.print(f"[green]SQLite 已更新：[/green]{DATABASE_PATH}")
    return transcript


@app.command("status")
def status_command() -> None:
    """Show the latest state of every recorded Stage."""
    with open_database() as connection:
        rows = connection.execute(
            "SELECT episode_key, stage, status, error, updated_at FROM stage_runs "
            "ORDER BY updated_at DESC, episode_key, stage"
        ).fetchall()
    if not rows:
        console.print("[yellow]没有 Stage 运行记录[/yellow]")
        return
    for row in rows:
        detail = f"  {row['error']}" if row["error"] else ""
        console.print(f"{row['status']:<7}  {row['stage']:<10}  {row['episode_key']}{detail}")


@app.command("retry")
def retry_command(
    episode: str | None = typer.Option(None, "--episode", help="只重试指定 episode key"),
    stage: str | None = typer.Option(None, "--stage", help="只重试指定 Stage"),
) -> None:
    """Retry failed stages and continue their downstream pipeline."""
    query = "SELECT episode_key, stage, input_ref FROM stage_runs WHERE status='FAILED'"
    parameters: list[str] = []
    if episode:
        query += " AND episode_key=?"
        parameters.append(episode)
    if stage:
        query += " AND stage=?"
        parameters.append(stage)
    with open_database() as connection:
        rows = connection.execute(query, parameters).fetchall()
    if not rows:
        console.print("[green]没有需要重试的 Stage[/green]")
        return

    transcriber = None
    for row in rows:
        input_ref = row["input_ref"]
        if not input_ref:
            console.print(f"[red]无法重试：[/red]{row['episode_key']} / {row['stage']} 缺少输入引用")
            continue
        try:
            if row["stage"] == "download":
                run_command(input_ref, TRANSCRIBER_LANGUAGE, None)
                continue
            source = Path(input_ref)
            if row["stage"] == "normalize":
                normalized = normalize_audio(source, overwrite=True)
                if transcriber is None:
                    from faster_whisper import WhisperModel

                    transcriber = FasterWhisperTranscriber(
                        WhisperModel(
                            TRANSCRIBER_MODEL,
                            device=TRANSCRIBER_DEVICE,
                            compute_type=TRANSCRIBER_COMPUTE_TYPE,
                        ),
                        TRANSCRIBER_MODEL,
                    )
                finish_pipeline(normalized, transcriber, language=TRANSCRIBER_LANGUAGE, hotwords=None)
            elif row["stage"] == "transcribe":
                if transcriber is None:
                    from faster_whisper import WhisperModel

                    transcriber = FasterWhisperTranscriber(
                        WhisperModel(
                            TRANSCRIBER_MODEL,
                            device=TRANSCRIBER_DEVICE,
                            compute_type=TRANSCRIBER_COMPUTE_TYPE,
                        ),
                        TRANSCRIBER_MODEL,
                    )
                finish_pipeline(source, transcriber, language=TRANSCRIBER_LANGUAGE, hotwords=None)
            elif row["stage"] == "validate":
                validate_transcript(source)
                transcript = TRANSCRIPTS_DIR / f"{source.name.removesuffix('.segments.json')}.txt"
                summarize_one(transcript)
                build_markdown_note(transcript, overwrite=True)
            elif row["stage"] == "summarize":
                if summarize_one(source, overwrite=True) is None:
                    raise RuntimeError("LLM 未配置")
                build_markdown_note(source, overwrite=True)
            elif row["stage"] == "markdown":
                build_markdown_note(source, overwrite=True)
            else:
                raise RuntimeError(f"未知 Stage：{row['stage']}")
        except Exception as error:
            record_stage(row["episode_key"], row["stage"], "FAILED", input_ref=input_ref, error=str(error))
            console.print(f"[red]重试失败：[/red]{row['episode_key']} / {row['stage']}\n{error}")


@app.command("run")
def run_command(
    source: str = typer.Argument(..., help="URL 或包含 URL 的文本文件"),
    language: str | None = typer.Option(TRANSCRIBER_LANGUAGE, "--language", "-l", help="语言代码，如 zh"),
    hotwords: str | None = typer.Option(None, "--hotwords", help="重点词语，或每行一个词的文本文件"),
) -> None:
    """Run download, normalize, transcribe, validate, optional summary, note and index stages."""
    if TRANSCRIBER_PROVIDER != "faster-whisper":
        raise RuntimeError(f"尚未实现转录 Provider：{TRANSCRIBER_PROVIDER}")
    urls = read_urls(source)
    if not urls:
        raise typer.BadParameter("输入中没有 URL")

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("未安装 faster-whisper，请在项目虚拟环境中运行") from error

    dispatcher = DownloaderDispatcher()
    transcriber = None
    failures = 0
    for url in urls:
        download_key = download_stage_key(url)
        download_fp = download_fingerprint(url)
        try:
            download = cached_download_result(url)
            if download is None:
                try:
                    download = dispatcher.download(url)
                except Exception as error:
                    record_stage(
                        download_key,
                        "download",
                        "FAILED",
                        fingerprint=download_fp,
                        input_ref=url,
                        error=str(error),
                    )
                    raise
                record_stage(
                    download_key,
                    "download",
                    "SUCCESS",
                    fingerprint=download_fp,
                    input_ref=url,
                    artifact=download.audio_path,
                )
            console.print(f"[green]下载完成：[/green]{download.audio_path}")
            normalized = normalize_audio(download.audio_path)
            console.print(f"[green]标准化完成：[/green]{normalized}")
            if transcriber is None:
                console.print(f"[cyan]加载 Whisper 模型：[/cyan]{TRANSCRIBER_MODEL}")
                transcriber = FasterWhisperTranscriber(
                    WhisperModel(
                        TRANSCRIBER_MODEL,
                        device=TRANSCRIBER_DEVICE,
                        compute_type=TRANSCRIBER_COMPUTE_TYPE,
                    ),
                    TRANSCRIBER_MODEL,
                )
            finish_pipeline(
                normalized,
                transcriber,
                language=language,
                hotwords=read_hotwords(hotwords),
            )
        except (ValueError, RuntimeError, requests.RequestException) as error:
            failures += 1
            console.print(f"[red]处理失败：[/red]{url}\n{error}")
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
