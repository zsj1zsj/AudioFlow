# AudioFlow

AudioFlow 将视频、播客 URL 或 RSS Feed 转换为可检索、可导入 Obsidian 的知识笔记。

```text
URL / RSS
  → 下载音频
  → 16 kHz mono WAV
  → faster-whisper 转录
  → 置信度质检
  → 可选 LLM 总结
  → Markdown
  → SQLite 索引
```

## 支持范围

| 来源 | 支持方式 |
|---|---|
| YouTube | yt-dlp |
| Bilibili | yt-dlp nightly + Chrome cookies |
| 小宇宙 | 网页公开音频地址 |
| RSS / Atom | enclosure 增量下载 |
| Apple Podcasts | Apple 页面解析为发布者 RSS |
| Spotify Podcast | 仅页面公开 `og:audio` 时；不绕过平台限制 |

## 环境

- Python 3.11+
- ffmpeg / ffprobe
- macOS 上的 Bilibili 下载默认使用 `~/.local/bin/yt-dlp-nightly`

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装后使用 `audioflow` 命令。也可以执行 `python audioflow.py`。

## REST API

启动 HTTP 服务：

```bash
audioflow serve --host 0.0.0.0 --port 8080
```

如果你要在手机上通过 Tailscale 访问，`--host 0.0.0.0` 是必须的。

打开首页可以先看接口列表：

```bash
curl http://127.0.0.1:8080/
```

健康检查：

```bash
curl http://127.0.0.1:8080/health
```

### 真实例子

以下用这条 YouTube 链接作为例子：

```text
https://youtu.be/ucVQconuTJI?si=Vkpu-enF9iFIAyjY
```

1. 创建任务

```bash
curl -X POST http://127.0.0.1:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"url":"https://youtu.be/ucVQconuTJI?si=Vkpu-enF9iFIAyjY"}'
```

返回会包含 `task_id`，例如：

```json
{
  "task_id": "0a7d1cdf0711",
  "status": "QUEUED"
}
```

2. 轮询任务状态

```bash
curl http://127.0.0.1:8080/tasks/0a7d1cdf0711
```

状态会从 `QUEUED` 走到 `DOWNLOADING`、`TRANSCRIBING`、`SUMMARIZING`，最后到 `FINISHED`。

3. 下载生成文件

```bash
curl -L -o note.md 'http://127.0.0.1:8080/files/0a7d1cdf0711?kind=markdown'
curl -L -o transcript.txt 'http://127.0.0.1:8080/files/0a7d1cdf0711?kind=transcript'
curl -L -o summary.md 'http://127.0.0.1:8080/files/0a7d1cdf0711?kind=summary'
curl -L -o review.md 'http://127.0.0.1:8080/files/0a7d1cdf0711?kind=review'
```

4. 用手机访问

```text
http://<你的Tailscale-IP>:8080/
http://<你的Tailscale-IP>:8080/health
http://<你的Tailscale-IP>:8080/tasks
```

如果你的 Mac Tailscale IP 是 `100.70.209.107`，手机就访问：

```text
http://100.70.209.107:8080/
```

## 完整流水线

单个 URL：

```bash
audioflow run "https://youtu.be/VIDEO_ID" --language zh
```

批量 URL：

```text
# urls.txt
https://youtu.be/VIDEO_ID
https://www.bilibili.com/video/BV...
https://www.xiaoyuzhoufm.com/episode/...
```

```bash
audioflow run urls.txt --language zh --hotwords hotwords.txt
```

`run` 依次执行下载、标准化、转录、质检、可选总结、Markdown 和 SQLite 索引。相同输入和配置会通过 Stage fingerprint 复用缓存。

## 独立 Stage

```bash
audioflow download urls.txt
audioflow normalize
audioflow transcribe --model turbo --language zh
audioflow validate
audioflow summarize
audioflow markdown
audioflow index
```

查看状态和恢复失败步骤：

```bash
audioflow status
audioflow retry
audioflow retry --stage transcribe
```

## RSS 与 Apple Podcasts

RSS/Atom 或 Apple Podcasts 节目页面都通过 `rss add` 添加：

```bash
audioflow rss add "https://example.com/feed.xml"
audioflow rss add "https://podcasts.apple.com/us/podcast/show/id123456789"
audioflow rss list
audioflow rss sync --limit 3
```

下载后自动处理：

```bash
audioflow rss sync --limit 1 --process
```

每个 Feed 使用 GUID 去重。首次同步默认只下载最新 3 期，避免意外拉取完整历史库。

## 配置

运行配置位于 `audioflow.toml`：

```toml
[transcriber]
provider = "faster-whisper"
model = "turbo"
device = "cpu"
compute_type = "int8"
language = ""

[llm]
provider = "openai-compatible"
api_key = ""
base_url = ""
model = ""
chunk_size = 12000
```

LLM 未配置时会安全跳过总结。密钥建议只通过环境变量提供：

```bash
export AUDIOFLOW_LLM_API_KEY="..."
export AUDIOFLOW_LLM_BASE_URL="https://api.example.com/v1"
export AUDIOFLOW_LLM_MODEL="model-name"
```

## 输出

```text
data/
├── raw_audio/
├── normalized_audio/
├── transcripts/
├── summaries/
├── notes/
├── metadata/
└── audioflow.db
```

转录质检报告只标记低置信度、疑似重复或疑似非语音片段，不能保证自动修正人名、地名等错字。使用 `--hotwords` 和人工复核提高专有名词准确率。

## 开发与验证

```bash
python -m unittest -v
python -m py_compile audioflow.py test_audioflow.py
```

- [目标架构](docs/architecture.md)
- [实现一致性审计](docs/implementation-audit.md)
# AudioFlow
