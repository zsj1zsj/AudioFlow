# AudioFlow Architecture

> 状态：目标架构（Target Architecture）  
> 当前实现：v1.0，仍集中在 `audioflow.py`；核心数据契约、Stage 状态、fingerprint 缓存、retry 和 Provider 配置已经落地。本文同时记录当前架构与后续演进边界。

## 1. 目标与边界

AudioFlow 是面向播客和视频的可恢复音频知识采集系统：

```text
CLI
 ↓
Workflow Engine
 ↓
Downloader → Normalize → Transcribe → Validate → Summarize → Write
                                                    ↓
                                              Storage / Cache
```

设计目标：

- 每个 Stage 可单独执行和重试。
- 产物可以缓存，并能判断缓存是否仍然有效。
- 下载器、转录器、LLM 和 Writer 可以替换。
- SQLite 保存索引和运行状态，大文件保存在文件系统。

非目标：当前阶段不实现分布式任务、动态第三方插件市场、向量数据库或 Web UI。

## 2. 分层

系统分为六层：

1. CLI：解析参数并展示结果，不实现业务逻辑。
2. Workflow：编排 Stage、恢复失败任务、记录状态。
3. Acquisition：URL/RSS 分发与下载。
4. Processing：标准化、转录、质检、分块和总结。
5. Output：Markdown 等输出格式。
6. Storage：文件产物、缓存和 SQLite 索引。

模块通过稳定的数据契约和产物路径通信。Stage 不直接依赖其他 Stage 的具体实现，但会依赖其输入契约；因此不能表述为“Stage 之间没有依赖”。

## 3. Workflow 与 Stage

标准流程：

```text
Source
  → DownloadResult
  → NormalizedAudio
  → Transcript
  → ValidationReport
  → Summary
  → Note
  → EpisodeIndex
```

建议接口：

```python
class Stage(Protocol):
    name: str

    def run(self, context: WorkflowContext) -> StageResult:
        ...
```

`StageResult` 至少包含：

- `status`: `SUCCESS | FAILED | SKIPPED`
- `artifacts`: 新生成或复用的产物
- `error`: 失败原因
- `fingerprint`: 输入与配置指纹

`RETRY` 是操作，不是最终状态。执行 retry 后，Stage 最终仍应落到 SUCCESS、FAILED 或 SKIPPED。

## 4. Downloader

```python
class Downloader(Protocol):
    def supports(self, url: str) -> bool:
        ...

    def download(self, url: str) -> DownloadResult:
        ...
```

```python
@dataclass
class DownloadResult:
    title: str
    author: str | None
    source: str
    url: str
    duration_seconds: float | None
    audio_path: Path
    cover_url: str | None
    published_at: datetime | None
```

内置实现：YouTube、Bilibili、小宇宙、公开音频 Spotify 页面和 RSS enclosure。Apple Podcasts 页面通过 Apple lookup 解析为发布者 RSS。

Downloader 通过 `register_downloader` 显式注册；新增内置 Downloader 不需要修改 Dispatcher。第三方包的自动发现不属于 v1.0 范围，未来确有外部插件需求时再增加 Python entry points。

Apple Podcasts 通常作为 RSS 发现入口；Spotify 只有在公开音频 URL 可合法获取时才能下载，不承诺所有节目可用。

## 5. Audio Processing

下载层接受 MP3、M4A、AAC、Opus、WAV 等格式。Normalize Stage 统一输出：

```text
WAV / 16 kHz / mono
```

转录器只接收标准化音频，避免每个 Provider 重复处理格式差异。

## 6. Transcriber 与 Transcript

```python
class Transcriber(Protocol):
    def transcribe(self, audio: Path, options: TranscribeOptions) -> Transcript:
        ...
```

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None

@dataclass
class Transcript:
    language: str | None
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        ...
```

候选 Provider：FasterWhisper、Whisper.cpp、OpenAI、Gemini。

转录质检只能标记低置信度和异常片段，不能保证自动修正专有名词。词表/hotwords、人工复核或 LLM 校对属于独立步骤。

## 7. Chunk 与 LLM

流程：

```text
Transcript → Chunk Summary[] → Merge → Final Summary
```

Chunk 必须在 Segment 或行边界切分。目标值可以是 1,500–2,500 token，但 token 数依赖具体模型 tokenizer；无法获得 tokenizer 时可使用字符数近似，并为模型上下文预留输出空间。

LLM Provider 应暴露最小生成能力：

```python
class LLM(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResult:
        ...
```

`summarize`、`extract_keywords`、`translate` 和 `rewrite` 是业务任务，不应成为每个 Provider 必须重复实现的方法。

当前 v1.0 使用 OpenAI-compatible Chat Completions；配置为空时总结 Stage 跳过。

## 8. Writer

Writer 的输入不能只有 Summary，因为最终笔记还需要来源、发布时间、质检和 Transcript：

```python
class Writer(Protocol):
    def write(self, document: EpisodeDocument) -> Path:
        ...
```

`EpisodeDocument` 聚合 Episode、Transcript、ValidationReport 和可选 Summary。可扩展 Writer：Markdown/Obsidian、HTML、Notion、Logseq、PDF。

## 9. Storage 与状态

SQLite 保存：

- Episode 和 Feed 索引
- 来源 URL 与稳定 ID
- 各 Stage 状态、错误和更新时间
- 产物路径与 fingerprint

音频、Transcript、Summary 和 Note 保存在文件系统，数据库不存大文件正文。

`stage_runs` 保存每个 Episode/来源的 Stage 状态、输入引用、fingerprint、产物和错误；`audioflow retry` 根据这些记录恢复。若未来需要保留同一 Stage 的多次历史执行，再增加 append-only `workflow_runs`/`stage_run_history`，当前 latest-state 表满足本地恢复需求。

## 10. Cache

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

不能只以“文件存在”判定缓存有效。v1.0 的 Stage fingerprint 包含：

- 输入文件内容 hash 或来源稳定 ID
- Stage 版本
- 关键配置（模型、语言、采样率、prompt 版本等）

只有 fingerprint 一致且产物完整时才能 SKIP；否则重新执行。

## 11. Config

项目使用 Python 3.11+ 标准库读取 `audioflow.toml`：

```toml
[download.youtube]
executable = "yt-dlp"

[download.bilibili]
executable = "yt-dlp-nightly"

[transcriber]
provider = "faster-whisper"
model = "turbo"

[llm]
provider = ""
api_key_env = "AUDIOFLOW_LLM_API_KEY"

[storage]
sqlite = "data/audioflow.db"
```

密钥只从环境变量或系统密钥存储读取，不写入配置文件或 SQLite。

## 12. Error Handling

每个 Stage 捕获并记录自己的错误；下游 Stage 在上游产物缺失时标记 SKIPPED，不覆盖原始失败原因。

```text
Download   SUCCESS
Normalize  SUCCESS
Transcribe FAILED
Validate   SKIPPED
Summarize  SKIPPED
```

重试必须从第一个 FAILED Stage 开始，并复用 fingerprint 仍有效的上游产物。

## 13. 当前实现状态

v1.0 已实现统一数据契约、显式 Downloader registry、TOML 配置、Stage latest-state、SHA-256 fingerprint、retry 和完整本地流水线。代码仍为单模块，这是当前规模下的有意选择；当 Provider 或独立维护者数量增长时，再按层拆包，不做仅改变文件位置的重构。

不应在实际需要前创建空的 Web UI、RAG、MCP 或插件框架。

## 14. Future

- RSS 定时同步
- 并发下载和任务队列
- GPU/Apple Silicon 转录后端
- AI 标签与全文搜索
- 向量索引（RAG）
- MCP Server
- Web UI / Obsidian Plugin
- iOS Shortcut / Telegram Bot

系统长期原则：可恢复、可缓存、契约稳定、按实际需求扩展。
