# AudioFlow 实现一致性审计

审计基线：`audioflow.py`、`docs/architecture.md`、最初 Roadmap。  
状态含义：DONE 已验证；PARTIAL 部分实现；TODO 尚未实现。

| 要求 | 当前证据 | 状态 | 统一方式 |
|---|---|---:|---|
| YouTube/Bilibili/小宇宙下载 | 三个 Downloader；真实下载已验证 | DONE | 保持 |
| 16 kHz mono WAV | `normalize_audio`；ffprobe 实测 | DONE | 保持 |
| faster-whisper 转录 | `FasterWhisperTranscriber`；turbo 端到端实测 | DONE | 保持 |
| 转录质检 | Segment 指标与 review 报告 | DONE | 文档明确“标记而非自动纠错” |
| Chunk + LLM Summary | 分块、合并和兼容 API 响应测试；无配置时安全跳过 | DONE | 真实供应商调用需用户密钥 |
| Obsidian Markdown | `EpisodeDocument` → Markdown；真实笔记已生成 | DONE | 保持 |
| SQLite | Episode/Feed/Stage 索引与迁移已实现 | DONE | 保持 |
| RSS 增量下载 | RSS/Atom、GUID 去重、limit、可选 `--process` | DONE | 保持 |
| Apple Podcasts | Apple URL → lookup → publisher RSS；真实 The Daily feed 已验证 | DONE | 保持 |
| Spotify Podcast | 支持页面公开 `og:audio` 的节目；不绕过未公开媒体 | DONE | 平台不公开音频时明确失败 |
| Provider 配置 | `audioflow.toml` 驱动下载、转录、LLM、存储 | DONE | 密钥仍优先环境变量 |
| Downloader registry | `register_downloader` 显式注册 | DONE | 不做动态插件市场 |
| Stage 独立状态 | `stage_runs` 保存状态、输入、fingerprint、产物和错误 | DONE | 保持 latest-state；历史审计按需增加 |
| retry | `audioflow retry`；Markdown 失败恢复测试 | DONE | 保持 |
| fingerprint cache | Download/Normalize/Transcribe/Validate/Summarize/Markdown 使用 SHA-256 | DONE | 保持 |
| `run` 完整流程 | 下载→标准化→转录→质检→可选总结→Markdown→SQLite | DONE | 保持 |
| 可安装 `audioflow` CLI | `pyproject.toml` console script 已安装实测 | DONE | 保持 |
| 权威用户文档 | README、架构和实现审计已统一 | DONE | 变更时同步更新 |

## 实施顺序

1. 配置、registry、Stage 状态与 fingerprint。
2. retry 和 `run` 行为统一。
3. Apple Podcasts 与公开 RSS/audio 来源解析。
4. 可安装 CLI、README、全量回归和端到端验收。

本文是差异跟踪表；项目完成前不得仅通过修改文档把 TODO 标记为完成。
