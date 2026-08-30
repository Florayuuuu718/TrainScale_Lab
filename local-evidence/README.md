# 本地完整证据归档

这个目录用于本机保存完整租卡归档，不提交 Git。公开仓库只跟踪本 README、各模块的
`*_final_summary.json` 和最终报告。

当前已校验的本地文件应包括：

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `trainscale-rental-0407.tar.gz` | `133b1ea5…78a13462` | 04–07 主 campaign，内部 999 个文件 |
| `module06-extension-1m.tar.gz` | `3baafa93…671648f` | 06 的 1 MiB overlap 延伸 |
| `nccl-explainability-extension.tar.gz` | `705d3165…aa71d1da3` | 04 的拓扑/算法/协议延伸 |
| `summary.json` | 见归档/本地哈希 | NCCL 延伸的原始完整汇总 |

每个 `.tar.gz` 的同名 `.sha256` 文件也应保留。不要把唯一 raw evidence 删除；需要节省空间时，
先确认外层哈希、内部 `SHA256SUMS` 和另一处备份均可用。
