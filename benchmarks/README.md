# Shared Benchmarks

本目录只保存 04–07 真正跨模块复用的 artifact envelope、统计公式、汇总和绘图工具。
模块专属 runner 留在对应阶段中，避免公共目录变成第二套训练框架。

> 状态：04 已实现首版公共 artifact/status/hash/config 契约及 CPU 测试；后续模块按
> 实际需要扩展。01–03 的历史 JSON 和 SHA-256 保持冻结。

## 采用原则

- 向前统一，不回写或重新生成已经封存的 01–03 结果；
- 公共层只描述 provenance、状态和通用统计，不理解模块专属 payload；
- smoke、formal、acceptance 分开；
- correctness gate 失败时不发布性能结论；
- 硬件不足使用 `unavailable`，不使用 0 或空性能值伪装执行成功。

## 04–07 公共 artifact envelope

正式结果至少包含：

| 字段 | 目的 |
|---|---|
| `schema_version`、`artifact_type` | 拒绝静默读取不兼容结果 |
| `generated_at` | 记录生成时间与时区 |
| `git.commit`、`git.dirty` | 绑定源代码状态 |
| `environment` | OS、Python、PyTorch、CUDA/NCCL、GPU 与拓扑 |
| `config`、`config_sha256` | 保存控制变量和配置身份 |
| `measurement` | warm-up、迭代、重复数、同步与统计方法 |
| `status` | `success`、`failed` 或 `unavailable` |
| `correctness` | 性能解释之前必须通过的检查 |
| `metrics` | 模块专属测量 payload |
| `raw_artifacts` | stdout、trace、CSV/JSON 的相对路径和 SHA-256 |
| `boundary` | 未测硬件、已知限制和不可外推范围 |

每个模块可以扩展自己的 payload，但不得改变相同字段的含义。正式实现前先为 schema、
百分位数、重复聚合、哈希和 unavailable 语义建立 CPU 单元测试。
