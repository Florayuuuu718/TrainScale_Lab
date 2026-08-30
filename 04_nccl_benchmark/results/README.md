# 04 结果与证据

正式 4 GPU 实验已经完成。推荐先读
[`module04_final_summary.json`](module04_final_summary.json) 和
[`../experiments/06_final_report.md`](../experiments/06_final_report.md)。

付费前验收见 [`module04_local_readiness.json`](module04_local_readiness.json)：代码、
CPU/WSL 测试、固定源码 checkout、单 GPU 能力探针和 unavailable 语义均已验证；其中
没有任何多 GPU 性能值；它和正式摘要承担不同角色。

- `results/raw/`：stdout、stderr、trace、rank JSON 和租卡原始包，Git 忽略；
- `results/*.json`：通过三次校验后的紧凑正式结果；
- `module04_final_summary.json`：从已校验正式 artifact 提取的紧凑公开结果；
- 本地下载的主归档 SHA-256 为 `133b1e…a13462`，内部 999 个文件全部校验通过；
- NCCL explainability 延伸归档 SHA-256 为 `705d31…d1da3`，36/36 作业通过。

原始 trace/log 不提交 Git，避免把教程仓库变成数据仓库；它们保存在带 SHA-256 的本地归档。
旧的失败/`unavailable` artifact 仅作为排障证据，不进入最终曲线。

本机若存在 `04_nccl_benchmark/trainscale-module04-*`，它们是早期正式分析引用的忽略目录，
不是公开教程内容。只要对应归档尚未做异地备份就不要删除；新学习者不需要创建这些目录，统一
使用 `local-evidence/` 保存下载归档即可。
