# 04 Results

当前尚无正式多 GPU 性能结果。

本地付费前验收见 [`module04_local_readiness.json`](module04_local_readiness.json)：代码、
CPU/WSL 测试、固定源码 checkout、单 GPU 能力探针和 unavailable 语义均已验证；其中
没有任何多 GPU 性能值。

- `results/raw/`：stdout、stderr、trace、rank JSON 和租卡原始包，Git 忽略；
- `results/*.json`：通过三次校验后的紧凑正式结果；
- `module04_summary.json` / `module04_acceptance.json`：所有必需 gate 关闭后生成。

本地硬件不足产生的 `unavailable` artifact 可以验证 runner 语义，但不能进入正式性能
曲线。正式结果必须绑定 clean commit、冻结环境、拓扑、配置和原始文件哈希。
