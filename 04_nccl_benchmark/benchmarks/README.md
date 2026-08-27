# 04 Benchmark Entrypoints

| 脚本 | 是否需要多 GPU | 作用 |
|---|---|---|
| `check_environment.py` | 否 | 记录系统、PyTorch/CUDA/NCCL、GPU 与拓扑 |
| `build_nccl_tests.py` | 执行构建时需要 Linux | 固定并验证官方源码版本 |
| `run_collectives.py` | 是 | 运行 TOML case 或记录 unavailable |
| `aggregate_runs.py` | 否 | 校验三次正式结果并逐行取中位数 |
| `plan_ddp_bridge.py` | 否 | 从 03 模型配置推导 FP32 梯度载荷 |
| `run_ddp_bridge.py` | 是 | 采集 2/4 GPU DDP/NCCL timeline |
| `show_results.py` | 否 | 显示结果摘要 |

runner 只在 `nccl-tests` return code 为 0、stdout 可解析且所有可用 `#wrong` 为 0 时
写 `success`。缺 Linux、binary 或 GPU 写 `unavailable`；超时、解析失败和数值错误写
`failed`。三种状态不能互换。

