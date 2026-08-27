# 03 Benchmark 与实验入口

| 脚本 | 作用 |
|---|---|
| `check_environment.py` | 检查 distributed/Gloo/NCCL/CUDA 和 GPU 数量 |
| `run_correctness.py` | 运行 semantics、sampler、gradient、checkpoint 四类实验 |
| `run_scaling.py` | 读取 TOML，运行 strong/weak scaling 或记录 unavailable |
| `run_profile.py` | 采集两个 Gloo rank 的 DDP/AllReduce CPU trace |
| `show_distributed_results.py` | 将 JSON 打印为小白可读表格 |
| `aggregate_scaling_runs.py` | 验证三次云端运行并生成中位数、离散程度与源文件哈希 |
| `summarize_module03.py` | 校验正式结果并记录源文件 SHA-256 |

`launcher.py` 使用当前 `sys.executable` 调用 `torch.distributed.run`，为每个 job
建立新 rendezvous 和 rank 结果目录。父进程只有在退出码为 0、rank 文件数量等于
world size 且实验门通过时才写成功。

完整命令和预期输出见[实验索引](../experiments/README.md)。性能 runner 使用最慢
rank elapsed time，因为一次分布式 step 的完成时间取决于最后到达同步点的 rank。
