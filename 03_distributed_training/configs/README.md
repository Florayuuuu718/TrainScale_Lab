# 03 配置说明

配置只保存控制变量，不在 Python 脚本中散落 shape、batch 和步数。所有 TOML 由
`trainscale_distributed/contract.py` 严格解析；未知字段、非法 backend/device、
重复 world size、不能整除的 global batch 会在启动多进程前报错。

| 文件 | 用途 |
|---|---|
| `correctness.toml` | sampler、梯度等价、4 epoch checkpoint/resume |
| `cpu_scaling_smoke.toml` | 初学者 1/2 rank、3 step 快速流程 |
| `cpu_scaling.toml` | 正式 1/2/4 rank strong/weak CPU 实验 |
| `gpu_scaling_smoke.toml` | 小模型 GPU/NCCL 能力门 |
| `gpu_scaling.toml` | 1/2/4/8 GPU strong/weak 配方；不足设备写 unavailable |

CPU 快速命令：

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/cpu_scaling_smoke.toml \
  --output 03_distributed_training/results/raw/tutorial/cpu_scaling_smoke.json
```

正式配置不是追求作者机器的最高吞吐，而是固定模型、global/per-rank batch、
warm-up 和 measured steps，使 1/2/4 rank 的比较有相同口径。

`gpu_scaling.toml` 在云端应完整运行三次，再用
`benchmarks/aggregate_scaling_runs.py` 取中位数，不能只保留最快一次。当前配方只有
5 次 warm-up 和 20 次 measured step，适合教学和发现通信占比，不适合作为硬件
排行榜；需要研究生产性能时，应另建冻结配置、延长测量窗口并重新采集全部 world
size，不能修改配置后与现有 `scaling_nccl_4x4090d.json` 混用。
