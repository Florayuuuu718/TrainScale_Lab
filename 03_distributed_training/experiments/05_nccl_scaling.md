# 实验 05：单 GPU NCCL 能证明什么，不能证明什么

## 为什么做

GPU DDP 应使用一进程一 GPU 的 NCCL 路线。当前机器只有一张 RTX 5060，因此可
真实验证 NCCL process group、CUDA DDP、forward/backward、计时和显存记录，但
不能用同一 GPU 启动多个 rank 伪造多卡 speedup。

## 小白名词

- **NCCL**：NVIDIA GPU collective 通信库。
- **visible GPU**：当前进程通过 CUDA 能枚举到的 GPU。
- **one process per GPU**：local rank 0/1/... 分别绑定 GPU 0/1/...。
- **unavailable**：硬件条件不满足，因此没有测量值；不是失败，也不是零性能。

## 一般预期

world=1 会通过并给出吞吐/显存；2/4/8 因只有 1 张 GPU 被标 unavailable。world=1
没有跨 GPU AllReduce，因此只能作为多 GPU scaling 的基线和工具链门。

## 跟着做

```bash
nvidia-smi -L

# 第一次用小模型
.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/gpu_scaling_smoke.toml \
  --output 03_distributed_training/results/raw/tutorial/05_gpu_smoke.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/05_gpu_smoke.json

# 流程通过后使用正式配方
.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/gpu_scaling.toml \
  --output 03_distributed_training/results/raw/tutorial/05_gpu_formal.json
```

一张 GPU 时应看到 world=1 `success`、world=2/4/8 `unavailable`，最终仍可为
`all_executable_cases_passed=True`，因为没有可执行 case 失败。

## 实际结果

| mode | world | global batch | throughput | peak allocated | 状态 |
|---|---:|---:|---:|---:|---|
| strong | 1 | 256 | 171,727.0 samples/s | 112,237,568 B | success |
| weak | 1 | 128 | 117,956.4 samples/s | 110,663,680 B | success |
| strong/weak | 2/4/8 | — | — | — | unavailable：只有 1 GPU |

## 理论解释

strong world=1 的 batch 256 比 weak 的 batch 128 更能占满 GPU，因此吞吐更高，
这不是 strong scaling 优于 weak scaling，因为 world size 根本没有变化。多 GPU
时新增计算资源，也新增梯度 AllReduce；模型越大、互连越慢，通信占比越高。

## 结论与收尾

本机证明 NCCL/CUDA DDP 基线可运行，没有证明 2/4/8 GPU speedup。换到多 GPU
机器后直接复用同一 TOML；runner 会自动执行可见 GPU 数以内的 world size。
