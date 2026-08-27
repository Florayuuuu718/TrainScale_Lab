# 实验 05：先建立单 GPU NCCL 基线，再进入多 GPU

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

## 后续四卡扩展

本地门通过后，已在 AutoDL 单机 4×RTX 4090D 上复用同一正式 TOML，分别运行三次
1/2/4 GPU strong/weak scaling。中位数显示 strong 四卡 speedup `0.510×`，weak
四卡 speedup `2.018×`、efficiency `50.4%`。这说明工具链能够真实启动多 GPU，
也说明小模型不会因为增加 GPU 自动线性加速。

云主机选型、环境的严格/快速路线、私有仓库上传、拓扑、三次测量、下载校验与理论
分析全部放在[实验 07：云端四卡](07_cloud_4gpu.md)。这里保留单卡基线，是为了让
读者先学会区分“代码路径通过”和“多卡性能已经证明”。

## 结论与收尾

本机证明 NCCL/CUDA DDP 基线可运行；后续云端补齐了 2/4 GPU 实测，8 GPU 仍因
实例只有四卡而不可用。runner 会自动执行可见 GPU 数以内的 world size，但读者仍
必须记录软件环境和拓扑，不能把不同机器的绝对吞吐直接混为一条性能排名。
