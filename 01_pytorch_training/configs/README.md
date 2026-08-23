# 01 实验配置

每个 TOML 是一份可审计实验配方。运行时必须显式传入：

```bash
.venv/bin/python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
.venv/bin/python -m trainscale_training.train --help
```

| 配置 | 用途 |
|---|---|
| `synthetic_cpu.toml` | 最小 CPU FP32 基线 |
| `synthetic_cuda.toml` | CUDA 正确性与小任务消融 |
| `synthetic_accumulation.toml` | micro-batch 16 × 4 = effective batch 64 |
| `cifar10_baseline.toml` | 5,000/1,000、5 epoch 图像基线 |
| `cifar10_ablation.toml` | 固定 4,096/1,024，对比 FP32/AMP/compile |
| `cifar10_profiler.toml` | 小型 CUDA Profiler 配置 |
| `cifar10_modes_wsl.toml` | 完整 50,000/10,000 CIFAR-10，正式 Ubuntu AMP/compile 与 Profiler |

字段分为数据、模型、优化、执行和产物五组。配置快照写入 summary/checkpoint。练习时复制 TOML，一次只改一个主要变量，同时修改 `experiment_name` 和 `output_dir`，避免覆盖基线。AMP 只允许 CUDA；错误字段和不合理组合会在运行前报错。
