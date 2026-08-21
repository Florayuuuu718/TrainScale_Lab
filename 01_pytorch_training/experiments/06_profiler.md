# 实验 06：PyTorch Profiler trace 解读

## 问题

synthetic CPU baseline 的时间主要花在 DataLoader、forward/backward 还是 optimizer？

## 复现

```powershell
.venv\Scripts\python -m trainscale_training.profile `
  --config 01_pytorch_training/configs/synthetic_cpu.toml `
  --trace 01_pytorch_training/results/raw/synthetic_cpu_profiler_trace.json `
  --summary 01_pytorch_training/results/profiler_summary.json
```

Profiler schedule 为 wait 1、warmup 1、active 3，共分析 3 个 active steps。Chrome trace 放在 ignored raw 目录，小型 operator 摘要进入 Git。

## 实测 CPU activity

| event | count | CPU total (µs) |
|---|---:|---:|
| `ProfilerStep*` | 3 | 6,712.6 |
| `DataLoaderIter.__next__` | 3 | 3,364.0 |
| `train_step` | 3 | 2,979.4 |
| `aten::select` | 384 | 1,286.1 |
| `aten::stack` | 6 | 780.7 |
| `AddmmBackward` evaluate | 6 | 610.1 |
| `aten::linear` | 6 | 445.9 |
| `Optimizer.step#SGD.step` | 3 | 234.6 |

完整摘要：[`profiler_summary.json`](../results/profiler_summary.json)。

## 解读

在这个极小 workload 中，DataLoader 取 batch 的累计 CPU 时间比被 `record_function` 包裹的 train step 更高。大量 `select/stack` 来自把 TensorDataset 的逐样本 tuple 拼成 batch。真正的线性层、反向和 SGD 都很小。

这与 synthetic GPU 比 CPU 慢的现象一致：系统固定开销和数据组 batch 成本占主导，算术计算量不足。优化 MLP kernel 对这个端到端 workload 的收益上限很低。

## 限制

当前 CUDA profiler activity 因 CUPTI 初始化失败而缺失，所以本报告只声称 CPU activity 结论。不能用 CPU time 代替 CUDA kernel time。失败详情见[实验 07](07_failure_gpu_profiler.md)。

## 知识总结

- profiler 先回答“时间在哪里”，再决定优化对象；
- operator table 适合汇总，Chrome trace 适合观察时间线；
- profiler 本身有开销，短任务的绝对时间不能当无扰动 benchmark；
- 没有 CUDA activity 时必须明确披露，不能伪造 GPU 瓶颈分析。
