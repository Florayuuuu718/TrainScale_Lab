# 实验 04：CPU 多 rank 为什么可能越跑越慢

## 为什么做

在没有多 GPU 时也可以学习 strong/weak scaling 口径、最慢 rank 计时和同步开销。
但多个 Gloo rank 共享同一颗 CPU 和内存带宽，因此这不是 GPU 加速模拟器。

## 小白名词

- **strong scaling**：global batch 128 固定；1/2/4 rank 的 local batch 为
  128/64/32。
- **weak scaling**：local batch 32 固定；global batch 为 32/64/128。
- **speedup**：N rank global throughput 除以 1 rank throughput。
- **efficiency**：speedup 再除以 N；理想为 100%。
- **straggler**：最慢 rank；同步训练一步必须等待它。

## 一般预期

共享 CPU 上增加进程会增加进程调度、Gloo AllReduce 和内存竞争。strong scaling
不一定加速；weak throughput 可能略增，但远小于理想 N 倍。

## 跟着做

先用小模型 smoke：

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/cpu_scaling_smoke.toml \
  --output 03_distributed_training/results/raw/tutorial/04_cpu_smoke.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/04_cpu_smoke.json
```

流程通过后复现正式配置：

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/cpu_scaling.toml \
  --output 03_distributed_training/results/raw/tutorial/04_cpu_formal.json
```

终端每个 mode/world size 都应为 `success`，最后为
`all_executable_cases_passed=True`。

## 实际结果

| mode | ranks | global batch | samples/s | speedup | efficiency |
|---|---:|---:|---:|---:|---:|
| strong | 1 | 128 | 27,110.7 | 1.000 | 100.0% |
| strong | 2 | 128 | 21,232.1 | 0.783 | 39.2% |
| strong | 4 | 128 | 13,691.1 | 0.505 | 12.6% |
| weak | 1 | 32 | 12,954.2 | 1.000 | 100.0% |
| weak | 2 | 64 | 13,462.6 | 1.039 | 52.0% |
| weak | 4 | 128 | 15,291.0 | 1.180 | 29.5% |

## 理论解释

strong scaling 中每 rank 计算量变小，但每一步仍要同步全部模型梯度；固定通信和
进程开销占比升高。同时四个进程争用相同 CPU core/cache/内存，所以 4 rank 只有
单 rank 约一半吞吐。weak scaling 增大总工作量，global throughput 略升，但资源
没有随 rank 增加，效率仍下降。

## 结论与收尾

本机结论是“共享 CPU 的多 rank DDP 不适合追求吞吐”，不是“DDP 一般无效”。
真正多 GPU 每个 rank 获得独立计算资源，是否加速取决于计算/通信比和互连。
