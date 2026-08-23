# 实验 06：PyTorch Profiler——从 CPU 调度到 CUDA kernel

## 1. 概念是什么

Profiler 用时间证据回答“训练时间花在哪里”。CPU event 记录 host 侧的数据读取、算子调度和 optimizer 调用；CUDA event 依赖 Kineto/CUPTI 收集 device kernel、memcpy 等活动。CPU time 不能替代异步 CUDA kernel time。

`key_averages()` 适合按 operator 聚合，Chrome trace 适合看先后顺序、空隙与重叠。Profiler 自身有开销，所以先 wait/warmup，再只分析少量 active steps。

## 2. 为什么 workload 会影响结果

极小 synthetic MLP 的算术量少，组 batch、Python 调度和 launch 固定成本占比高；真实 CNN 含卷积和反向传播，GPU kernel 时间更有解释价值。因此先用 CPU profile 学会读表，再用 CIFAR-10 CUDA profile 验证 device activity。

## 3. CPU profile：复现与观察

```bash
.venv/bin/python -m trainscale_training.profile \
  --config 01_pytorch_training/configs/synthetic_cpu.toml \
  --trace 01_pytorch_training/results/raw/synthetic_cpu_profiler_trace.json \
  --summary 01_pytorch_training/results/profiler_summary.json
```

wait 1、warmup 1、active 3 的结果中，`DataLoaderIter.__next__` 累计约 3,364.0 µs，`train_step` 约 2,979.4 µs，`Optimizer.step#SGD.step` 约 234.6 µs。这里取 batch 比被标记的训练 step 更重，说明端到端瓶颈主要是固定调度与组 batch，而不是 MLP 算术。

## 4. CUDA profile：复现与验收

正式 GPU 路线使用 WSL2 Ubuntu 和项目锁定的 PyTorch 2.12.1+cu129：

```bash
.venv/bin/python -m trainscale_training.profile \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --trace 01_pytorch_training/results/raw/cifar10_cuda_profiler_trace.json \
  --summary 01_pytorch_training/results/cifar10_cuda_profiler_wsl_cu129.json \
  --wait-steps 2 --warmup-steps 2 --active-steps 10
```

成功不能只看 `torch.cuda.is_available()` 或 `supported_activities()`；摘要中还必须出现 `cuda_events_present=true`、正的 `device_time_aggregate_row_count` 和正的 device time。

10 个 active steps 得到 100 个带正 device time 的聚合行。`train_step` 聚合 device time 为 17,522.708 µs，`aten::convolution_backward` 为 21,095.352 µs，`aten::cudnn_convolution` 为 5,545.618 µs，`aten::cudnn_batch_norm_backward` 为 9,150.452 µs。结果见 [`cifar10_cuda_profiler_wsl_cu129.json`](../results/cifar10_cuda_profiler_wsl_cu129.json)。

## 5. 怎样正确解释聚合值

父算子与嵌套子算子会同时出现在 `key_averages()` 中，所以“100 行”不是原始 kernel 数，各行 device time 的总和也不是 GPU wall time。若要判断 kernel 次数、并发和空闲间隙，应打开 Chrome trace；不要把嵌套聚合行再次相加后声称得到总耗时。

## 6. 完整推理链

synthetic 对象很小 → host 组 batch 与调度固定成本占比高 → CPU 表中 DataLoader 时间突出 → 优先扩大或改进 workload，而不是盲目优化 MLP kernel。真实 CNN 提供持续 GPU 计算 → CUDA trace 出现卷积和反向传播的正 device time → GPU activity 的采集链路确实工作 → 才有资格继续用时间线判断 compute、memory、launch 或输入等待瓶颈。

## 7. 有限结论与一般结论

本实验只证明当前环境和指定 active window 中的事件分布，不代表完整 epoch 的 wall time，也不能仅凭 operator 表判定某个 kernel 是 memory-bound。一般方法是：先确认 device events 真实存在，再联合 DataLoader wait、host launch、memcpy、kernel 与同步间隙解释时间线，最后才选择优化对象。

> 排障提示：若出现 `CUPTI_ERROR_INVALID_DEVICE` 或摘要没有正 device time，通常是当前 PyTorch/Kineto/CUDA/CUPTI 与 GPU 不兼容；先确认使用 WSL2 和项目锁定的 2.12.1+cu129，不要把 CPU event 或显存记录误当成 CUDA trace。

完整 GPU 验收与字段解释见[真实 CNN CUDA Profiler 补充实验](06_cont_cuda_profiler_wsl.md)。
