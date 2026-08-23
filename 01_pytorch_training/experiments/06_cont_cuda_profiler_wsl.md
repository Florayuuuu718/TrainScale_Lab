# 实验 06（补充）：真实 CNN 的 CUDA Profiler 验收

## 问题

CUDA 训练成功只说明 GPU 能执行计算，并不自动证明 Profiler 已采集到 kernel 时间。怎样在 WSL2 的真实 CNN workload 上建立可验证的 CUDA profiling 基线？

## 概念与判定标准

PyTorch Profiler 的 CPU event 来自 host 侧调度，CUDA event 依赖 Kineto/CUPTI 收集 device kernel 与 memcpy。下面三项是不同能力：

1. CUDA forward/backward 能完成；
2. `supported_activities()` 声明支持 CUDA；
3. 导出的事件真实包含正的 device time。

只有第三项成立，才能解释 GPU kernel 时间。

## 对象特点与机制预测

CIFAR-10 CNN 含卷积、BatchNorm、激活和反向传播，10 个 active steps 比微型 synthetic workload 提供更稳定的 device activity。wait 阶段排除启动噪声，warmup 阶段让算子和缓存进入状态，active 阶段才计入摘要。因此预期能看到卷积与反向算子的正 device time，但聚合表仍不能替代时间线。

## 控制变量与复现

使用项目正式 WSL2 GPU 环境：Python 3.11、PyTorch 2.12.1+cu129、Triton 3.7.1。

```bash
.venv/bin/python -m trainscale_training.profile \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --trace 01_pytorch_training/results/raw/cifar10_cuda_profiler_trace.json \
  --summary 01_pytorch_training/results/cifar10_cuda_profiler_wsl_cu129.json \
  --wait-steps 2 --warmup-steps 2 --active-steps 10
```

摘要见 [`cifar10_cuda_profiler_wsl_cu129.json`](../results/cifar10_cuda_profiler_wsl_cu129.json)；Chrome trace 体积较大，可由命令重建，因此不进入 Git。

## 实测结果

| 验收项 | 结果 |
|---|---:|
| CUDA requested / supported | `true` / `true` |
| 实际正 device-time 聚合行 | 100 |
| 聚合行 device time 求和 | 332,447.304 µs |
| `train_step` device time | 17,522.708 µs |
| `aten::convolution_backward` | 21,095.352 µs |
| `aten::cudnn_convolution` | 5,545.618 µs |
| `aten::cudnn_batch_norm_backward` | 9,150.452 µs |

这些正值证明真实 GPU activity 已采集。`device_time_aggregate_row_count=100` 表示带正 device time 的聚合行数，不是原始 kernel 数；各行包含父子嵌套，332,447.304 µs 也不是 GPU 墙钟时间。

## 完整推理链

真实 CNN 持续发射卷积与反向 kernel → active window 内出现大量 device event → 摘要中卷积相关 operator 获得正 device time → CUDA profiling 链路通过验收。聚合行存在父子重叠 → 行数和总和不能解释为 kernel 数或端到端耗时 → 次数、并发、空闲间隙和 CPU/GPU 重叠必须回到 Chrome trace。

## 有限结论与一般结论

有限结论是：锁定环境在当前 GPU 和 CIFAR-10 active window 上能可靠采集 CUDA activity。一般方法是：任何新硬件和依赖组合都要重新执行“训练成功 → CUDA activity 声明 → 正 device time”三层验收，再讨论瓶颈。

> 排障提示：若出现 `CUPTI_ERROR_INVALID_DEVICE`，先确认没有偏离项目锁定的 PyTorch 2.12.1+cu129，并检查 WSL GPU 映射；这表示用户态 profiling 工具链兼容问题，不是学习者必须经历的实验结果。

## 回归验收

锁定环境已经通过 10/10 pytest、synthetic FP32/AMP/compile，以及本页 CIFAR-10 CUDA Profiler。Profiler 可用性没有以破坏训练正确性或 compile 为代价。
