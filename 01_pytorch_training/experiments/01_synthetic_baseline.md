# 实验 01：Synthetic FP32 baseline

## 问题

最小 `Dataset -> DataLoader -> model -> loss -> backward -> optimizer` 链路能否在 CPU 和 CUDA 上学习同一条合成分类规律？

## 运行前假设

- 固定 seed 后 CPU/GPU 的 loss 和 accuracy 应接近；
- 5 epoch 内 train/validation loss 应下降，accuracy 应上升；
- 模型和数据太小，GPU 可能不比 CPU 快；
- 第一个 CUDA epoch 会包含 context 初始化等固定成本。

## 数据与配置

- 512 个样本，每个样本 16 个 FP32 特征；
- 标签为隐藏规则 `argmax(X @ W)` 的 4 分类结果；
- 400 train / 112 validation；
- MLP：`16 -> 32 -> ReLU -> 4`；
- batch 64、SGD lr 0.1、momentum 0.9；
- StepLR 每 epoch 乘 0.9；seed 7；5 epochs。

配置：[`synthetic_cpu.toml`](../configs/synthetic_cpu.toml) 和 [`synthetic_cuda.toml`](../configs/synthetic_cuda.toml)。

## 复现命令

```powershell
.venv\Scripts\python -m trainscale_training.train `
  --config 01_pytorch_training/configs/synthetic_cpu.toml

.venv\Scripts\python -m trainscale_training.train `
  --config 01_pytorch_training/configs/synthetic_cuda.toml
```

## 实测结果

| epoch | train loss | train accuracy | valid loss | valid accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.3776 | 0.2850 | 1.3097 | 0.3393 |
| 2 | 1.1145 | 0.5225 | 1.0220 | 0.5982 |
| 3 | 0.7640 | 0.7675 | 0.6554 | 0.7857 |
| 4 | 0.4434 | 0.8850 | 0.4333 | 0.8482 |
| 5 | 0.2619 | 0.9375 | 0.3259 | 0.8929 |

CPU 与 CUDA 的 accuracy 完全相同，loss 只在浮点末位有微小差异。本次 CPU 总 wall time 0.0707 s，CUDA 为 0.2419 s；CUDA peak allocated memory 为 18,127,872 bytes。极短任务的计时会波动，因此这里只解释量级和瓶颈，不比较细小差值。

曲线：[`synthetic_cpu_curve.svg`](../results/synthetic_cpu_curve.svg)、[`synthetic_cuda_curve.svg`](../results/synthetic_cuda_curve.svg)。完整逐 epoch JSON 和 checkpoint 位于 `results/raw/`。

## 分析

loss 单调下降且 validation accuracy 从 0.3393 上升到 0.8929，说明训练循环确实找到了数据生成规则。CPU/GPU 指标对齐说明设备迁移没有改变数学语义。

GPU 更慢不是异常。每 epoch 只有 400 个样本和 7 个微小 MLP step，kernel launch、CPU 到 GPU 调度和首次 CUDA 初始化的固定成本占主导。这个实验验证 CUDA 正确性，不能用来评价 GPU 对大模型的加速能力。

## 知识总结

- synthetic dataset 可以隔离验证训练代码，不需要网络或真实数据文件；
- loss 下降是“优化链路有效”的证据，validation 指标用于观察未参与更新的数据；
- 同一算法在不同设备上应数值接近，但性能结论必须与 workload 规模绑定；
- smoke test 成功不等于真实任务已训练充分。
