# 实验 04（补充）：长 workload 下的 AMP 与 compile

## 问题

短 synthetic workload 中 AMP 反而更慢，Windows 上 CIFAR compile 又因 Triton 缺失失败。换到 WSL2、完整 CIFAR-10 和较长 epoch 后，AMP 与 `torch.compile` 是否表现出一般训练中预期的稳态收益？

## 概念

- AMP 用较低精度执行适合的算子，并用 GradScaler 降低 FP16 梯度下溢风险；
- `torch.compile` 通过 Dynamo 捕获 Python 图、AOTAutograd 生成反向图，再由 Inductor/Triton 生成与融合 GPU kernel；
- cold compile 是首次图捕获和代码生成成本，steady-state 是缓存完成后的持续执行速度；
- 显存收益与吞吐收益是不同指标，必须分别测量。

## 对象特点与机制预测

完整 CIFAR-10 每 epoch 有 50,000 个样本，CNN 包含卷积、BatchNorm 和 ReLU，远重于微型 MLP。较长运行能够摊薄 autocast、GradScaler 和编译的固定成本，因此预期 AMP 降低显存并提高稳态吞吐，compile 首 epoch 明显慢但稳态可能超过 eager。AMP 与 compile 的收益可能重叠，组合收益不应简单相加。

## 控制变量与复现

四组实验固定数据、seed、模型、batch 256、optimizer、scheduler、5 epochs、4 workers，只改变 precision 和 compile。每个 compile variant 使用独立冷缓存。

```bash
.venv/bin/python -m trainscale_training.benchmark_modes \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --output 01_pytorch_training/results/cifar10_modes_wsl.json
```

环境：WSL2 Ubuntu 26.04、Python 3.11.16、PyTorch 2.11.0+cu128、Triton 3.6.0、RTX 5060 Laptop GPU。原始摘要：[`cifar10_modes_wsl.json`](../results/cifar10_modes_wsl.json)。

## 实测结果

| variant | first epoch samples/s | steady samples/s | 相对 FP32 eager | peak CUDA bytes | final valid acc |
|---|---:|---:|---:|---:|---:|
| FP32 eager | 17,221 | 27,172 | 1.00x | 205,435,392 | 0.6467 |
| AMP eager | 26,219 | 31,403 | 1.16x | 123,752,960 | 0.6497 |
| FP32 compile | 3,247 | 32,043 | 1.18x | 193,860,608 | 0.6405 |
| AMP compile | 4,691 | 32,182 | 1.18x | 121,975,808 | 0.6463 |

AMP eager 相对 FP32 eager 的稳态吞吐提高约 15.6%，峰值 allocated memory 降低约 39.8%。FP32 compile 稳态提高约 17.9%，但首次编译令首 epoch 吞吐下降约 81.1%，总 wall time 也未在 5 epoch 内收回冷编译成本。AMP compile 比 AMP eager 只再提高约 2.5%，说明两种优化在此模型上的收益明显重叠。

## 完整推理链

CNN 提供了 Tensor Core 可利用的卷积和更高的持续计算量 → 低精度降低算术与激活开销 → AMP 的稳态吞吐和显存均优于 FP32。Inductor 首次生成 host launcher 与 Triton kernel → compile 首 epoch 很慢 → 缓存后 kernel 融合和调度优化使稳态超过 eager。四组最终准确率都在 64.05%–64.97% → 当前 5 epoch 内没有明显正确性异常，但不能据此声称长期收敛完全等价。

## 有限结论与一般预期

本实验支持一般预期：在计算量足够、可使用低精度硬件且运行足够长时，AMP 通常能节省显存并提高吞吐；compile 必须把冷编译成本与稳态收益分开。它不支持“AMP/compile 必然加速”或“二者收益可以相加”。是否值得 compile 取决于模型图稳定性、训练总步数、graph break、动态 shape 和目标硬件。

## 后续验证

- 增加 epoch，计算 compile 的 break-even step；
- 使用更大模型和 batch，观察 Tensor Core 利用与显存边界；
- 报告多次独立运行的中位数，而不是只依赖单次五 epoch；
- 在 CUDA Profiler 可用后验证 kernel 数量、launch gap 和 device time 是否与吞吐解释一致。
