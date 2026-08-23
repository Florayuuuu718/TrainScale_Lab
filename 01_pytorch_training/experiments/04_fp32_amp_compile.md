# 实验 04：FP32、AMP 与 torch.compile

## 1. 概念是什么

FP32 是单精度基线。AMP（自动混合精度）让适合的算子使用较低精度，并用 GradScaler 降低 FP16 小梯度下溢风险。`torch.compile` 则捕获计算图，由 Inductor/Triton 生成和融合 GPU kernel。

两者优化的不是同一层：AMP 主要减少算术与数据搬运成本，compile 主要减少 Python/launch 开销并寻找融合机会。它们都会增加固定成本，所以“功能可用”和“端到端更快”必须分别验证。

## 2. 为什么对象特点会影响结果

微型 MLP 的计算很少，autocast、scaler、图捕获和 kernel launch 的固定成本可能比节省的计算更多；完整 CIFAR-10 CNN 含卷积、BatchNorm 和 ReLU，每 epoch 50,000 个样本，能更充分使用 Tensor Core，也能用多个 epoch 摊薄编译成本。

因此机制预测是：小 workload 上 AMP/compile 未必更快；代表性 CNN 上 AMP 更可能降低显存并提高稳态吞吐；compile 的首 epoch 会很慢，但缓存后的稳态可能快于 eager。AMP 与 compile 可能优化同一部分开销，组合收益不能相加。

## 3. 控制变量与复现

正式 GPU 性能实验在 WSL2 Ubuntu 的锁定环境运行。四组固定数据、seed、模型、batch 256、optimizer、scheduler、5 epochs 和 4 workers，只改变 precision 与 compile；每个 compile 变体使用独立冷缓存。

```bash
.venv/bin/python -m trainscale_training.benchmark_modes \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --output 01_pytorch_training/results/cifar10_modes_wsl.json
```

环境：WSL2 Ubuntu、Python 3.11、PyTorch 2.11.0+cu128、Triton 3.6.0、RTX 5060 Laptop GPU。结果文件保存了本次实测环境，因此升级项目依赖不会改写历史数据。

## 4. 实测看到了什么

| variant | first epoch samples/s | steady samples/s | 相对 FP32 eager | peak CUDA bytes | final valid acc |
|---|---:|---:|---:|---:|---:|
| FP32 eager | 17,221 | 27,172 | 1.00x | 205,435,392 | 0.6467 |
| AMP eager | 26,219 | 31,403 | 1.16x | 123,752,960 | 0.6497 |
| FP32 compile | 3,247 | 32,043 | 1.18x | 193,860,608 | 0.6405 |
| AMP compile | 4,691 | 32,182 | 1.18x | 121,975,808 | 0.6463 |

AMP eager 的稳态吞吐提高约 15.6%，峰值 allocated memory 降低约 39.8%。FP32 compile 的稳态提高约 17.9%，但首 epoch 吞吐下降约 81.1%，5 epochs 内尚未收回冷编译成本。AMP compile 比 AMP eager 只再提高约 2.5%，说明收益明显重叠。四组准确率接近，支持“没有明显正确性异常”，但 5 epochs 不足以证明长期收敛完全等价。

结构化结果见 [`cifar10_modes_wsl.json`](../results/cifar10_modes_wsl.json)，更完整的 break-even 与组合优化解读见[长 workload 补充实验](04_cont_amp_compile_wsl.md)。

## 5. 从特点到结论的推理链

CNN 提供持续卷积计算 → 低精度减少计算与激活开销 → AMP 的稳态吞吐提高且显存下降。Inductor 首次生成 host launcher 与 Triton kernel → compile 首 epoch 出现明显冷启动 → 缓存后融合和调度优化使稳态超过 eager。两种优化作用范围重叠 → 组合后的增益小于单项增益之和。

## 6. 有限结论与一般结论

本实验的有限结论是：在这台 GPU、这个 CNN、batch 256 和 5 epochs 下，AMP 与 compile 都改善稳态吞吐，但 compile 尚未在总 wall time 上回本。

一般结论不是“AMP/compile 必然加速”，而是：计算量越大、低精度硬件利用率越高、图越稳定、重复步数越多，固定成本越容易被摊薄。每个新模型仍应测量正确性、首轮成本、稳态吞吐、峰值显存和 break-even。

> 排障提示：如果在原生 Windows 上出现 Triton 缩进错误或 `torch.compile` 后端不可用，说明当前平台没有走通本教程要求的 Linux Inductor/Triton 路径；请回到 WSL2 Ubuntu 锁定环境，不需要先安装原生 Windows Triton。
