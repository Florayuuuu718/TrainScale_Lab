# 实验 05：CIFAR-10 小型 CNN baseline

## 问题

synthetic 链路通过后，同一训练引擎能否在真实图像、卷积、BatchNorm 和数据增强条件下学习？

## 配置与复现

- CIFAR-10 固定子集：5,000 train / 1,000 validation；
- `SmallCifarCNN`，10 类输出；batch 128；
- SGD lr 0.05、momentum 0.9、weight decay 0.0005；
- StepLR 每 epoch ×0.8；CUDA FP32；seed 7；5 epochs。

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/cifar10_baseline.toml
```

首次运行会下载 CIFAR-10，后续使用本地缓存。

## 本机结果

| epoch | train loss | train acc | valid loss | valid acc | train samples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.0347 | 0.2174 | 2.2587 | 0.222 | 4,388 |
| 2 | 1.8505 | 0.2930 | 2.0027 | 0.254 | 7,170 |
| 3 | 1.7506 | 0.3206 | 1.7417 | 0.336 | 7,381 |
| 4 | 1.6601 | 0.3600 | 1.6834 | 0.346 | 7,865 |
| 5 | 1.5815 | 0.3972 | 1.5684 | 0.421 | 7,614 |

总 wall time 4.30 s，峰值 CUDA allocated memory 142,372,352 bytes。曲线见 [`cifar10_curve.svg`](../results/cifar10_curve.svg)。

## 分析

10 类随机猜测约 10%，验证准确率从 22.2% 上升到 42.1%，train/validation loss 总体下降，说明真实数据管道、CNN、BatchNorm、反向传播和 scheduler 已串通。首 epoch 吞吐较低含 CUDA/cuDNN 初始化等冷启动，所以消融报告使用后续 epoch 平均值。

该实验只用 10% 训练集和 5 epoch，目标是系统验证，不应拿 42.1% 与完整 CIFAR-10 训练结果横向比较。若继续提高精度，需要更长训练、完整数据、更强增强和模型，但那属于后续模型实验。

## 知识总结

- synthetic 用于快速隔离训练代码，真实数据用于暴露图像变换、CNN 和 I/O 问题；
- `model.train()`/`eval()` 对 BatchNorm 的行为有实际影响；
- 冷启动和稳态吞吐要分开；
- 子集实验必须披露样本数，不能把结果表述成完整数据集精度。
