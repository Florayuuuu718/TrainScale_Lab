# M0/M1 从这里开始

## 一句话说明

M0 建立“别人能重建的项目”，M1 建立“可以证明正确、可以恢复、可以测量的 PyTorch 单卡训练系统”。所有 M1 源码、配置、测试、结果与分析都集中在 [`01_pytorch_training/`](../../01_pytorch_training/README.md)。

## 学习顺序

1. 创建 Python 3.11 的项目专属 `.venv`；
2. 选择 CPU 或 CUDA 12.8 PyTorch wheel；
3. 运行 10 个正确性测试；
4. 用 synthetic MLP 学懂一次 train/validation；
5. 验证梯度累积和 checkpoint resume；
6. 用 CIFAR-10 子集和 CNN 验证真实图像训练；
7. 运行 workers、AMP/compile、Profiler 实验；
8. 对照结构化结果阅读分析报告。

## 第一次复现

命令均在仓库根目录执行。CPU 用户：

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\pytest -v
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

NVIDIA GPU 用户把同步命令换成 `uv sync --extra cu128 --extra dev`，确认 `torch.cuda.is_available()` 为 `True` 后运行：

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cuda.toml
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/cifar10_baseline.toml
```

CPU/GPU wheel、driver、CUDA runtime 和 `nvcc` 的区别见[环境搭建](m0-m1-environment.md)。完整命令、预期指标和知识总结以[阶段总指导](../../01_pytorch_training/README.md)为准。

## 你需要能回答的问题

- Dataset、DataLoader、batch、step、epoch 分别是什么？
- 为什么训练需要 backward/optimizer.step，而验证不能更新参数？
- 为什么 single-batch overfit 是正确性测试但不是泛化测试？
- 为什么 checkpoint 不能只保存 model weights？
- micro-batch 16 累积 4 次为什么等效 batch 是 64？
- 为什么小模型 GPU 可能比 CPU 慢？
- 为什么 workers、AMP 和 compile 都不能假定必然加速？
- 为什么“CUDA 训练能跑”不等于 nvcc、CUPTI、Triton 都可用？

回答不清楚时不要进入 M2，回到阶段 README 按实验顺序重跑并对照源码。
