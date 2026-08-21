# 01 · PyTorch Training（M1 完整学习单元）

这个目录是一间可以独立进入的训练实验室。你在这里学习的不是“调用一个现成模型”，而是亲手验证一条训练系统：数据怎样进入模型、loss 怎样产生梯度、参数怎样更新、验证为什么不能更新参数、训练怎样保存和恢复，以及性能数据应该怎样测量和解释。

本阶段的源码、自动测试、实机实验、结构化结果和分析报告均在本目录。建议严格按下面顺序边运行边理解。

## 1. 我们正在做什么

```text
Dataset -> DataLoader -> batch -> model -> logits -> loss
                                              |
optimizer <- parameter gradients <- backward-+

validation: Dataset -> model.eval() -> metrics（不 backward、不更新参数）
```

本阶段回答：最小训练链路是否能学习；真实 CIFAR-10 能否训练；checkpoint 能否精确续训；梯度累积是否正确；workers、AMP、compile 如何影响性能；Profiler 能看到什么以及工具失败怎样记录。

## 2. 目录地图

```text
01_pytorch_training/
├── README.md                 # 本总指导
├── trainscale_training/      # 本阶段全部可执行源码
│   ├── data.py               # synthetic / CIFAR-10 与 DataLoader
│   ├── models.py             # MLP / SmallCifarCNN
│   ├── engine.py             # FP32、AMP、累积、训练、验证、scheduler
│   ├── checkpoint.py         # 完整状态保存与恢复
│   ├── train.py              # 单次训练入口
│   ├── benchmark*.py         # 消融和 workers 实验
│   ├── profile.py            # PyTorch Profiler
│   └── plot.py / summarize.py
├── configs/                  # TOML 实验配方
├── tests/                    # 10 个 CPU 正确性测试
├── experiments/              # 假设、命令、结果与分析
├── results/                  # 可提交的小型 JSON / SVG
│   └── raw/                  # checkpoint、逐 epoch JSON、trace（Git 忽略）
└── data/                     # CIFAR-10（Git 忽略）
```

## 3. 创建隔离环境

项目使用独立 `.venv`。PyTorch wheel 会安装在这个虚拟环境中，所以不同项目通常各有一份；代价是占磁盘，收益是项目之间不会互相破坏。

本阶段固定 Python 3.11、PyTorch 2.11。GPU 使用 CUDA 12.8 wheel（`cu128`）；本机驱动 577.05 可运行。暂不选 PyTorch 2.12 + CUDA 13.0，因为它要求更高的 Windows 驱动。CUDA wheel 自带训练需要的运行库；`nvcc` 仅在编译自定义 CUDA 扩展或从源码构建时需要，M1 不需要。

从仓库根目录执行 CPU 或 CUDA 二选一，两个 extra 已显式互斥：

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev

# NVIDIA GPU 用户改用这一条
uv sync --extra cu128 --extra dev
```

每次打开新 PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python --version
```

GPU 检查：

```powershell
nvidia-smi
.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

本机预期关键输出为 `2.11.0+cu128`、`12.8`、`True`。若看到 `+cpu / None / False`，当前 venv 装的是 CPU wheel，重新执行 CUDA 的 `uv sync` 即可，不需要安装 nvcc。详见[环境指导](../docs/getting-started/m0-m1-environment.md)。

## 4. 先验证代码

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\mypy 01_pytorch_training/trainscale_training
.venv\Scripts\pytest -v
```

当前应为 10 个测试通过。其中 single-batch overfit 让模型反复看同一个 batch 并达到 100% accuracy，用来快速发现 forward、loss、backward、optimizer 或标签连接错误。它不证明泛化能力。逐项解释见[测试说明](tests/README.md)。

## 5. Synthetic FP32：第一次完整训练

synthetic 数据有 16 个特征、4 个类别，标签由固定隐藏规则 `argmax(X @ W)` 生成，不是无规律随机标签。模型是 `16 -> 32 -> ReLU -> 4` 的 MLP。

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml

# 有可用 GPU 再运行
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cuda.toml
```

validation loss 应从约 1.31 降至 0.326，accuracy 从约 33.9% 升至 89.3%。每次运行产生 `metrics.jsonl`、`summary.json` 和 `last.pt`；位置为 `results/raw/<run>/`。解释见[实验 01](experiments/01_synthetic_baseline.md)。极小 MLP 上 CPU 比 GPU 快并不反常，因为 GPU 固定启动成本大于计算量。

## 6. 梯度累积与 checkpoint 续训

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_accumulation.toml
```

micro-batch 16 累积 4 次再更新，有效 batch 为 64。测试将其与直接 batch 64 的参数更新逐项比较。`--resume` 会恢复 model、optimizer、scheduler、GradScaler、epoch/global step、Python/NumPy/PyTorch/CUDA/DataLoader RNG；字段见 [checkpoint 契约](../docs/checkpoint-contract.md)，解释见[实验 02](experiments/02_accumulation_and_resume.md)。

## 7. 真实 CIFAR-10

CIFAR-10 包含 32×32 RGB 图像和 10 个类别。本实验为学习速度固定取 5,000 train / 1,000 validation，使用小型 CNN，不是追求榜单精度的完整训练。

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/cifar10_baseline.toml
```

首次运行下载数据到 `01_pytorch_training/data/`。本机 5 epoch 的 validation accuracy 为 `22.2% -> 42.1%`，最终 validation loss 为 `1.5684`。查看[曲线](results/cifar10_curve.svg)和[实验 05](experiments/05_cifar10_baseline.md)。

## 8. 性能实验

一次只改变一个主要变量：

```powershell
# DataLoader workers
.venv\Scripts\python -m trainscale_training.benchmark_workers --workers 0 1 2 4 --samples 512 --input-dim 128 --batch-size 32 --delay-ms 1 --repeats 3 --output 01_pytorch_training/results/dataloader_workers.json

# FP32 / AMP / compile
.venv\Scripts\python -m trainscale_training.benchmark --config 01_pytorch_training/configs/cifar10_ablation.toml --output 01_pytorch_training/results/cifar10_ablation.json

# 可靠的 CPU activity profile
.venv\Scripts\python -m trainscale_training.profile --config 01_pytorch_training/configs/synthetic_cpu.toml --trace 01_pytorch_training/results/raw/synthetic_cpu_profiler_trace.json --summary 01_pytorch_training/results/profiler_summary.json
```

本机 workers=0 最快；CIFAR AMP 稳态吞吐约提高 7.8%，峰值显存约减少 50.8%；CIFAR compile 因 Windows 环境没有可用 Triton 而失败。这些是有边界的本机观测，不是普适规律。完整解释见[实验索引](experiments/README.md)。

## 9. 怎样读结果

先问正确性，再问性能：loss 是否下降；accuracy 是否高于随机猜测（4 类 25%、10 类 10%）；validation 是否不更新参数；对照是否只改变一个主变量；吞吐是否排除首轮冷启动；AMP/compile 后准确率是否仍合理；环境、失败和适用边界是否记录完整。

`results/m1_summary.json` 是机器可读总表，`experiments/*.md` 才是人读的解释。不要只看一个 samples/s 数字就下结论。

## 10. M1 完成清单

- [x] synthetic dataset 与 single-batch overfit；
- [x] FP32 train/validation 与 CIFAR-10 baseline；
- [x] 配置、seed、设备和结构化日志；
- [x] 完整 checkpoint 与精确 resume；
- [x] AMP、梯度累积、scheduler；
- [x] workers、FP32/AMP/compile 消融；
- [x] Profiler 摘要与失败实验；
- [x] 10 个 CPU 测试与首个 CPU CI；
- [x] 源码、结果、分析与复现说明均归档在本目录。

继续阅读：[配置](configs/README.md) → [测试](tests/README.md) → [实验](experiments/README.md) → [结果](results/README.md)。
