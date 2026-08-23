# 01 · PyTorch Training

> 状态：已封存。源码、锁定环境、10 项测试、训练/恢复、workers、AMP/compile 与 CUDA Profiler 均已在正式路线验收；后续只接受修正，不再追加改变 01 范围的功能。

这个目录是一间可以独立进入的训练实验室。你在这里学习的不是“调用一个现成模型”，而是亲手验证一条训练系统：数据怎样进入模型、loss 怎样产生梯度、参数怎样更新、验证为什么不能更新参数、训练怎样保存和恢复，以及性能数据应该怎样测量和解释。

本阶段的源码、自动测试、实机实验、结构化结果和分析报告均在本目录。建议严格按下面顺序边运行边理解。

## 1. 我们正在做什么

```text
Dataset -> DataLoader -> batch -> model -> logits -> loss
                                              |
optimizer <- parameter gradients <- backward-+

validation: Dataset -> model.eval() -> metrics（不 backward、不更新参数）
```

本阶段回答：最小训练链路是否能学习；真实 CIFAR-10 能否训练；checkpoint 能否精确续训；梯度累积是否正确；workers、AMP、compile 如何影响性能；Profiler 怎样从 CPU 调度走到真实 CUDA kernel。

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

Windows + NVIDIA GPU 学习者若要完成本阶段全部实验，请先完成 [WSL2 Ubuntu 从零教程](../docs/getting-started/wsl2-gpu.md)，并把仓库放在 Ubuntu 的 `~/projects/TrainScale_Lab`。这样 compile、Profiler 以及后续 Triton/NCCL 都沿用同一套 Linux 工具链；不要等原生 Windows 功能失败后再迁移，也不要复用 Windows `.venv`。

项目正式基线为 Python 3.11、PyTorch 2.12.1。GPU 路线使用 CUDA 12.9 wheel（`cu129`），CPU 路线使用匹配的 CPU wheel。CUDA wheel 自带 01 训练需要的 runtime；只有 02 编译自定义 CUDA C++ 扩展时才需要系统 CUDA Toolkit/`nvcc`。

在 Ubuntu 项目根目录执行：

```bash
uv sync --extra cu129 --extra dev --python 3.11
.venv/bin/python -c "import torch, triton; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(triton.__version__)"
```

GPU 路线应看到 `2.12.1+cu129`、`12.9`、`True`、实际 GPU 名称和 Triton `3.7.1`。版本号正确只证明依赖已导入；后面的训练、compile 和 CUDA Profiler 实验才分别验证对应能力。

只有 CPU，或只学习基础训练时，可以在原生 Windows PowerShell 执行：

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU 与 `cu129` extra 互斥。完整的路径、驱动、runtime、Triton 和 CUPTI 分层解释见[环境教程](../docs/getting-started/wsl2-gpu.md)。

## 4. 先验证代码

```bash
.venv\Scripts\ruff check .
.venv\Scripts\mypy 01_pytorch_training/trainscale_training
.venv\Scripts\pytest -v
```

当前应为 10 个测试通过。其中 single-batch overfit 让模型反复看同一个 batch 并达到 100% accuracy，用来快速发现 forward、loss、backward、optimizer 或标签连接错误。它不证明泛化能力。逐项解释见[测试说明](tests/README.md)。

## 5. Synthetic FP32：第一次完整训练

synthetic 数据有 16 个特征、4 个类别，标签由固定隐藏规则 `argmax(X @ W)` 生成，不是无规律随机标签。模型是 `16 -> 32 -> ReLU -> 4` 的 MLP。

```bash
.venv/bin/python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml

# 有可用 GPU 再运行
.venv/bin/python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cuda.toml
```

validation loss 应从约 1.31 降至 0.326，accuracy 从约 33.9% 升至 89.3%。每次运行产生 `metrics.jsonl`、`summary.json` 和 `last.pt`；位置为 `results/raw/<run>/`。解释见[实验 01](experiments/01_synthetic_baseline.md)。极小 MLP 上 CPU 比 GPU 快并不反常，因为 GPU 固定启动成本大于计算量。

## 6. 梯度累积与 checkpoint 续训

```bash
.venv/bin/python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_accumulation.toml
```

micro-batch 16 累积 4 次再更新，有效 batch 为 64。测试将其与直接 batch 64 的参数更新逐项比较。`--resume` 会恢复 model、optimizer、scheduler、GradScaler、epoch/global step、Python/NumPy/PyTorch/CUDA/DataLoader RNG；字段见 [checkpoint 契约](../docs/checkpoint-contract.md)，解释见[实验 02](experiments/02_accumulation_and_resume.md)。

## 7. 真实 CIFAR-10

CIFAR-10 包含 32×32 RGB 图像和 10 个类别。本实验为学习速度固定取 5,000 train / 1,000 validation，使用小型 CNN，不是追求榜单精度的完整训练。

```bash
.venv/bin/python -m trainscale_training.train --config 01_pytorch_training/configs/cifar10_baseline.toml
```

首次运行下载数据到 `01_pytorch_training/data/`。本机 5 epoch 的 validation accuracy 为 `22.2% -> 42.1%`，最终 validation loss 为 `1.5684`。查看[曲线](results/cifar10_curve.svg)和[实验 05](experiments/05_cifar10_baseline.md)。

## 8. 性能实验

一次只改变一个主要变量：

```bash
# DataLoader workers
.venv/bin/python -m trainscale_training.benchmark_workers --workers 0 1 2 4 --samples 512 --input-dim 128 --batch-size 32 --delay-ms 1 --repeats 3 --output 01_pytorch_training/results/dataloader_workers.json

# FP32 / AMP / compile
.venv/bin/python -m trainscale_training.benchmark_modes --config 01_pytorch_training/configs/cifar10_modes_wsl.toml --output 01_pytorch_training/results/cifar10_modes_wsl.json

# 可靠的 CPU activity profile
.venv/bin/python -m trainscale_training.profile --config 01_pytorch_training/configs/synthetic_cpu.toml --trace 01_pytorch_training/results/raw/synthetic_cpu_profiler_trace.json --summary 01_pytorch_training/results/profiler_summary.json
```

短 synthetic worker 扫描中 workers=0 最快，但真实 JPEG 长 epoch 中多 worker 能隐藏解码等待；完整 CIFAR-10 上 AMP 和 compile 都改善了稳态吞吐。对象复杂度、固定成本和运行长度共同决定结果，所以每个新数据管线和模型都要重新测量。完整解释见[实验索引](experiments/README.md)。

正式 Ubuntu 实验把首 epoch 冷编译与 steady-state 分开，并成功采集真实 CUDA device time；见[实验 04 补充](experiments/04_cont_amp_compile_wsl.md)和[实验 06 补充](experiments/06_cont_cuda_profiler_wsl.md)。

## 9. 怎样读结果

先问正确性，再问性能：loss 是否下降；accuracy 是否高于随机猜测（4 类 25%、10 类 10%）；validation 是否不更新参数；对照是否只改变一个主变量；吞吐是否排除首轮冷启动；AMP/compile 后准确率是否仍合理；环境、失败和适用边界是否记录完整。

`results/summary.json` 是机器可读总表，`experiments/*.md` 才是人读的解释。不要只看一个 samples/s 数字就下结论。

## 10. 01 完成清单

- [x] synthetic dataset 与 single-batch overfit；
- [x] FP32 train/validation 与 CIFAR-10 baseline；
- [x] 配置、seed、设备和结构化日志；
- [x] 完整 checkpoint 与精确 resume；
- [x] AMP、梯度累积、scheduler；
- [x] workers、FP32/AMP/compile 消融；
- [x] CPU/CUDA Profiler 摘要与真实 device-time 验收；
- [x] 10 个 CPU 测试与首个 CPU CI；
- [x] 源码、结果、分析与复现说明均归档在本目录。

继续阅读：[配置](configs/README.md) → [测试](tests/README.md) → [实验](experiments/README.md) → [结果](results/README.md)。
