# 01 · PyTorch Training 从这里开始

## 一句话说明

01 模块先建立“别人能重建的项目”，再建立“可以证明正确、可以恢复、可以测量的 PyTorch 单卡训练系统”。它的源码、配置、测试、结果与分析都集中在 [`01_pytorch_training/`](../../01_pytorch_training/README.md)。仓库不再使用独立的顶层里程碑编号。

## 先选择运行平台

- **Windows + NVIDIA GPU，准备完成完整路线**：推荐先按[从零搭建 WSL2 + Ubuntu + PyTorch GPU 环境](wsl2-gpu.md)操作。教程会说明为什么选择 WSL、Windows 与 Ubuntu 各运行哪些命令、项目放在哪里，以及六层验收标准。
- **只有 CPU，或暂时只学习基础训练**：可以留在原生 Windows，按[通用环境搭建](environment.md)选择 CPU wheel。
- **原生 Linux**：沿用通用环境中的 Python/uv 规则，并使用 Linux 的 `.venv/bin/...` 命令；不需要 WSL。

如果 Windows pytest 显示 05–07 的 Linux/Gloo 集成项被跳过，按
[Linux/Gloo 集成测试教程](linux-gloo-validation.md)在 WSL2 CPU 环境补齐；这一步不需要 GPU。

不要先在 Windows 中创建 CUDA 环境，再把同一个 `.venv` 复制给 Ubuntu。Windows 与 Linux 应分别创建环境；性能实验的 Linux 仓库应位于 `/home/<用户名>/...`。

进入 02 时也不要预先创建两套 WSL Python 环境：默认 stable 根 `.venv` 先跑[真实 kernel 环境探针](../../02_gpu_kernels/ENVIRONMENT.md)，只有更新驱动后仍失败才建立隔离 nightly 兜底；CUDA Toolkit 则等 CUDA C++ 实验再装。

## 学习顺序

1. 选择原生 Windows、WSL2 Ubuntu 或原生 Linux，并创建各自的项目专属 `.venv`；
2. 选择 CPU 或 CUDA 12.9 PyTorch wheel；
3. 运行 10 个正确性测试；
4. 用 synthetic MLP 学懂一次 train/validation；
5. 验证梯度累积和 checkpoint resume；
6. 用 CIFAR-10 子集和 CNN 验证真实图像训练；
7. 运行 workers、AMP/compile、Profiler 实验；
8. 对照结构化结果阅读分析报告。

## 第一次复现：原生 Windows 基础路线

以下命令均在 Windows 仓库根目录的 PowerShell 中执行。WSL2 用户不要照抄这一组路径，应使用 [WSL2 教程](wsl2-gpu.md)中的 `.venv/bin/...` 命令。CPU 用户：

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\pytest -v
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

NVIDIA GPU 用户不要在这组原生 Windows 命令中直接替换 extra。请先进入 WSL2 Ubuntu，再按 [WSL2 教程](wsl2-gpu.md)创建 cu129 环境并使用 `.venv/bin/...` 运行 GPU 实验。


CPU/GPU wheel、driver、CUDA runtime 和 `nvcc` 的区别见[环境搭建](environment.md)。Windows + NVIDIA GPU 学习者若要完成 compile、Profiler 和后续 CUDA/Triton/NCCL 路线，应从一开始使用[WSL2 Ubuntu 完整教程](wsl2-gpu.md)，而不是等原生 Windows 实验失败后再临时迁移。完整命令、预期指标和知识总结以[模块总指导](../../01_pytorch_training/README.md)为准。

## 你需要能回答的问题

- Dataset、DataLoader、batch、step、epoch 分别是什么？
- 为什么训练需要 backward/optimizer.step，而验证不能更新参数？
- 为什么 single-batch overfit 是正确性测试但不是泛化测试？
- 为什么 checkpoint 不能只保存 model weights？
- micro-batch 16 累积 4 次为什么等效 batch 是 64？
- 为什么小模型 GPU 可能比 CPU 慢？
- 为什么 workers、AMP 和 compile 都不能假定必然加速？
- 为什么“CUDA 训练能跑”不等于 nvcc、CUPTI、Triton 都可用？

回答不清楚时不要进入 02，回到 01 README 按实验顺序重跑并对照源码。

## 完成 01 以后

进入 [02 · GPU Kernels](../../02_gpu_kernels/README.md) 前，先确认你能解释一次训练 step、
checkpoint 为什么保存完整状态，以及 benchmark 为什么需要 warm-up。之后按
[文档总导航](../README.md) 的 01→07 主线推进；等到 04–07 的正式多 GPU 实验时，再使用
[JupyterLab 四卡教程](jupyterlab-4gpu.md)，无需提前为尚未调通的代码付费租卡。
