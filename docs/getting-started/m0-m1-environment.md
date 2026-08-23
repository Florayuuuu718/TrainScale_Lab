# M0/M1 环境搭建：Python、PyTorch 与 CUDA wheel

本页解释环境分层；需要完成 GPU、AMP、compile、Profiler 和后续 Triton/NCCL 的 Windows + NVIDIA 学习者，请直接使用 [WSL2 Ubuntu 完整教程](wsl2-gpu.md)。原生 Windows 只作为 CPU 或基础训练路线，不是完整 GPU 路线。

## 1. 虚拟环境与缓存为什么要分开

每个项目都应有独立 `.venv`，避免一个项目升级依赖后破坏另一个项目。uv 的用户级下载缓存可以在多个虚拟环境间复用 wheel，因此“环境隔离”不等于“每个项目重复联网下载”。

```powershell
uv cache dir
uv venv --python 3.11 .venv
```

`.venv` 属于操作系统：Windows 使用 `.venv\Scripts\...`，Linux/Ubuntu 使用 `.venv/bin/...`，两者不能复制或共用。

## 2. 锁定版本解决什么问题

| 层 | M0/M1 基线 | 作用 |
|---|---|---|
| Python | 3.11 | 由 `.python-version` 与 `pyproject.toml` 约束 |
| PyTorch | 2.12.1 | CPU/GPU 共用 API 基线 |
| CPU wheel | `2.12.1+cpu` | CPU CI 和无 NVIDIA GPU 路线 |
| GPU wheel | `2.12.1+cu129` | Ubuntu NVIDIA 路线，含 CUDA 12.9 runtime |
| Triton | 3.7.1 | Linux `torch.compile` 与后续 kernel 学习 |
| NVIDIA driver | 由 Windows/宿主系统安装 | 让 CUDA 程序访问 GPU |
| CUDA Toolkit / `nvcc` | M1 不需要 | M2 编译 CUDA C++ 时再安装 |

锁文件固定 Python 包和 wheel 解析，不能固定 GPU、driver、操作系统与 workload。因此安装后仍要分层验收。`nvidia-smi` 的 `CUDA Version` 是驱动可支持的最高版本，不表示安装了同版本 Toolkit；`torch.version.cuda` 才是 PyTorch wheel 使用的 runtime。

## 3. CPU 路线：原生 Windows 或 Linux

Windows PowerShell：

```powershell
uv sync --extra cpu --extra dev --python 3.11
.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
.venv\Scripts\python -m pytest -v
```

Ubuntu/Linux 把解释器路径改为 `.venv/bin/python`。预期版本为 `2.12.1+cpu`，CUDA runtime 为 `None`，CUDA available 为 `False`；这是 CPU wheel 的正确验收结果。

## 4. GPU 路线：Windows + NVIDIA 从 Ubuntu 开始

先按 [WSL2 Ubuntu 教程](wsl2-gpu.md)安装发行版、验证 GPU 映射，并把项目放入 `/home/<用户名>/...`。然后在 Ubuntu 项目根目录执行：

```bash
uv sync --extra cu129 --extra dev --python 3.11
.venv/bin/python -c "import torch, triton; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(triton.__version__)"
```

预期为 PyTorch `2.12.1+cu129`、runtime `12.9`、CUDA `True`、实际 GPU 名称和 Triton `3.7.1`。`cpu` 与 `cu129` extras 显式互斥，不能同时安装：

```bash
# 错误示例：不要执行
uv sync --extra cpu --extra cu129 --extra dev
```

## 5. 为什么 M1 不要求 nvcc

PyTorch CUDA wheel 已包含训练所需的 CUDA runtime、cuDNN、cuBLAS 等用户态库；M1 只调用已编译的库。`nvcc` 用于编译 `.cu` 文件和 CUDA C++ extension，到 M2 需要自定义 CUDA 源码时才安装。

这些检查回答不同问题：

- `nvidia-smi`：驱动和 GPU 映射是否可见；
- `torch.cuda.is_available()`：PyTorch runtime 能否初始化 CUDA；
- forward/backward：训练链是否真的执行；
- Triton import 与 compile benchmark：Inductor/Triton 完整路径是否可用；
- Profiler 正 device time：Kineto/CUPTI 是否真的采到 GPU activity。

## 6. 验收与常见误区

CPU 路线至少运行：

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\mypy 01_pytorch_training/trainscale_training
.venv\Scripts\python -m pytest -v
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

完整 GPU 路线按 WSL2 教程逐层验收。`+cpu / None / False` 表示 CPU wheel；CUDA 训练可用但 compile/Profiler 不可用时，应检查 Triton 或 CUPTI 用户态链路；首次 compile 慢通常是冷编译成本。原生 Windows 完成 eager 训练，不等于已满足本项目完整 Linux GPU 路线。

继续学习：[WSL2 Ubuntu 完整教程](wsl2-gpu.md) → [01 · PyTorch Training](../../01_pytorch_training/README.md)。
