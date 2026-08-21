# M0/M1 环境搭建：Python、PyTorch 与 CUDA wheel

本阶段的目标不是“让某台机器碰巧能运行”，而是让新机器能够根据锁文件重建同一套环境，并能明确判断当前运行的是 CPU 还是 GPU 版本。

## 1. 环境与缓存不是一回事

每个项目都应该有独立虚拟环境。`trainscale-lab` 的依赖安装在自己的 `.venv` 中，不读取其他项目的 `site-packages`。因此，即使另一项目已经安装 PyTorch，本项目仍需要执行一次 `uv sync`。

但这不等于每个项目都必须重新从网络下载相同 wheel：uv 默认使用用户级共享下载缓存，并可通过复制或硬链接将缓存内容安装到不同虚拟环境。可用以下命令查看缓存位置：

```powershell
uv cache dir
```

正常开发无需设置 `UV_CACHE_DIR`。只有在权限受限、离线归档或明确要求缓存也随项目隔离时，才使用项目内缓存：

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
```

项目内缓存会提高隔离程度，但不同项目之间不能复用下载，会占用更多磁盘和网络流量。无论缓存放在哪里，真正的依赖环境仍由 `.venv` 和 `uv.lock` 隔离。

## 2. 冻结版本

| 层 | M0/M1 基线 | 作用 |
|---|---|---|
| Python | 3.11 | 项目解释器，由 `.python-version` 与 `pyproject.toml` 约束 |
| PyTorch | 2.11.0 | CPU/GPU 共用的 API 基线 |
| CPU wheel | `2.11.0+cpu` | CPU CI 和无 NVIDIA GPU 的学习路径 |
| GPU wheel | `2.11.0+cu128` | 本地 NVIDIA GPU 训练，包含 CUDA 12.8 runtime |
| NVIDIA driver | 本机 577.05 | 由操作系统安装，为 CUDA 程序访问 GPU |
| CUDA Toolkit / `nvcc` | M1 不需要 | 到 M2 编译自定义 CUDA C++ 时再安装 |

`nvidia-smi` 显示的 `CUDA Version` 是驱动能支持的最高 CUDA 版本，不代表系统已安装相同版本的 CUDA Toolkit。`torch.version.cuda` 才是当前 PyTorch wheel 使用的 CUDA runtime 版本。

## 3. 创建 Python 3.11 虚拟环境

在仓库根目录执行：

```powershell
uv venv --python 3.11 .venv
```

激活不是必需的；文档使用 `.venv\Scripts\...` 可以明确调用本项目解释器。需要交互式激活时执行：

```powershell
.venv\Scripts\Activate.ps1
```

检查解释器边界：

```powershell
.venv\Scripts\python -c "import sys; print(sys.executable); print(sys.version)"
```

输出路径必须位于本仓库的 `.venv`，Python 必须为 3.11。

## 4. 选择且只选择一种 PyTorch wheel

### CPU 学习路径与 CI

```powershell
uv sync --extra cpu --extra dev
```

验证：

```powershell
.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

预期为 `2.11.0+cpu`、`None`、`False`。这不是错误，而是 CPU 环境的验收结果。

### NVIDIA GPU 学习路径

```powershell
uv sync --extra cu128 --extra dev
```

该命令会下载体积较大的 PyTorch/CUDA wheel，并将同一个 `.venv` 从 CPU 变体切换为 CUDA 变体。CUDA runtime、cuDNN、cuBLAS 等运行依赖由 wheel 提供；M1 不需要单独安装 CUDA Toolkit 或 `nvcc`。

验证：

```powershell
.venv\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

本项目预期为 `2.11.0+cu128`、`12.8`、`True` 和实际 NVIDIA GPU 名称。

`cpu` 与 `cu128` extras 在 `pyproject.toml` 中显式互斥，禁止同时安装：

```powershell
# 错误示例：不要执行
uv sync --extra cpu --extra cu128 --extra dev
```

切回 CPU 版本时重新执行 `uv sync --extra cpu --extra dev`。远端 CPU CI 始终显式选择 `cpu`，不会被本机选择影响。

## 5. 驱动、runtime 和 nvcc 的检查方法

```powershell
# 操作系统中的 NVIDIA 驱动，以及驱动最高支持的 CUDA 版本
nvidia-smi

# 系统是否安装 CUDA Toolkit 编译器；M1 找不到该命令是正常状态
nvcc --version

# wheel 自带的 runtime 与最终可用性
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

只有进入 M2、开始编译 `.cu` 文件或 PyTorch CUDA C++ extension 时，才把 CUDA Toolkit 12.8 与 `nvcc` 加入环境要求。安装 Toolkit 不会替代 NVIDIA driver，也不会改变当前虚拟环境中的 PyTorch wheel。

## 6. M0/M1 环境验收

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\pytest
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

GPU wheel 安装后额外运行：

```powershell
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cuda.toml
```

验收日志至少记录 Python、PyTorch、`torch.version.cuda`、driver、GPU 名称、`torch.cuda.is_available()` 和实际 smoke run 结果。不要把 `.venv`、下载缓存或 CUDA wheel 提交到 Git；提交 `pyproject.toml` 与 `uv.lock` 即可复现依赖选择。
