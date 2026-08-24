# 从零搭建 WSL2 + Ubuntu + PyTorch GPU 环境

这是一篇给第一次接触 WSL 的学习者准备的完整教程。它不要求你先理解虚拟机、CUDA Toolkit 或 Linux 驱动；请按顺序执行，每一步都先看清楚标题中的“在哪里运行”。

完成后，你会得到 Windows 宿主层和 WSL2 Linux 学习层。若两边都运行 Python，它们必须各建自己的 `.venv`；下面的完整 GPU 路线只使用 Ubuntu 根环境：

```text
Windows
├── NVIDIA Windows 驱动                 # 真正管理显卡
├── PowerShell                          # 安装和管理 WSL
└── WSL2
    └── Ubuntu                          # 推荐的 Linux 学习环境
        └── ~/projects/TrainScale_Lab   # 推荐的项目位置
            └── .venv                   # 只属于 Ubuntu 的 Python 环境
```

## 1. 为什么这个项目会用到 WSL2

WSL 是 Windows Subsystem for Linux。WSL2 在 Windows 中运行一个真实 Linux 内核，同时把 Windows NVIDIA 驱动提供的 GPU 能力映射给 Linux。它不是把 Windows 命令“翻译成”Linux 命令，也不是另一份 CUDA 模拟器。

本项目的基础训练、CPU 测试和一部分 CUDA eager/AMP 实验可以在原生 Windows 完成；但后续内容会使用 Linux 生态更成熟的工具链：

- `torch.compile` 的 Inductor/Triton 后端；
- CUDA/Triton 自定义 kernel；
- CUPTI 与 PyTorch Profiler 的 GPU 时间线；
- 后续 NCCL、DDP 和多 GPU 实验。

因此推荐 Windows + NVIDIA GPU 学习者从一开始就建立 WSL2 环境。这样后续章节沿用同一套 Linux 路径和命令，不需要做到一半再迁移。WSL2 不是性能“开关”：它解决的是工具和运行环境问题，是否加速仍要由实验测量。

没有 NVIDIA GPU 也可以使用 WSL2，并安装 CPU 版依赖学习训练循环和测试；只需跳过 GPU、Triton、CUPTI 和 NCCL 实验。

## 2. 为什么推荐 WSL 官方提供的 Ubuntu

新手推荐使用 `wsl --install` 安装的 Ubuntu，也就是 Microsoft 支持的 WSL 发行版入口，而不是自己下载 rootfs、手工制作虚拟机或一开始选择更偏运维/安全用途的发行版。

原因不是“只有 Ubuntu 能运行 PyTorch”，而是：

- Microsoft、NVIDIA 和 PyTorch 的 WSL 教程通常以 Ubuntu 命令为例；
- `apt`、Python 和构建工具的资料多，遇到问题更容易定位；
- 本项目的安装与实验命令已经在 Ubuntu 上验证；
- 新手可以把注意力放在训练系统，而不是发行版差异。

只做 PyTorch/Triton 时，项目已在 Ubuntu 26.04 LTS 实测通过。若从零安装并计划完成 CUDA C++，推荐 Ubuntu 24.04 LTS：CUDA Toolkit 13.0 的官方 Linux 支持表列出了 Ubuntu 24.04/22.04，没有列出 26.04。已经能工作的 26.04 不必重装；它可完成 PyTorch/Triton，本项目也记录了 standalone CUDA smoke 的兼容参数，但那不是 NVIDIA 的正式支持承诺。

项目正式 GPU 基线是 Python 3.11、PyTorch 2.12.1+cu129 和 Triton 3.7.1。这组依赖已在 WSL2 + RTX 5060（SM 12.0）+ Windows driver 610.88 中通过训练、AMP、`torch.compile`、最终 15 项 Triton GPU 测试和 CUDA Profiler 验收；学习者应直接使用锁文件创建环境，不必先复现开发阶段使用旧驱动遇到的问题。

## 3. 安装前需要什么

- Windows 11，或支持 WSL2 的 Windows 10；
- 管理员权限，用于首次启用 WSL；
- 推荐 NVIDIA GPU 和较新的 Windows NVIDIA 驱动；
- 至少预留约 20 GB 磁盘空间。PyTorch CUDA wheel、数据集和编译缓存会持续占用空间；
- 能访问 GitHub、PyTorch wheel 源和 uv 安装地址的网络。

01 不要求安装完整 CUDA Toolkit，也不要求 `nvcc`。PyTorch CUDA wheel 会携带训练需要的 CUDA runtime。进入 02 的自定义 CUDA C++ 编译章节时，教程才会明确要求 Toolkit。

## 4. 安装 WSL2 和 Ubuntu

### 4.1 在“管理员 PowerShell”运行

从开始菜单搜索 PowerShell，右键选择“以管理员身份运行”，然后执行：

```powershell
wsl --install -d Ubuntu
```

如果命令要求重启 Windows，请先重启，再继续。不要在 Ubuntu 终端中运行这条命令。

如果电脑已经安装 WSL，先检查而不是重复安装：

```powershell
wsl --status
wsl --version
wsl --list --verbose
```

`wsl --list --verbose` 的目标状态是 Ubuntu 的 `VERSION` 为 `2`。如果已经有 Ubuntu 但它是 WSL 1，发行版名称以你的输出为准，例如：

```powershell
wsl --set-version Ubuntu 2
```

如果尚未选择发行版，可先查看 Microsoft 当前提供的名称，再安装列表中的 Ubuntu：

```powershell
wsl --list --online
wsl --install -d Ubuntu
```

### 4.2 在“Ubuntu 终端”完成首次初始化

从开始菜单打开 Ubuntu。第一次启动会要求创建 Linux 用户名和密码：

- 这套用户名和密码只属于 Ubuntu，不必与 Windows 相同；
- 输入密码时终端不会显示星号，这是 Linux 的正常行为；
- 以后执行 `sudo` 安装系统包时会使用这个密码。

看到类似 `yourname@computer:~$` 的提示符后，执行：

```bash
cat /etc/os-release
uname -a
pwd
```

`pwd` 此时通常输出 `/home/<你的用户名>`。从这里开始，本教程标为 `bash` 的命令都在 Ubuntu 终端运行。

## 5. GPU 驱动应该安装在哪里

GPU 驱动安装在 Windows，不安装在 Ubuntu。先在普通 Windows PowerShell 中确认：

```powershell
nvidia-smi
```

然后在 Ubuntu 终端再次执行：

```bash
nvidia-smi
```

两个位置都能看到显卡，说明 Windows 驱动已经映射进 WSL。根据 NVIDIA 的 WSL 指南，不要在 Ubuntu 中执行 `apt install nvidia-driver-*` 或安装 Linux display driver；这可能覆盖 WSL 使用的映射库。

`nvidia-smi` 中显示的 `CUDA Version` 表示驱动可支持的最高 CUDA 版本，不表示 Ubuntu 已安装同版本 CUDA Toolkit。稍后用 `torch.version.cuda` 查看 PyTorch wheel 实际使用的 runtime。

如果 Ubuntu 中看不到 GPU，先在普通 PowerShell 执行：

```powershell
wsl --update
wsl --shutdown
```

重新打开 Ubuntu 后再检查。仍失败时，应先更新 Windows NVIDIA 驱动或检查 Windows/WSL 版本，而不是在 Ubuntu 内反复安装驱动。

## 6. 项目应该放在哪里

推荐位置是 Ubuntu 自己的 Linux 文件系统：

```text
/home/<你的用户名>/projects/TrainScale_Lab
```

也就是 `~/projects/TrainScale_Lab`。不要从 `/mnt/c/...`、`/mnt/d/...` 运行性能实验。那些路径是 Windows 磁盘在 WSL 中的挂载点，跨文件系统访问、元数据操作和 DataLoader 多进程行为可能改变性能结果。

### 推荐路线：直接在 Ubuntu 中克隆

在 Ubuntu 终端执行：

```bash
sudo apt-get update
sudo apt-get install -y git curl rsync build-essential

mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Florayuuuu718/TrainScale_Lab.git
cd TrainScale_Lab
pwd
```

最后的路径应该以 `/home/` 开头。

### 已经在 Windows 下载过仓库时

新手仍推荐在 Ubuntu 中重新 `git clone`，因为最容易理解和更新。若还要带入 Windows 中未提交的源码改动，先在 Ubuntu 克隆一份干净仓库，再用 `rsync` 覆盖源码；不要复制 Windows `.venv`、缓存、数据集或大型 trace。Windows 与 Linux 的虚拟环境不能共用。

例如 Windows 仓库位于 `D:\projects\TrainScale_Lab` 时，可在 Ubuntu 中执行：

```bash
mkdir -p ~/projects
git clone https://github.com/Florayuuuu718/TrainScale_Lab.git \
  ~/projects/TrainScale_Lab

rsync -a \
  --exclude=.git --exclude=.venv \
  --exclude=.pytest_cache --exclude=.mypy_cache --exclude=.ruff_cache \
  --exclude=01_pytorch_training/data \
  --exclude=01_pytorch_training/results/raw \
  /mnt/d/projects/TrainScale_Lab/ ~/projects/TrainScale_Lab/

cd ~/projects/TrainScale_Lab
```

安装 Ubuntu 前先确认系统盘或自定义安装位置有足够空间；发行版、项目 `.venv`、uv 缓存、数据集和编译缓存都会增长。发行版具体放在哪个 Windows 盘由学习者按自己的磁盘规划决定，本教程只要求最终实验仓库位于 Ubuntu 的 Linux 文件系统 `/home/<用户名>/...`，不提供发行版迁移流程。

## 7. 安装 uv 和项目依赖

以下命令都在 Ubuntu 项目根目录执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

cd ~/projects/TrainScale_Lab
uv sync --extra cu129 --extra dev --python 3.11
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
uv pip check --python .venv/bin/python
```

`uv sync` 会读取仓库中的 `pyproject.toml` 和 `uv.lock`，下载 Python 3.11，并在项目根目录创建 Linux 专用的 `.venv`。不需要先在系统中手工安装 Python 3.11。紧随其后的探针会在隔离子进程中真正 launch eager、`torch.compile`、Triton Softmax 与 Vector Add；`uv pip check` 再检查 Python 包依赖是否自洽。

如果没有 NVIDIA GPU，改为：

```bash
uv sync --extra cpu --extra dev --python 3.11
```

不要同时选择 `cpu` 和 `cu129`。项目已经把这两个 extra 声明为互斥选项。

如果重新打开终端后找不到 `uv`，可以使用完整路径 `~/.local/bin/uv`，或执行 `source $HOME/.local/bin/env`。

## 8. 从“安装成功”到“实验可用”的分层验收

不要只看到一个版本号就认为全部配置完成。每一层回答不同问题。

### 第 1 层：WSL2 与项目位置

```powershell
# 普通 Windows PowerShell
wsl --list --verbose
```

```bash
# Ubuntu 项目目录
pwd
git status --short
```

确认 Ubuntu 使用 WSL 2，项目路径以 `/home/` 开头。

### 第 2 层：Python 环境与依赖

```bash
.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)"
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

GPU 路线应看到 Python 路径位于当前仓库 `.venv`、PyTorch 带 `+cu129`、runtime 为 `12.9`、CUDA available 为 `True`，以及实际 GPU 名称。CPU 路线的 `None` 和 `False` 是预期结果。

### 第 3 层：代码正确性

```bash
.venv/bin/ruff check .
.venv/bin/mypy 01_pytorch_training/trainscale_training
.venv/bin/python -m pytest -p no:cacheprovider
```

当前基线应有 10 项测试通过。测试通过证明训练组件满足项目的正确性约束，不证明 GPU 性能一定良好。

### 第 4 层：CUDA 真正完成训练

```bash
.venv/bin/python -m trainscale_training.train \
  --config 01_pytorch_training/configs/synthetic_cuda.toml
```

`torch.cuda.is_available()` 只证明 PyTorch 能发现 CUDA；这个命令完成 forward、backward 和 optimizer step，才证明基础 CUDA 训练链可用。

### 第 5 层：先跑小型 Triton/`torch.compile` 探针

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
TRAINSCALE_RUN_SM120_TRITON=1 \
  .venv/bin/python -m pytest -q 02_gpu_kernels/tests/test_triton_ops.py
```

导入 Triton 只证明 Python 包存在。小型探针和最终 15 项测试实际运行 JIT/launch、ragged mask、forward/backward 与错误边界，能在几秒内发现兼容性问题，不必先下载 CIFAR-10 或跑长 benchmark。SM 12.0 GPU 测试要求显式变量，是因为旧驱动下发生过进程级段错误；只有探针先通过才开启。

若探针失败，按固定顺序处理：

1. 在 Windows 更新 NVIDIA 驱动；
2. 在普通 PowerShell 执行 `wsl --shutdown`；
3. 重新打开 Ubuntu，在同一个根 `.venv` 重跑探针；
4. 仍失败时，才按 [02 环境指南](../../02_gpu_kernels/ENVIRONMENT.md)建立仓库外 cu130 nightly 环境做诊断，不改 `uv.lock`。

探针通过后，如需复现 01 的完整 CNN compile 实验，再运行：

```bash
.venv/bin/python -m trainscale_training.benchmark_modes \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --output 01_pytorch_training/results/cifar10_modes_wsl.json
```

首次编译明显慢于稳态是正常 JIT 成本；应分开记录，而不是把它当安装失败。

### 第 6 层：CUDA Profiler

```bash
.venv/bin/python -m trainscale_training.profile \
  --config 01_pytorch_training/configs/cifar10_modes_wsl.toml \
  --trace 01_pytorch_training/results/raw/cifar10_cuda_profiler_wsl_trace.json \
  --summary 01_pytorch_training/results/cifar10_cuda_profiler_wsl.json \
  --wait-steps 2 --warmup-steps 2 --active-steps 10
```

Profiler summary 中实际存在 CUDA device events 和大于零的 device time，才证明 CUPTI trace 成功。`supported_activities()` 声明支持 CUDA、CPU 事件名中出现 `cudnn`，或记录到显存用量，都不能代替 GPU kernel 时间。

项目锁定的 PyTorch 2.12.1+cu129 已在真实 CIFAR-10 workload 上成功采集 device time。若出现 `CUPTI_ERROR_INVALID_DEVICE`，通常说明当前 PyTorch/Kineto/CUDA/CUPTI 与 GPU 的组合不兼容；先确认自己使用了项目锁定版本，而不是把它当成学习流程中的必经步骤。聚合时间的正确读法见[实验 06 补充](../../01_pytorch_training/experiments/06_cont_cuda_profiler_wsl.md)。

## 9. 日常怎样进入项目

以后不需要重新安装。每次从开始菜单打开 Ubuntu，然后执行：

```bash
cd ~/projects/TrainScale_Lab
git status --short
.venv/bin/python --version
```

本教程通常直接写 `.venv/bin/python`，所以不要求激活环境。喜欢激活形式也可以执行：

```bash
source .venv/bin/activate
```

退出虚拟环境使用 `deactivate`，退出 Ubuntu shell 使用 `exit`。从普通 PowerShell 临时进入默认 WSL 发行版可执行 `wsl`。

使用 VS Code 时，推荐从 Ubuntu 项目目录执行 `code .` 并使用 WSL 远程窗口。Windows 资源管理器可以通过 `\\wsl$\Ubuntu\home\<用户名>\projects\TrainScale_Lab` 查看文件，但训练和 benchmark 命令仍应在 Ubuntu 终端执行。

## 10. 常见问题速查

| 现象 | 原因判断 | 处理方式 |
|---|---|---|
| `wsl` 不是命令 | WSL 未启用或 Windows 版本过旧 | 回到管理员 PowerShell 执行 `wsl --install`，按提示重启 |
| Ubuntu 显示 WSL 1 | 发行版不是 WSL2 | 普通 PowerShell 执行 `wsl --set-version <发行版名> 2` |
| Ubuntu 中 `nvidia-smi` 失败 | Windows 驱动或 WSL 映射未就绪 | 先确认 Windows `nvidia-smi`，再 `wsl --update`、`wsl --shutdown` |
| `sudo` 输入密码没有字符 | Linux 默认不回显密码 | 正常输入完成后按 Enter |
| `uv: command not found` | 新 PATH 尚未载入 | `source "$HOME/.local/bin/env"` 或重开 Ubuntu |
| Python 路径含 `Scripts` | 错用了 Windows `.venv` | 在 Ubuntu 仓库根目录重新 `uv sync`，Linux 环境使用 `.venv/bin` |
| 项目路径以 `/mnt/` 开头 | 正从 Windows 文件系统运行 | 在 `~/projects` 重新 clone 或复制源码 |
| 首次 compile 很慢 | Inductor 正在生成和编译代码 | 分开记录首 epoch 与 steady-state，不要立即判断失败 |
| Triton launch 段错误/无 Python traceback | driver、SM 架构与 JIT 加载链不兼容 | 用隔离探针确认；先更新 Windows 驱动并 `wsl --shutdown`，仍失败才建 nightly 诊断环境 |
| CUDA 13.0 在 Ubuntu 26.04 报 `rsqrt/rsqrtf` 声明冲突 | Toolkit 13.0 未正式支持该发行版/glibc 组合 | 新安装用 Ubuntu 24.04；已有 26.04 按 [CUDA smoke 说明](../../02_gpu_kernels/cuda/README.md)使用已验证兼容参数，不修改系统头文件 |
| `CUPTI_ERROR_INVALID_DEVICE` | 当前用户态工具链与 GPU 不兼容 | 确认使用锁定的 2.12.1+cu129；不要在 WSL 安装 Linux display driver |
| CIFAR-10 首次运行很久 | 正在下载、解压并创建 workers | 等待完成；之后数据会复用，不把 `data/` 提交 Git |

## 11. 学完本教程应该能解释什么

- WSL2 为什么比原生 Windows 更适合作为后续 CUDA/Triton/NCCL 学习环境；
- Windows 驱动、WSL GPU 映射、PyTorch CUDA runtime、CUDA Toolkit 分别属于哪一层；
- 为什么推荐 Ubuntu，以及为什么项目应位于 `/home/...` 而不是 `/mnt/...`；
- 为什么 Windows 与 Ubuntu 必须各有自己的 `.venv`；
- 为什么项目默认只维护一个 WSL stable `.venv`，而 nightly 和 Toolkit 都是按失败/章节引入；
- 为什么 `nvidia-smi`、`torch.cuda.is_available()`、CUDA 训练、Triton import、compile 和 CUPTI trace 是六层不同的验收；
- 出现问题时应该定位哪一层，而不是无目的地重装所有 CUDA 组件。

完成前五层后，可以回到 [01 · PyTorch Training](../../01_pytorch_training/README.md) 按实验顺序学习；进入算子实验先读 [02 环境指南](../../02_gpu_kernels/ENVIRONMENT.md)。需要 compile 时阅读[实验 04 补充](../../01_pytorch_training/experiments/04_cont_amp_compile_wsl.md)，需要 Profiler 时阅读[实验 06 补充](../../01_pytorch_training/experiments/06_cont_cuda_profiler_wsl.md)。

## 参考资料

- [Microsoft：安装 WSL](https://learn.microsoft.com/windows/wsl/install)
- [Microsoft：WSL 基本命令](https://learn.microsoft.com/windows/wsl/basic-commands)
- [NVIDIA：CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA：CUDA 13.0 Linux Installation Guide / OS support matrix](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html)
- [PyTorch：torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [PyTorch：Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Triton：安装说明](https://triton-lang.org/main/getting-started/installation.html)
- [uv：安装说明](https://docs.astral.sh/uv/getting-started/installation/)
