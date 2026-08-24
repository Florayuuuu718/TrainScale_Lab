# 02 环境教程：先用稳定环境，探针失败才隔离

## 先说结论

绝大多数学习者只需要仓库根目录的一套 `.venv`。本项目锁定的
`Python 3.11 + PyTorch 2.12.1+cu129 + Triton 3.7.1` 已在 RTX 5060 Laptop
（SM 12.0）和 Windows driver 610.88 上通过真实 kernel 探针、最终 15 项 GPU
correctness tests 和正式性能实验。

不要因为显卡架构较新就直接安装 nightly，也不要在根 `.venv` 中反复升级。
正确顺序是：

```text
创建稳定 .venv
    ↓
运行独立子进程环境探针
    ├── 全部通过 → 继续使用稳定 .venv
    └── Triton 失败 → 更新 Windows driver，重启 WSL 后重试
                           ├── 通过 → 仍使用稳定 .venv
                           └── 仍失败 → 才创建仓库外 cu130 诊断环境
```

CUDA Toolkit 是另一条分支：Triton 不需要系统 `nvcc`；只有 CUDA C++
实验才需要 Toolkit。

## 1. 三种用途，不是三套都要安装

| 用途 | 应使用什么 | 是否默认安装 |
|---|---|---|
| 01、PyTorch eager/compile/Profiler、02 Triton | 仓库根 `.venv` | 是 |
| 02 CUDA C++ | 同一学习路线再加系统 CUDA Toolkit | 到 CUDA C++ 时再装 |
| 稳定 Triton 探针仍失败 | `~/.venvs/trainscale-02-cu130` | 否，仅诊断/兼容 |

隔离 nightly 环境放在仓库外，因此不会被误当作项目锁定环境，也不会改写
`pyproject.toml`、`uv.lock` 或 01 的历史结果。

## 2. 六个容易混淆的版本

| 名称 | 本机实测值 | 它回答什么 |
|---|---|---|
| GPU 架构 | SM 12.0 | 硬件能执行哪一代机器代码 |
| Windows driver | 610.88 | 宿主驱动能否管理 GPU、加载 CUDA 代码 |
| `nvidia-smi` CUDA | 13.3 | driver 可支持的最高用户态 CUDA，不是已安装 Toolkit |
| PyTorch wheel runtime | 12.9 或隔离环境 13.0 | 这个 PyTorch wheel 携带和使用的 CUDA runtime |
| 系统 Toolkit / `nvcc` | 13.0.88 | 编译 `.cu` 源码的开发工具 |
| Triton | 3.7.1 或隔离环境 3.8.0 nightly | 从 Python DSL 生成 GPU kernel 的编译器 |

这些层可以有不同版本号。看到 `nvidia-smi` 写 CUDA 13.3，不能推断
`nvcc` 已安装，更不能推断 PyTorch wheel 是 cu133。

## 3. 新安装优先选 Ubuntu 24.04 LTS

只做 PyTorch/Triton 时，本项目在 Ubuntu 26.04 也已实测通过。但 CUDA
Toolkit 会依赖系统 GCC 和 glibc 头文件。CUDA 13.0 官方支持表列出 Ubuntu
24.04/22.04，没有列出 Ubuntu 26.04。因此从零安装、并计划完成 CUDA C++
的学习者，推荐在 PowerShell 查看名称后选择 Ubuntu 24.04：

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
```

发行版名称以你电脑的实际列表为准。已经有能工作的 Ubuntu 26.04 不必立刻
删除；先阅读第 7 节的已验证边界。

## 4. 默认稳定路线

以下命令在 Ubuntu 的仓库根目录运行，正式性能实验的路径必须以 `/home/`
开头：

```bash
cd ~/projects/TrainScale_Lab
uv sync --extra cu129 --extra dev --python 3.11
uv pip check --python .venv/bin/python
```

然后运行真正会 launch kernel 的隔离探针：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
echo $?
```

探针分别启动子进程检查：

1. CUDA eager；
2. `torch.compile` 的冷、热调用；
3. 单输入 load 的 Triton Softmax；
4. 双输入 load 的项目 Vector Add。

退出码 `0` 且 `all_required_checks_passed=True` 才算通过。只执行
`import triton` 不算，因为历史问题发生在编译后的加载/launch 阶段。

SM 12.0 的 pytest 默认保护性 skip。探针通过后，才显式运行真实测试：

```bash
TRAINSCALE_RUN_SM120_TRITON=1 \
PYTHONPATH=02_gpu_kernels \
.venv/bin/python -m pytest -p no:cacheprovider -q -rs \
  02_gpu_kernels/tests/test_triton_ops.py
```

本机最终稳定环境实际得到 `15 passed`。不设置 opt-in 时，GPU 文件会被保护性
skip；这是为了避免在未知 SM 12.0 driver 上直接触发历史段错误，不代表测试已经
执行。

## 5. 稳定探针失败时怎么做

若 CUDA eager 也失败，先检查 Windows/WSL GPU 映射；不要调整 Triton：

```powershell
# 普通 Windows PowerShell
nvidia-smi
wsl --update
wsl --shutdown
```

若 eager 通过、Triton 子进程以 `139` 或 `SIGSEGV` 失败，先更新 Windows
NVIDIA driver，执行 `wsl --shutdown` 后重试同一探针。项目历史快照中，driver
577.05 会崩溃；升级到 610.88 后，同一稳定 Python 栈通过。610.88 只是本项目
实测版本，不是对所有 SM 12.0 GPU 的最低版本承诺。

不要这样“修复”：

- 在 WSL 安装 `nvidia-driver-*`；WSL 使用 Windows driver 映射；
- 把 SM 12.0 伪装成旧架构；
- 在根 `.venv` 中手工覆盖任意 torch/Triton；
- 看到一个进程崩溃后继续在同一 pytest 进程批量 launch；
- 把 import 成功当作 kernel 可执行。

## 6. 只有仍失败时才创建 cu130 隔离环境

以下是 2026-08-24 本机实际验收过的精确构建，不是“安装最新版”：

```bash
~/.local/bin/uv venv --python 3.11 --seed \
  ~/.venvs/trainscale-02-cu130

~/.venvs/trainscale-02-cu130/bin/python -m pip install \
  'torch==2.15.0.dev20260823+cu130' \
  'triton==3.8.0+git3f6e4113' \
  --index-url https://download.pytorch.org/whl/nightly/cu130

~/.venvs/trainscale-02-cu130/bin/python -m pip install \
  numpy==2.4.6 pytest==9.0.2

~/.venvs/trainscale-02-cu130/bin/python -m pip check
```

用这个解释器重复探针和测试：

```bash
~/.venvs/trainscale-02-cu130/bin/python \
  02_gpu_kernels/benchmarks/check_environment.py

TRAINSCALE_RUN_SM120_TRITON=1 \
PYTHONPATH=02_gpu_kernels \
~/.venvs/trainscale-02-cu130/bin/python -m pytest -p no:cacheprovider -q \
  02_gpu_kernels/tests/test_triton_ops.py
```

nightly 环境在早期 13 项测试快照中同样通过。此后新增的 Attention head-dim
覆盖只在最终 stable 环境验收；因为 stable 本来就通过，不能把结论写成
“nightly 修复了 SM 12.0”。nightly 只是一条隔离的对照/诊断路线。

## 7. CUDA C++：Toolkit 是单独的能力门

在 WSL 中只安装 Toolkit，不安装 Linux display driver。按 NVIDIA WSL/CUDA
安装文档配置仓库后，安装 Toolkit 13.0：

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

nvcc --version
nvcc --list-gpu-code | grep -Fx sm_120
```

版本输出和 `sm_120` 只证明编译器存在。还必须编译并运行真实程序：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py \
  --require-nvcc
```

### 已有 Ubuntu 26.04 + CUDA 13.0 时

本机 glibc 2.43 与 CUDA 13.0 的 `rsqrt/rsqrtf` 声明冲突，默认 smoke 会在
编译阶段失败。以下不修改系统头文件的参数已让 standalone smoke 通过：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py \
  --require-nvcc \
  --nvcc-flag=-U_GNU_SOURCE \
  --nvcc-flag=-D_DEFAULT_SOURCE
```

等价的手工命令见 [`cuda/README.md`](cuda/README.md)。这是一条已实测的
workaround，不会把 CUDA 13.0 与 Ubuntu 26.04 变成官方支持组合。它只证明
standalone `.cu` 的编译、driver load、launch、同步、拷回和数值检查，不证明
PyTorch C++ extension ABI 已通过。正式 standalone CUDA Vector Add/Softmax 已在后续实验完成，但它们刻意不与 PyTorch wheel 进程内混合，因此仍不构成 extension ABI 证明。

## 8. 正式性能实验

`/mnt/c`、`/mnt/d` 只用于快速 correctness 排查；正式数字必须在 WSL Linux
文件系统的 `/home/...` 仓库中采集：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full \
  --samples 21 \
  --warmup 10 \
  --output 02_gpu_kernels/results/triton_comparison_sm120_cu129.json

.venv/bin/python 02_gpu_kernels/benchmarks/profile_triton_comparison.py \
  --iterations 20 \
  --output 02_gpu_kernels/results/triton_profiler_sm120_cu129.json
```

correctness 通过、环境可执行和性能实验完成是三个不同结论。nightly 与稳定
环境、不同 driver、不同 GPU 或 `/mnt/...` 路径的绝对延迟不得混在一张 speedup
表中。

## 9. 本机最终结果

完整机器可读证据见：

- [`sm120_environment_validation.json`](results/sm120_environment_validation.json)；
- [`triton_comparison_sm120_cu129.json`](results/triton_comparison_sm120_cu129.json)；
- [`triton_profiler_sm120_cu129.json`](results/triton_profiler_sm120_cu129.json)；
- [`module02_acceptance_sm120.json`](results/module02_acceptance_sm120.json)。

最终环境决策是：默认稳定 `.venv` 已足够完成当前 02 Triton 实验；cu130
nightly 只保留为探针失败时的隔离路线；Toolkit 只在 CUDA C++ 分支加入。

## 参考资料

- [NVIDIA：CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA：CUDA 13.0 Linux Installation Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html)
- [PyTorch SM 12.0 Triton issue #176426](https://github.com/pytorch/pytorch/issues/176426)
- [Triton installation](https://triton-lang.org/main/getting-started/installation.html)
