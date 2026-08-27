# 04 环境与多 GPU 能力门

04 的解析器、配置、公式、聚合和 unavailable 语义可以在 Windows/CPU CI 开发；
真实 NCCL 数据必须来自 Linux/WSL 中至少两张可见 NVIDIA GPU。原生 Windows 不作为
`nccl-tests` 正式环境。

## 本地先做

```powershell
.venv\Scripts\python 04_nccl_benchmark\benchmarks\check_environment.py
.venv\Scripts\python -m pytest -q 04_nccl_benchmark/tests
```

能力探针退出 0 表示探针本身成功；只有输出
`module04_multi_gpu_ready=True` 才表示可以执行真实实验。使用
`--require-multi-gpu` 时，硬件不足会非零退出，适合租卡后的强制门。

## 冻结的外部源码

| 项目 | 值 |
|---|---|
| repository | `https://github.com/NVIDIA/nccl-tests.git` |
| tag | `v2.19.7` |
| commit | `1a65d7f0514b8da6a61ae235d1c5f38549478e29` |

在 Linux GPU 主机执行：

```bash
python 04_nccl_benchmark/benchmarks/build_nccl_tests.py \
  --source-directory /root/autodl-tmp/nccl-tests --execute
```

脚本会验证 checkout 的精确 commit，再运行 `make`。如主机需要显式 `CUDA_HOME`、
`NCCL_HOME` 或编译器设置，先通过环境变量提供并记录，不修改固定源码版本。

本机 Ubuntu 26.04 + GCC 15 + CUDA Toolkit 13.0 的实编译已经验证为不兼容：默认
构建首先触发 glibc `rsqrt/rsqrtf` exception specification 冲突。与 02 的简单 CUDA
探针相同的参数可以继续诊断：

```bash
python 04_nccl_benchmark/benchmarks/build_nccl_tests.py \
  --source-directory /home/shunzi/projects/nccl-tests-v2.19.7 \
  --nvcc-flag=-U_GNU_SOURCE --nvcc-flag=-D_DEFAULT_SOURCE --execute
```

参数通过 NVIDIA 支持的 `NVCC_PREPEND_FLAGS` 环境变量传入，不修改 `nccl-tests`
源码。它能越过第一处错误，但 nccl-tests 更完整的 C++ 依赖随后会触发 GCC 15
`pthread_cond_clockwait/pthread_mutex_clocklock` 声明错误，因此本机没有宣称构建通过。
本阶段不为这一套超前发行版继续替换系统编译器或修改第三方源码；在推荐的 Ubuntu
22.04/24.04 租卡环境先做无额外 flag 构建，只有复现相同诊断时才启用兼容参数。

## 租卡最低要求

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| 主机 | 单机 2×相同 NVIDIA GPU | 单机 4×相同 GPU |
| 系统 | Ubuntu 22.04/24.04 | 与 03 云端路线一致 |
| GPU 显存 | 每卡至少 12 GiB | 每卡 24 GiB |
| 拓扑 | `nvidia-smi topo -m` 可读取 | 至少两个不同 GPU pair 可比较 |
| 软件 | PyTorch CUDA 可用、NCCL 可用、可编译 nccl-tests | 冻结 wheel/driver/toolkit |

2 GPU 可以完成四种 collective 主曲线；4 GPU 才能完成 pair01/pair02/world4 拓扑对照
以及 4-rank DDP bridge。不要租四台彼此独立的单卡实例。

## 付费前/后的门

付费前保证代码、配置和测试已经冻结。开机后依次执行：

```bash
python 04_nccl_benchmark/benchmarks/check_environment.py \
  --output 04_nccl_benchmark/results/raw/rental/environment.json \
  --require-multi-gpu

nvidia-smi -L
nvidia-smi topo -m
```

如果 GPU 数、NCCL 或拓扑不符合租用目标，停止正式实验并先处理实例问题。正式结果
下载并校验哈希后再关机；关闭 SSH/浏览器或 GPU 利用率归零不会自动停止计费。
