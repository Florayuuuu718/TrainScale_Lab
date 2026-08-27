# TrainScale Lab

[English](README.md) | **简体中文**

> 从一个可验证的 PyTorch 训练循环出发，逐层构建 GPU 算子、分布式训练、集合通信与迷你训练引擎。

TrainScale Lab 是一个面向 **ML Systems / AI Infrastructure / Distributed Training** 初学者的实践型开源项目。它不是一份链接合集，也不是对现有框架的简单封装；每一阶段都要求你先实现最小版本，再测量、解释瓶颈并完成一次可复现的优化。

## 快速导航

[新手入口](#新手从这里开始) · [环境与运行](#01-快速开始) · [学习路线](#学习路线) · [分阶段任务](#分阶段实践任务) · [可复现约定](#可复现性约定)

| 跳转位置 | 内容 |
|---|---|
| [01 · PyTorch Training](01_pytorch_training/README.md) | 01 模块完整教学与复现顺序 |
| [02 · GPU Kernels](02_gpu_kernels/README.md) | 可运行 Triton 算子、验收清单与九项实验报告 |
| [03 · Distributed Training](03_distributed_training/README.md) | 可运行 Gloo/DDP 教程、云端四卡流程与八项实验报告 |
| [训练源码](01_pytorch_training/trainscale_training) | 数据、模型、引擎、checkpoint、benchmark 与 profiler |
| [实验配置](01_pytorch_training/configs/README.md) | TOML 实验配方 |
| [正确性测试](01_pytorch_training/tests/README.md) | 10 个测试的逐项解释 |
| [实验报告](01_pytorch_training/experiments/README.md) | 成功实验、理论分析与排障提示 |
| [结果目录](01_pytorch_training/results/README.md) | 精简 JSON 总表与 SVG 曲线 |
| [文档导航](docs/README.md) | 概念、环境和仓库基建 |
| [01 验收清单](docs/01-issues.md) | 01 模块的准备工作与实现状态 |

## 新手从这里开始

仓库目前已完成 01 模块及其仓库基建：从 synthetic MLP 正确性基线扩展到 CIFAR-10 CNN、checkpoint/resume、AMP、梯度累积、Profiler 和性能消融。目标是验证训练系统与实验方法，不是追求 CIFAR-10 榜单精度。

请按顺序阅读和操作：

1. [文档总导航](docs/README.md)
2. [01 模块从这里开始](docs/getting-started/README.md)
3. [PyTorch 训练基础概念](docs/concepts/pytorch-training-basics.md)
4. [01 · PyTorch Training 完整复现](01_pytorch_training/README.md)

如果你使用 Windows + NVIDIA GPU，并准备完成后续 compile、Profiler、CUDA/Triton 与 NCCL 路线，请在运行实验前先完成[WSL2 + 官方 Ubuntu + PyTorch GPU 从零教程](docs/getting-started/wsl2-gpu.md)。教程明确标注每条命令应在管理员 PowerShell、普通 PowerShell 还是 Ubuntu 终端执行，并解释项目为什么应放在 Ubuntu 的 `/home/<用户名>/projects/` 中。

## 01 快速开始

第一阶段冻结版本为 Python 3.11、PyTorch 2.12.1，NVIDIA GPU 主环境使用 CUDA 12.9；CPU CI 使用同一 PyTorch 2.12.1 API 线的 CPU wheel。

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\ruff check .
.venv\Scripts\pytest
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

完整 NVIDIA 路线不要在上面的原生 Windows 环境中直接换 extra；请进入 WSL2 Ubuntu 后执行 `uv sync --extra cu129 --extra dev --python 3.11`，并使用 `.venv/bin/...`。CPU 与 CUDA extra 被显式设为互斥，避免混装。大型 checkpoint/trace、数据集、本地环境和缓存不会进入 Git；精简 JSON/SVG 结果会连同分析一起保留。

虚拟环境隔离、uv 缓存复用、CPU/GPU wheel 选择、CUDA 验证以及何时需要 `nvcc`，详见 [环境搭建指导](docs/getting-started/environment.md)。

01 与 02 均已通过 Windows CPU 和 Ubuntu GPU 本地验收并封存。RTX 5060（SM 12.0）上的 02 默认 cu129/Triton 环境最终通过 15 项 GPU 测试；归档证据覆盖 14 组 PyTorch/Triton forward case、41 条 PyTorch/Triton/CUDA kernel 路径、LayerNorm 前反向、MatMul 有限调参与代表性 Profiler。[02 机器可读验收记录](02_gpu_kernels/results/module02_acceptance_sm120.json)把静态检查与真实 GPU 执行分开保存。

02 默认继续使用仓库根 `.venv`，大型实验前先运行[崩溃隔离环境探针](02_gpu_kernels/ENVIRONMENT.md)。若 Triton 失败，先更新 Windows NVIDIA 驱动并重启 WSL；同一探针仍失败时，才建立文档中的仓库外 cu130 nightly 诊断环境。系统 CUDA Toolkit 只在 CUDA C++ 章节需要，PyTorch/Triton 主线不要求预装。

03 已完成：本地 2/4 rank CPU/Gloo 正确性、scaling、Profiler 和 RTX 5060 单 GPU NCCL 基线全部通过；同一冻结配置随后在 AutoDL 单机 4×RTX 4090D 上运行三次，归档了真实 1/2/4 GPU strong/weak 结果、拓扑、原始重复值、中位数汇总、下载哈希和关机止费流程。只有未租用的 8 GPU case 继续诚实记录为 `unavailable`。

## 你会学到什么

完成主线后，你将能够：

- 独立组织 PyTorch 的训练、验证、断点续训和性能分析流程；
- 用 PyTorch、CUDA 与 Triton 实现并比较典型 GPU 算子；
- 解释 DDP 中的进程、rank、数据切分和梯度同步；
- 使用 NCCL 测量 collective 的延迟、算法带宽与总线带宽；
- 从零实现 Naive AllReduce 与 Ring AllReduce，并分析通信复杂度；
- 实现一个支持 AMP、梯度累积、梯度分桶及通信计算重叠的迷你训练引擎；
- 根据显存、吞吐、扩展效率和 profile 证据选择 DDP、FSDP2 或 TP。

## 学习路线

```text
训练正确性
   ↓
单卡性能与 Profiler
   ↓
CUDA / Triton 算子
   ↓
DDP 多 GPU 扩展
   ↓
NCCL 通信分析
   ↓
手写 Ring AllReduce
   ↓
迷你分布式训练引擎
   ↓
FSDP2 / Tensor Parallel
```

| 阶段 | 要构建的东西 | 核心问题 | 最终证据 |
|---|---|---|---|
| [01](01_pytorch_training/README.md) | PyTorch 单卡训练框架 | 一次可靠训练需要哪些系统组件？ | loss/accuracy、吞吐、显存、断点恢复一致性 |
| [02](02_gpu_kernels/README.md) | GPU Kernel Lab | 算子为什么快或慢？ | 正确性、延迟、带宽/TFLOPS、profiler 证据 |
| [03](03_distributed_training/README.md) | Distributed Training Lab | 分布式训练为什么不能线性加速？ | DDP 正确性、CPU scaling、本地 NCCL 与云端 1/2/4 GPU 实证 |
| 04 | NCCL Performance Lab | 通信何时受延迟或带宽限制？ | message size–bandwidth 曲线 |
| 05 | TinyCollective | AllReduce 内部究竟发生了什么？ | Naive/Ring/NCCL 对照实验 |
| 06 | Mini Training Engine | 如何隐藏通信并控制显存？ | 消融实验、timeline、吞吐提升 |
| 07 | FSDP2 / TP 扩展 | 模型放不下一张卡时如何切分？ | 峰值显存、正确性与扩展效率 |

## 仓库结构

七个模块目录均已建立；01–03 已封存。03 同时完成本地正确性和云端真实 1/2/4 GPU scaling，8 GPU 保留为明确的可选边界。各后续目录的 README 明确记录范围和当前状态。

```text
trainscale-lab/
├── 01_pytorch_training/       # 可复现的单卡训练基线
├── 02_gpu_kernels/            # PyTorch / CUDA / Triton 算子对照
├── 03_distributed_training/   # DDP 与 scaling benchmark
├── 04_nccl_benchmark/         # collective 通信测试与分析
├── 05_tiny_collective/        # Naive 与 Ring AllReduce
├── 06_training_engine/        # 最终的迷你分布式训练引擎
├── 07_parallelism/            # FSDP2、TP 与组合并行
├── benchmarks/                # 统一 benchmark 入口与结果 schema
├── docs/                      # 跨模块概念、环境与实验记录
└── README.md
```

`01`–`07` 是唯一的模块编号。模块内部可以按实验编号继续细分，但不再使用另一套顶层里程碑编号。每个模块都应包含自己的 `README.md`、环境说明、最小运行命令、测试和实验报告。

## 如何学习，而不是只把代码跑起来

每个实验都遵循同一个闭环：

1. **预测**：运行前写下性能瓶颈和预期现象。
2. **基线**：先实现最简单、结果正确的版本。
3. **测量**：固定软硬件、数据、warm-up 和迭代次数。
4. **解释**：用 profiler 或通信指标定位原因。
5. **只改一个变量**：例如 precision、batch size、bucket size 或 kernel tile。
6. **验证**：重新检查数值正确性和训练收敛，不能只看速度。
7. **记录**：提交配置、原始数据、图表和结论，使他人能够复现。

推荐为每项优化保留如下实验表：

| experiment | hardware | precision | batch size | throughput | peak memory | correctness |
|---|---|---|---:|---:|---:|---|
| baseline | 待实测 | FP32 | 待实测 | 待实测 | 待实测 | 待验证 |
| optimization-A | 同上 | 待定 | 同上 | 待实测 | 待实测 | 待验证 |

## 分阶段实践任务

### 01 · PyTorch Training

- 从 `Dataset → DataLoader → forward → loss → backward → optimizer.step` 写起；
- 加入 validation、scheduler、checkpoint/resume、随机种子和日志；
- 对照 FP32、AMP、梯度累积与 `torch.compile`；
- 用 Profiler 判断瓶颈在数据、CPU launch、GPU compute 还是显存。

建议从 CIFAR-10 或可离线生成的 synthetic dataset 开始。验收重点不是最高精度，而是同配置可复现、断点恢复正确、测量方法可信。

### 02 · GPU Kernels

按 `Vector Add → ReLU → Softmax → LayerNorm → MatMul → Attention` 推进。每个算子至少包含：

- PyTorch reference；
- 朴素实现；
- 优化后的 CUDA 或 Triton 实现；
- `torch.testing.assert_close` 数值检查；
- 多种 shape/dtype 下的 benchmark。

学习 thread/block/warp、合并访存、shared memory、register pressure、occupancy 和 roofline 时，都要对应到一次实测现象。

### 03 · Distributed Training

- 先用 Gloo + CPU 理解多进程语义，再用 NCCL + GPU 做性能实验；
- 实现 `init_process_group`、`DistributedSampler`、DDP、分布式 checkpoint；
- 对比单卡与 2/4/8 卡吞吐，计算 speedup 与 scaling efficiency；
- 检查每个 rank 的样本切分、loss 聚合和参数一致性。

### 04 · NCCL Benchmark

使用 `nccl-tests` 测试 AllReduce、AllGather、ReduceScatter 和 Broadcast，扫描小消息到大消息，区分 latency-bound 与 bandwidth-bound 区域，并记录拓扑、NCCL/CUDA/driver 版本。

### 05 · TinyCollective

先用 `torch.distributed.send/recv` 实现教学版算法：

- Gather + Reduce + Broadcast；
- Ring ReduceScatter + Ring AllGather；
- 与 `torch.distributed.all_reduce` 比较正确性和性能；
- 推导通信数据量、轮数和理论复杂度，再用曲线验证。

### 06 · Mini Training Engine

把前五阶段组合为一个小型但可读的训练引擎，逐项加入：

- 单卡与 DDP；
- mixed precision 与 gradient accumulation；
- gradient bucketing；
- asynchronous collective；
- communication/computation overlap；
- 训练、显存、计算和通信 profiling；
- checkpoint 与故障恢复的最小实现。

每项能力都通过 feature flag 开关，并用消融实验说明它带来的收益与代价。

### 07 · FSDP2 / Tensor Parallel

当模型确实无法放进单卡时，再引入 FSDP2；当 FSDP2 遇到扩展瓶颈时，再学习 TP/PP 和 DeviceMesh。这里关注“为什么选择”，而不只是“配置能跑”。

## 硬件不足也可以开始

| 资源 | 可以完成的内容 |
|---|---|
| 只有 CPU | 训练循环、测试、Gloo 多进程、collective 算法逻辑 |
| 1 张 NVIDIA GPU | AMP、Profiler、CUDA/Triton、单卡 benchmark |
| 2–4 张 GPU | DDP、NCCL、Ring AllReduce、FSDP2 入门 |
| 8 张或多节点 | scaling、拓扑影响、通信计算重叠、组合并行 |

昂贵实验应先通过小规模 correctness test；云端只运行已经冻结配置的 benchmark，并在报告中公开实例型号与费用。

## 参考项目怎么读

| 项目 | 在本项目中的用途 | 建议关注 |
|---|---|---|
| [pytorch/examples](https://github.com/pytorch/examples) | 训练与分布式基线 | 代码组织、官方 API 用法 |
| [timm](https://github.com/huggingface/pytorch-image-models) | 成熟图像训练系统 | data pipeline、优化器与训练工程 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 小而完整的 GPT 训练器 | 训练循环、checkpoint、DDP |
| [Triton](https://github.com/triton-lang/triton) | GPU kernel 学习 | 官方 tutorials 与编程模型 |
| [TritonBench](https://github.com/triton-lang/tritonbench) | benchmark 设计参考 | operator corpus 与测量方法 |
| [CUDA Samples](https://github.com/NVIDIA/cuda-samples) | CUDA 基础与设备能力 | 内存、并行模型、工具链 |
| [nccl-tests](https://github.com/NVIDIA/nccl-tests) | 通信性能基线 | `all_reduce_perf` 与 `busbw` |
| [TorchTitan](https://github.com/pytorch/torchtitan) | 最终架构对照 | parallelize、checkpoint、profiling |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 大规模并行参考 | TP/PP/DP 的组合方式 |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 系统设计对照 | ZeRO、通信与显存优化 |
| [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) | 优化方法论范例 | 每次修改如何影响时间与收敛 |

原则是“带着问题读源码”：先在 TrainScale Lab 得到一个瓶颈，再去成熟项目寻找设计答案。不要直接复制最终实现。

## 可复现性约定

实验报告至少记录：

- Git commit、命令和完整配置；
- GPU/CPU/互联拓扑；
- OS、Python、PyTorch、CUDA、NCCL 和 driver 版本；
- 数据集版本、随机种子与预处理；
- warm-up、重复次数、均值和离散程度；
- 吞吐的统计口径、峰值显存和正确性阈值；
- 已知限制与排障说明。

未经实际测量的结果一律标注“待实测”。不同硬件的绝对性能不直接排名，优先比较同一环境下的相对变化。

## 参与贡献

项目尚在早期阶段。欢迎贡献：

- 更小、更清楚的原理实现；
- 可在不同硬件复现的 benchmark；
- 正确性测试、性能回归测试和故障排查记录；
- 对实验结论的反例或更严谨解释。

提交代码时请同时给出运行环境、复现命令和结果文件；单独粘贴一张无法追溯的性能截图不视为完整实验。

## 项目边界

TrainScale Lab 是教学与研究型实现，不承诺生产环境所需的稳定性、安全性和容错能力。早期阶段不会把 RDMA verbs、DPDK、Linux 内核网络栈或完整 NCCL 源码作为前置知识；这些主题将在实际瓶颈需要时再扩展。

## License

项目采用 Apache-2.0 License，见 [`LICENSE`](LICENSE)。引用项目用于学习和对照，不代表其代码会被直接复制到本仓库。
