# TrainScale Lab 文档导航

TrainScale Lab 的目标不是让你照抄一串命令，而是让你逐步建立分析训练系统的能力。每一章都
遵循同一个循环：先理解问题和做出预测，再实现最小正确版本，最后用 benchmark、Profiler 和
结构化结果验证或修正预测。

如果你是第一次接触 PyTorch，不要从源码目录开始逐文件阅读。先选择下面的一条路线。

## 我应该从哪里开始

### 路线 A：从零完整学习

适合刚接触 PyTorch、GPU 或分布式训练的学习者：

1. [01 新手入口](getting-started/README.md)：认识训练循环、环境和第一次运行；
2. [01 · PyTorch Training](../01_pytorch_training/README.md)：训练、验证、恢复和可信测量；
3. [02 · GPU Kernels](../02_gpu_kernels/README.md)：从算子正确性进入 GPU 性能；
4. [03 · Distributed Training](../03_distributed_training/README.md)：理解 rank、数据分片和 DDP；
5. [04–07 学习地图](04-07-development-plan.md)：进入通信、reducer 和模型切分；
6. 依次完成 [04](../04_nccl_benchmark/README.md) →
   [05](../05_tiny_collective/README.md) →
   [06](../06_training_engine/README.md) →
   [07](../07_parallelism/README.md)。

遇到不熟悉的缩写时，先查
[分布式训练与通信术语表](concepts/distributed-systems-glossary.md)，不需要停下来通读框架源码。

### 路线 B：我只想先在本地体验

- 只有 CPU：使用 [通用环境教程](getting-started/environment.md)，可完成训练语义、Gloo、
  collective 调度、配置和大部分 correctness tests。
- Windows + NVIDIA GPU：使用 [WSL2 + Ubuntu 完整教程](getting-started/wsl2-gpu.md)，完成
  PyTorch GPU、CUDA/Triton、单卡 Profiler 和 NCCL 基础检查。
- Windows 上看到 05–07 的 Linux/Gloo 测试被跳过：使用
  [Linux/Gloo 正确性门教程](getting-started/linux-gloo-validation.md)在 WSL2 CPU 环境补齐，
  不需要租卡。
- 本地只有一张 GPU：先完成 01–03 和 04–07 的本地 gate；多 GPU 性能结果明确记为
  `unavailable`，不需要伪造数据。

### 路线 C：我要复现四卡实验

第一次租用云 GPU，直接使用
[JupyterLab 四卡一站式教程](getting-started/jupyterlab-4gpu.md)。它从实例选择、打开 Terminal、
环境预检开始，依次运行 04–07，最后打包、下载、校验并关机。熟悉 Linux shell 后，可以改用
[四卡 Runbook](04-07-rental-runbook.md) 作为简洁速查表。

## 七章怎样衔接

| 模块 | 先解决的问题 | 做出的最小系统 | 用什么证据判断 |
|---|---|---|---|
| [01](../01_pytorch_training/README.md) | 一次训练怎样才可靠 | 可恢复的 PyTorch trainer | loss、更新、checkpoint、吞吐 |
| [02](../02_gpu_kernels/README.md) | 单个算子为什么快或慢 | PyTorch/CUDA/Triton kernels | 数值误差、延迟、带宽、Profiler |
| [03](../03_distributed_training/README.md) | 多个进程怎样训练同一模型 | DDP + sampler + checkpoint | 数据覆盖、梯度一致、scaling |
| [04](../04_nccl_benchmark/README.md) | DDP 的通信时间从哪里来 | NCCL benchmark + DDP bridge | latency/bandwidth 曲线、timeline |
| [05](../05_tiny_collective/README.md) | AllReduce 内部如何移动数据 | centralized 与 ring | 通信轮次、通信量、NCCL 对照 |
| [06](../06_training_engine/README.md) | 怎样安排梯度同步 | mini engine + reducers | correctness、消融、真实 overlap |
| [07](../07_parallelism/README.md) | 模型或状态放不下怎么办 | FSDP2 与 TP | 更新一致、峰值显存、collective |

顺序不是随意的：04 用通信曲线解释 03；05 把 04 使用的 AllReduce 拆开；06 把 collective
放回 backward；07 再利用 06 的 Tiny Transformer 比较状态分片和层内切分。这样每一章只增加
一个主要变量，实验结果才容易解释。

## 当前完成状态

| 模块 | 状态 | 仍然保留的边界 |
|---|---|---|
| 01 | 已完成并封存 | 不追求数据集排行榜精度 |
| 02 | 已完成并封存 | 不覆盖所有 GPU 架构和算子 |
| 03 | 已完成并封存 | 8 GPU 与多节点为可选扩展 |
| 04 | 已完成 | DDP scaling 波动限制高精度效率结论 |
| 05 | 已完成 | 教学实现不替代 NCCL |
| 06 | 已完成 | overlap 与 AMP 不保证当前负载加速 |
| 07 | 已完成 | CPU/Gloo FSDP2 限制与 CUDA/NCCL 分开记录 |

“已完成”表示范围内的实现、correctness、正式实验、结果校验和报告齐全，不表示所有优化都
变快，也不表示扩展项被偷偷当作必修项。

## 文档怎样分工

| 文档 | 什么时候读 |
|---|---|
| 模块 `README.md` | 进入一章时，了解问题、源码入口、最短命令和边界 |
| `experiments/README.md` | 准备做实验时，按预测和控制变量推进 |
| `*_final_report.md` | 实验后，对照参考结果、一般规律和适用范围 |
| `results/*_summary.json` | 自动检查或精确引用代表性数字 |
| `docs/*-issues.md` | 维护者检查验收项，不是新手的必读主线 |
| [文档与实验发布规范](documentation-standard.md) | 新增或修改教程、实验和结果时 |

大型 stdout、rank JSON、checkpoint 和 Chrome trace 不进入 Git。它们保存在带 SHA-256 的
校验归档中；仓库只提交配置、紧凑摘要和解释报告。

## 阅读和复现约定

- 命令默认从仓库根目录执行；每份环境教程会注明使用 PowerShell 还是 Linux Terminal。
- 先确认 correctness，再看性能；`success`、`failed`、`unavailable` 的含义不能混用。
- 参考数值用于校验量级和趋势，不要求不同硬件得到完全相同的吞吐。
- 正式结论使用重复实验中位数和波动范围，不使用单次最快值。
- Profiler 用来解释机制，独立 benchmark 用来判断是否更快，两者不能互相替代。
- 遇到失败先保留日志并换新输出目录，不覆盖失败证据后反复运行到出现“漂亮数字”。
