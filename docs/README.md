# TrainScale Lab 文档导航

如果你第一次接触 PyTorch，请不要从源码逐文件阅读。这个项目的学习顺序是：先知道问题，再运行最小实验，最后回到源码理解每一步。

## 我们现在位于哪里

| 模块 | 目标 | 当前状态 |
|---|---|---|
| 01 | 仓库基建、可复现的单卡训练、恢复与性能实验 | 已封存：本地完整验收通过 |
| [02](../02_gpu_kernels/README.md) | CUDA/Triton 自定义算子 | 已封存：CUDA/Triton、前反向、Profiler 与发布验收通过 |
| [03](../03_distributed_training/README.md) | DDP、数据分片、checkpoint 与 scaling | 已封存：本地正确性、云端 1/2/4 GPU 与发布验收通过 |
| [04](../04_nccl_benchmark/README.md) | NCCL 曲线与 DDP 通信解释 | 本地工具与测试已完成，待真实多 GPU 实验 |
| [05](../05_tiny_collective/README.md) | centralized/ring collective 教学实现 | 规划已冻结，尚未实现 |
| [06](../06_training_engine/README.md) | Mini Engine、gradient reducer 与 overlap | 规划已冻结，尚未实现 |
| [07](../07_parallelism/README.md) | FSDP2、TP 与可选二维并行 | 规划已冻结，尚未实现 |

01 不追求真实数据集的最高准确率。我们用 synthetic 隔离验证数学链路，再用 CIFAR-10 子集验证真实图像管道和 CNN，并对 checkpoint、AMP、累积、workers、compile 与 Profiler 给出实测证据。

## 初学者推荐阅读顺序

1. [01 从这里开始](getting-started/README.md)：先建立全局认识，再按命令运行。
2. 选择环境教程：
   - **Windows + NVIDIA GPU 完整路线**：[从零搭建 WSL2 + Ubuntu + PyTorch GPU](getting-started/wsl2-gpu.md)，从安装发行版、选择项目位置一路验收到训练、compile 与 Profiler。
   - **原生 Windows CPU/基础路线或原生 Linux**：[通用环境搭建](getting-started/environment.md)，理解虚拟环境、wheel、driver、CUDA runtime 和 `nvcc`。
3. [01 仓库基建](getting-started/repository-foundation.md)：理解 Git、License、`pyproject.toml`、`uv.lock` 和 CI 在解决什么问题。
4. [PyTorch 训练基础概念](concepts/pytorch-training-basics.md)：理解 dataset、batch、epoch、logits、loss、反向传播和验证。
5. [01 · PyTorch Training](../01_pytorch_training/README.md)：完整复现当前训练、查看结果并定位源码。
6. [测试说明](../01_pytorch_training/tests/README.md)：理解 10 个测试分别证明了什么。
7. [实验说明](../01_pytorch_training/experiments/README.md)：阅读成功实验、理论推理与排障提示。
8. [checkpoint 状态契约](checkpoint-contract.md)：理解为什么断点恢复不能只保存模型权重。
9. [02 环境指南](../02_gpu_kernels/ENVIRONMENT.md)：先跑环境探针；默认只用 stable 根环境，失败才进入隔离兜底。
10. [02 · GPU Kernels](../02_gpu_kernels/README.md)：读范围和 benchmark 契约，再按实验顺序复现。
11. [02 验收清单](02-issues.md)：查看进入 03 前必须关闭的 12 项工作。
12. [03 · Distributed Training](../03_distributed_training/README.md)：从两 rank Gloo 开始，再进入 NCCL/scaling。
13. [03 云端四卡实验](../03_distributed_training/experiments/07_cloud_4gpu.md)：从租用、部署到结果校验和关机。
14. [03 验收清单](03-issues.md)：查看本地与云端证据怎样共同关闭阶段验收。
15. [04–07 开发总纲](04-07-development-plan.md)：查看后续依赖、共同完成定义和范围闸门。
16. [04 · NCCL Performance Lab](../04_nccl_benchmark/README.md)：从 03 的 scaling 反例进入通信测量。
17. [04 验收清单](04-issues.md)：按环境、曲线、DDP bridge 和发布验收推进。
18. [05–07 验收清单](05-issues.md)：完成 04 后按总纲进入后续阶段，并分别查看
    [06](06-issues.md) 和 [07](07-issues.md) 的必需项与可选边界。

## 文档类型

| 目录 | 内容 | 阅读目的 |
|---|---|---|
| `docs/getting-started/` | 环境和仓库搭建 | 让新机器可以从零复现 |
| `docs/concepts/` | 不依赖具体命令的概念解释 | 建立知识框架 |
| `docs/experiments/` | 已冻结实验的过程与实测结果 | 学会提出假设、控制变量和解释结果 |
| `01_pytorch_training/` | 01 模块入口、配置、测试和实验导航 | 把概念映射到代码与命令 |
| `02_gpu_kernels/` | 02 环境、实现、测试、benchmark 与实测报告 | 把算子机制映射到正确性与性能证据 |
| `03_distributed_training/` | DDP 源码、配置、测试、实验与结果 | 把多进程语义映射到正确性和 scaling 证据 |
| `04_nccl_benchmark/`–`07_parallelism/` | 后续模块问题、开发顺序和验收边界 | 按最小实现到可复现优化继续学习 |

## 阅读约定

- 命令默认从仓库根目录执行，Windows 使用 PowerShell。
- “预期输出”用于判断流程是否正确，数值可能因硬件和随机种子不同略有变化。
- `01_pytorch_training/results/` 的小型摘要和图表进入 Git；其 `raw/`、`data/` 和 `.venv/` 不提交。
- 02/03 的本机实测、历史失败和硬件限制分开标注；04–07 目前只有冻结规划，不会伪装成可运行功能。
