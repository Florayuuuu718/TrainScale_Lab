# TrainScale Lab 文档导航

如果你第一次接触 PyTorch，请不要从源码逐文件阅读。这个项目的学习顺序是：先知道问题，再运行最小实验，最后回到源码理解每一步。

## 我们现在位于哪里

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| M0 | 建立可复现仓库、Python 环境、依赖锁和 CPU CI | 本地完成，远端 CI 待首次 push |
| M1 | 建立可复现的单卡训练、恢复与性能实验 | 已封存：本地完整验收通过 |
| M2 | CUDA/Triton 自定义算子 | 尚未开始 |

M1 不追求真实数据集的最高准确率。我们用 synthetic 隔离验证数学链路，再用 CIFAR-10 子集验证真实图像管道和 CNN，并对 checkpoint、AMP、累积、workers、compile 与 Profiler 给出实测证据。

## 初学者推荐阅读顺序

1. [M0/M1 从这里开始](getting-started/README.md)：先建立全局认识，再按命令运行。
2. 选择环境教程：
   - **Windows + NVIDIA GPU 完整路线**：[从零搭建 WSL2 + Ubuntu + PyTorch GPU](getting-started/wsl2-gpu.md)，从安装发行版、选择项目位置一路验收到训练、compile 与 Profiler。
   - **原生 Windows CPU/基础路线或原生 Linux**：[通用环境搭建](getting-started/m0-m1-environment.md)，理解虚拟环境、wheel、driver、CUDA runtime 和 `nvcc`。
3. [M0 仓库基建](getting-started/m0-repository-foundation.md)：理解 Git、License、`pyproject.toml`、`uv.lock` 和 CI 在解决什么问题。
4. [PyTorch 训练基础概念](concepts/pytorch-training-basics.md)：理解 dataset、batch、epoch、logits、loss、反向传播和验证。
5. [01 · PyTorch Training](../01_pytorch_training/README.md)：完整复现当前训练、查看结果并定位源码。
6. [测试说明](../01_pytorch_training/tests/README.md)：理解 10 个测试分别证明了什么。
7. [实验说明](../01_pytorch_training/experiments/README.md)：阅读成功实验、理论推理与排障提示。
8. [checkpoint 状态契约](checkpoint-contract.md)：理解为什么断点恢复不能只保存模型权重。

## 文档类型

| 目录 | 内容 | 阅读目的 |
|---|---|---|
| `docs/getting-started/` | 环境和仓库搭建 | 让新机器可以从零复现 |
| `docs/concepts/` | 不依赖具体命令的概念解释 | 建立知识框架 |
| `docs/experiments/` | 已冻结实验的过程与实测结果 | 学会提出假设、控制变量和解释结果 |
| `01_pytorch_training/` | M1 阶段入口、配置、测试和实验导航 | 把概念映射到代码与命令 |

## 阅读约定

- 命令默认从仓库根目录执行，Windows 使用 PowerShell。
- “预期输出”用于判断流程是否正确，数值可能因硬件和随机种子不同略有变化。
- `01_pytorch_training/results/` 的小型摘要和图表进入 Git；其 `raw/`、`data/` 和 `.venv/` 不提交。
- 当前阶段只解释已经实现和验证的内容；尚未完成的 M2 不会伪装成可运行功能。
