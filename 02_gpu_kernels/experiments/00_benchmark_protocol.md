# 实验 00：先证明计时工具可信

> 状态：已完成。历史失败已复现并保留；升级驱动后，默认环境的 eager、`torch.compile` 和 Triton launch 均通过。

## 为什么先做这个实验

GPU 默认异步执行：Python 发出命令后可能立刻返回，GPU 还在工作。如果直接用普通时钟相减，测到的可能只是“下单时间”，不是“完成时间”。另外，Triton 第一次调用要即时编译（JIT），把编译时间混进稳态延迟也不公平。

所以在比较任何 kernel 前，先回答三个问题：

1. CUDA 基础计算能不能运行；
2. 计时有没有正确同步；
3. Triton 能不能完成“编译 → 加载 → launch”。

## 小白名词

- **kernel**：在 GPU 上并行执行的一小段程序。
- **launch**：CPU 向 GPU 提交一次 kernel 的动作。
- **异步**：CPU 提交后不等待 GPU 做完，继续往下执行。
- **CUDA Event**：记录 GPU 时间线位置的计时器，适合测 GPU 工作。
- **JIT**：运行到某个 shape 时才编译代码。
- **median/p10/p90**：中位数、较快的 10% 位置、较慢的 90% 位置，用来观察波动。

## 一般会得到什么

正常环境中，PyTorch eager 和最小 Triton kernel 都应通过。首次 Triton 调用明显更慢，后续 steady-state 稳定在微秒级。漏掉同步时，wall-clock 会低估真实时间。

## 本机环境

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB |
| Compute capability | SM 12.0 |
| OS | WSL2 Linux 6.18 |
| Python | 3.11.16 |
| PyTorch | 2.12.1+cu129 |
| Triton | 3.7.1 |
| Windows Driver | 610.88 |
| `nvcc` | CUDA Toolkit 13.0.88，支持 `sm_120` |

## 跟着做：先跑一次可信的短实验

代码入口是 [`check_environment.py`](../benchmarks/check_environment.py) 和
[`run_triton_comparison.py`](../benchmarks/run_triton_comparison.py)。在 WSL Ubuntu
的仓库根目录执行：

```bash
mkdir -p 02_gpu_kernels/results/raw/tutorial

.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py

TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py

.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite smoke --samples 3 --warmup 1 \
  --output 02_gpu_kernels/results/raw/tutorial/00_smoke.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/00_smoke.json
```

第一条探针末尾应为 `all_required_checks_passed=True`；pytest 应全部 passed；
benchmark 会逐行显示 `case/pytorch: success`、`case/triton: success`，最后显示
`all_cases_passed=True`。这一步只证明工具链和短流程可信，不产生正式性能结论。
若这里失败，先按[环境教程](../ENVIRONMENT.md)排错，不要继续后面实验。

## 实际做了什么、得到什么

| 日期/环境 | 实际结果 | 说明 |
|---|---|---|
| 2026-08-23，driver 577.05，cu129/Triton 3.7.1 | eager 通过；`torch.compile` 与 Triton 在加载/launch 附近退出或段错误 | 历史失败现场，保存在旧基线 JSON |
| 2026-08-24，driver 610.88，同一 cu129/Triton 3.7.1 | 五段隔离探针全部通过；GPU 测试 `13 passed` | 证明不必为了 SM 12.0 强制改 nightly |
| 2026-08-24，driver 610.88，隔离 cu130 nightly | 五段探针全部通过；GPU 测试 `13 passed` | 只作为兼容性诊断兜底 |

环境探针逐个在子进程执行环境采集、eager、`torch.compile`、Triton Softmax 和 Triton Vector Add。正式 benchmark 还为每个 case/implementation 使用独立进程、新 Triton/Inductor cache、10 次 warm-up、21 个 CUDA Event 样本。14 组 PyTorch/Triton 比较全部先通过 correctness gate，再产生性能数字。

历史现象与 PyTorch 的 [SM 12.0 Triton segfault issue](https://github.com/pytorch/pytorch/issues/176426)边界相似；本机升级驱动后已消失，因此教程把它视为“特定软件组合的兼容性失败”，不写成所有 SM 12.0 都必然失败。

## 理论解释

普通 PyTorch eager 使用已经构建好的 ATen/CUDA 库 kernel；Triton 则生成针对当前 GPU 的代码，再通过 CUDA Driver API 加载。旧环境中 eager 正常而 Triton 在 handle 初始化崩溃，说明故障边界位于生成代码/驱动加载链，而不是“GPU 完全不可用”。驱动升级后原 Python 栈直接恢复，又进一步支持这一定位。

Segmentation fault 是进程级错误，pytest 无法像普通 assertion 那样捕获。因此后续 Triton case 必须放在独立子进程，不能让一个崩溃吞掉全部结果。

## 结论与收尾

环境门与计时门现在都已通过。教程默认只维护根 `.venv`；遇到进程崩溃先升级驱动并重启 WSL，仍失败才建立仓库外 nightly 环境。旧失败见 [eager_baselines.json](../results/eager_baselines.json)，当前验证见 [sm120_environment_validation.json](../results/sm120_environment_validation.json)，正式结果见 [triton_comparison_sm120_cu129.json](../results/triton_comparison_sm120_cu129.json)。
