# 02 · GPU Kernels

> 状态：已完成并通过本机发布验收。Windows driver 610.88 下，默认 `torch 2.12.1+cu129 / Triton 3.7.1` 环境通过 15 项 GPU 测试；CUDA 13.0 Vector Add/Softmax、LayerNorm backward、MatMul 有限 autotune、Profiler、汇总与验收 JSON 均已归档。

02 从“训练过程中看见 GPU kernel”推进到“亲手实现、验证、测量并解释一个 kernel”。01 已经通过 [AMP/compile 实验](../01_pytorch_training/experiments/04_fp32_amp_compile.md)和 [CUDA Profiler 实验](../01_pytorch_training/experiments/06_profiler.md)观察 PyTorch/Inductor 产生的 kernel；02 不重复训练系统实验，而是把单个算子作为研究对象。

## 1. 本模块回答什么

~~~text
数学定义
   ↓
PyTorch reference
   ↓
输入域与正确性矩阵
   ↓
Triton baseline ──→ CUDA C++ 深挖（选定算子）
   ↓
tile / fusion / memory access / launch 优化
   ↓
steady-state benchmark + profiler
   ↓
有限结论与失败边界
~~~

完成 02 后，应能回答：

- 一个算子是 memory-bound、compute-bound，还是被 launch/同步开销主导；
- shape、dtype、layout、tile 和 num_warps 为什么改变性能；
- 为什么融合能减少中间张量和显存流量；
- 为什么“单一 shape 比 PyTorch 快”不能证明实现更好；
- CUDA C++ 与 Triton 在开发成本、可控性和性能上的差异；
- 如何拒绝不支持的输入，而不是静默返回错误结果。

## 2. 范围与边界

主线算子顺序：

1. Vector Add：索引、program id、mask、有效带宽；
2. ReLU 与融合：逐元素算子、branch、launch 与中间张量；
3. Softmax：数值稳定、reduction、行宽与融合；
4. LayerNorm：统计量、reduction、仿射参数与 backward；
5. MatMul：blocked tiling、数据复用、autotune 与 Tensor Core；
6. Attention：从组合 reference 到 fused forward，理解为何避免物化 score matrix。

主线要求：

- 六个算子都有 PyTorch reference、Triton 实现、正确性测试和 shape/dtype benchmark；
- Vector Add 与 Softmax 额外实现 CUDA C++ 版本，用于语言和工具链对照；
- ReLU、LayerNorm、MatMul 必须覆盖训练所需 backward；Attention 主线验收 forward，backward 属于进阶项；
- 不支持的 layout、dtype、shape 或设备必须显式报错或被测试标记，不得静默算错。

暂不进入 02 主线：

- 手写高性能卷积、完整 FlashAttention backward、跨 GPU kernel、NCCL、分布式算子；
- 为追求榜单数字复制大型生产 kernel；
- 把所有 shape 都强行塞进一个万能 kernel；
- 跨机器比较绝对性能排名。

## 3. 目录规划

~~~text
02_gpu_kernels/
├── README.md                  # 本总指导
├── pytorch_ref/               # 数学定义和 PyTorch reference
├── triton/                    # 六个算子的 Triton baseline/optimized 实现
├── cuda/                      # Vector Add、Softmax 的 CUDA C++ 对照
├── configs/                   # shape/dtype/layout 与 benchmark 配方
├── tests/                     # correctness、gradient、边界与错误处理
├── benchmarks/                # 计时、环境采集、指标计算和绘图入口
├── experiments/               # 运行前假设、实测、解释和限制
└── results/
    ├── README.md              # 结果 schema 与提交边界
    └── raw/                   # 大型 trace、Nsight 报告和临时产物（应忽略）
~~~

规划阶段只建立文档入口；实现对应验收项时再创建源码和配置，避免空目录伪装成已完成能力。

## 4. 环境门槛

正式性能路线使用 Linux 或 WSL2 Ubuntu + NVIDIA GPU。开始前记录：

- GPU 型号、compute capability、显存和功耗状态；
- driver、Python、PyTorch、PyTorch CUDA runtime、Triton 版本；
- CUDA C++ 实验使用的 Toolkit/`nvcc` 版本；
- 是否可用 PyTorch Profiler、Nsight Systems、Nsight Compute；
- commit、后台负载和 GPU 时钟/温度限制。

Triton 主线不要求先安装系统 CUDA Toolkit；进入 CUDA C++ 对照前再验证 `nvcc` 与 PyTorch extension 工具链。环境验收失败时，不得用原生 Windows 的偶然行为代替正式结论。

本教程采用“一个默认环境，失败后才隔离兜底”的路线：

| 情况 | 应该怎么做 |
|---|---|
| 新读者、探针全部通过 | 只使用仓库根目录 `.venv`，不要额外安装 nightly |
| Triton 子进程崩溃 | 先升级 Windows NVIDIA 驱动，执行 `wsl --shutdown` 后重测 |
| 新驱动下仍失败 | 再建立仓库外的 cu130 nightly 诊断环境，不改根环境锁文件 |
| 只做 PyTorch/Triton | 不需要系统 CUDA Toolkit |
| 要编译 `.cu` | 单独安装 Toolkit；新装 WSL 推荐 Ubuntu 24.04 + CUDA 13.0 |

先阅读并执行 [02 环境指南](ENVIRONMENT.md)。其中的探针把 eager、`torch.compile`、Triton Softmax 和 Triton Vector Add 放入独立子进程；即使底层段错误，也不会让整套诊断信息一起消失。

## 5. 像 01 一样，从命令行逐项做实验

以下命令全部在 **WSL Ubuntu 终端**执行，不是在 Windows PowerShell 中执行。
先确认项目位于 Linux 文件系统，例如 `~/projects/TrainScale_Lab`，然后只做一次
公共准备：

```bash
cd ~/projects/TrainScale_Lab
uv sync --extra cu129 --extra dev --python 3.11
mkdir -p 02_gpu_kernels/results/raw/tutorial

.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py
```

探针末尾应出现 `all_required_checks_passed=True`，测试应全部 passed。然后按下面
顺序逐个实验；每个链接内都给出“检查源码 → 跑对应测试 → 快速实验 → 查看数字
→ 正式复现”的完整命令：

| 实验 | 你亲手运行什么 | 快速结果文件 |
|---|---|---|
| [00](experiments/00_benchmark_protocol.md) | 环境探针与 3-sample smoke | `raw/tutorial/00_smoke.json` |
| [01](experiments/01_vector_add.md) | `--operator vector_add` | `raw/tutorial/01_vector_add.json` |
| [02](experiments/02_relu_fusion.md) | `--operator relu_add` | `raw/tutorial/02_relu_add.json` |
| [03](experiments/03_softmax.md) | `--operator softmax` | `raw/tutorial/03_softmax.json` |
| [04](experiments/04_layernorm.md) | forward + training/backward runner | `raw/tutorial/04_*.json` |
| [05](experiments/05_matmul.md) | forward + 四候选有限调参 | `raw/tutorial/05_*.json` |
| [06](experiments/06_attention.md) | causal/non-causal fused forward | `raw/tutorial/06_attention.json` |
| [07](experiments/07_cuda_triton_comparison.md) | 编译 `.cu` 并做五方对照 | `raw/tutorial/07_cuda_compare.json` |
| [08](experiments/08_profiler_roofline.md) | PyTorch Profiler 与 trace | `raw/tutorial/08_profiler.json` |

快速实验默认用 2 次 warm-up、5 个样本，让学习者先确认流程；正式复现改为 10 次
warm-up、21 个样本。runner 会先做正确性检查，失败时不会生成可冒充性能结论的
成功记录。查看任意结果时使用仓库提供的结果阅读器：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/01_vector_add.json
```

命令里的反斜杠 `\` 表示下一行仍属于同一条 Bash 命令；终端提示符 `$` 不需要
输入。`results/raw/tutorial/` 被 Git 忽略，学习者的运行不会覆盖仓库保存的本机
正式结果。

## 6. 正确性契约

每个算子先定义输入域，再写实现。测试矩阵至少包含：

- shape：极小、常见、非 2 次幂、质数边界、大 shape；
- dtype：FP32，以及硬件支持时的 FP16/BF16；
- layout：contiguous；算子声称支持时再加入转置或 strided；
- 数值：零、正负混合、大幅值和会触发稳定性问题的输入；
- 边界：不能整除 tile、空维度或非法维度、错误 device/dtype；
- backward：需要训练支持的算子对照 PyTorch gradient。

默认以 FP32 reference 或 FP32 accumulation 结果作为低精度比较基准。容差必须按算子、dtype 和累积长度写入测试配置，不能为了让测试通过临时放宽。所有性能结果必须先通过对应 correctness gate。

## 7. Benchmark 契约

所有正式 benchmark 必须：

1. 区分首次编译/冷启动与 steady-state；
2. 在计时边界正确同步，优先使用 CUDA events 或可信 benchmark 工具；
3. 固定输入、shape、dtype、layout 和实现顺序，记录 warm-up 与重复次数；
4. 报告 median，并至少保留 p10/p90 或其他离散程度；
5. 同时保存 latency；适用时再计算 GB/s、TFLOPS、峰值显存；
6. PyTorch eager、PyTorch compile/SDPA（适用时）、Triton baseline、Triton optimized、CUDA C++ 使用相同输入域；
7. 保存失败、OOM、unsupported 和 compile error，不能只保留最好结果；
8. 只比较同一台机器、同一功耗/软件环境中的相对变化。

性能验收不是“每个点必须快于 PyTorch”。一个合格结论可以是没有加速，但必须用 launch、访存量、算术强度、融合、occupancy 或库调用边界解释，并指出证据范围。

## 8. 实验顺序

1. [Benchmark 方法校准](experiments/00_benchmark_protocol.md)
2. [Vector Add：mask 与内存带宽](experiments/01_vector_add.md)
3. [ReLU：逐元素融合与 launch 开销](experiments/02_relu_fusion.md)
4. [Softmax：稳定 reduction 与行融合](experiments/03_softmax.md)
5. [LayerNorm：统计量、仿射与 backward](experiments/04_layernorm.md)
6. [MatMul：tiling、autotune 与 Tensor Core](experiments/05_matmul.md)
7. [Attention：避免物化中间矩阵](experiments/06_attention.md)
8. [CUDA C++ / Triton / PyTorch 对照](experiments/07_cuda_triton_comparison.md)
9. [Profiler 与 roofline 总结](experiments/08_profiler_roofline.md)

九份文档已经回填正式实测。结果入口是 [module02_summary_sm120.json](results/module02_summary_sm120.json)，发布门见 [module02_acceptance_sm120.json](results/module02_acceptance_sm120.json)。前者保存每份正式源 JSON 的 SHA-256；后者分别记录 Windows 静态/CPU 检查与 WSL 真实 GPU/Toolkit 检查。环境、14-case forward、Profiler、41-path CUDA/Triton、LayerNorm training 和 MatMul autotune 都可继续下钻。旧 driver 577.05 失败被保留为历史样本，不代表当前状态。

## 9. 完成定义

- [x] [02 验收清单](../docs/02-issues.md)全部关闭；
- [x] 六个算子的输入域、reference、Triton 实现和错误边界明确；
- [x] Vector Add 与 Softmax 有 CUDA C++ baseline/optimized 对照；
- [x] 要求 backward 的算子通过 gradient 对照；
- [x] benchmark harness 分离冷启动与稳态，结果 schema 稳定；
- [x] PyTorch/Triton/CUDA 性能数字可追溯到环境、commit、配置、二进制 hash 和 JSON；
- [x] 至少一份 profiler 证据支持 memory-bound 解释，一份支持 compute-bound 解释；
- [x] 当前失败点、没有加速的 shape 和硬件限制完整保留；
- [x] CPU 快速测试进入 CI，GPU 测试有显式本地/自托管 opt-in 说明；
- [x] 新环境 smoke、Windows 全仓回归与 WSL GPU/Toolkit 回归全部通过。

继续阅读：[环境指南](ENVIRONMENT.md) → [验收清单](../docs/02-issues.md) → [配置矩阵](configs/README.md) → [测试说明](tests/README.md) → [实验索引](experiments/README.md) → [结果契约](results/README.md)。
