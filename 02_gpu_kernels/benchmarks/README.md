# 02 Benchmark Harness

本目录保存模块专属的环境探针、正式计时与 Profiler 入口；根目录 `benchmarks/` 只保存跨模块 schema 或汇总能力。

## 可运行入口

| 脚本 | 解决的问题 |
|---|---|
| `check_environment.py` | 当前 Python/GPU/Triton/Toolkit 能否真正运行，而不只是能 import |
| `run_triton_comparison.py` | PyTorch 与 Triton 的 correctness、冷启动和 steady-state 对照 |
| `profile_triton_comparison.py` | 代表性算子的 CUDA kernel 数量、device time 与内存活动 |
| `run_cuda_triton_comparison.py` | Vector Add/Softmax 的 PyTorch、两种 Triton、两种 CUDA C++ kernel-only 对照 |
| `run_layer_norm_training.py` | LayerNorm hidden/dtype/eps 的 forward/backward 扫描 |
| `run_matmul_autotune.py` | 穷举有限 tile/warp 候选并按 correctness 后的 median 选型 |
| `show_results.py` | 把较大的 JSON 转成初学者可读的终端表格 |
| `summarize_module02.py` | 校验正式 JSON 并生成带 SHA-256 的模块汇总 |

在 WSL 仓库根目录先执行：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
TRAINSCALE_RUN_SM120_TRITON=1 \
  .venv/bin/python -m pytest -q 02_gpu_kernels/tests/test_triton_ops.py
```

正式性能结果必须在 WSL 的 Linux 文件系统（例如 `/home/...`）运行，不能把 `/mnt/c` 或 `/mnt/d` 的跨文件系统开销混入冷启动：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --output 02_gpu_kernels/results/triton_comparison.json
.venv/bin/python 02_gpu_kernels/benchmarks/profile_triton_comparison.py \
  --output 02_gpu_kernels/results/triton_profiler.json
```

学习时不必每次运行 14 个 case。按算子或按单个 case 选择，并把练习结果写入
Git 忽略的 `results/raw/tutorial/`：

```bash
mkdir -p 02_gpu_kernels/results/raw/tutorial

# 运行某个算子的全部已配置 shape
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator softmax --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/softmax.json

# 只运行一个 case；--case 可以重复，但不能和 --operator 混用
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --case matmul_509x509x509 --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/matmul_509.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/softmax.json
```

终端逐行打印 `case/implementation: success`。最后必须看到
`all_cases_passed=True`，否则先读 JSON 中的 correctness/error 字段，不要解释
性能数字。

CUDA C++ 构建与新增正式实验：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
.venv/bin/python 02_gpu_kernels/cuda/build_cuda_bench.py \
  --output /tmp/trainscale-kernel-bench

.venv/bin/python 02_gpu_kernels/benchmarks/run_cuda_triton_comparison.py \
  --cuda-binary /tmp/trainscale-kernel-bench
.venv/bin/python 02_gpu_kernels/benchmarks/run_layer_norm_training.py
.venv/bin/python 02_gpu_kernels/benchmarks/run_matmul_autotune.py
.venv/bin/python 02_gpu_kernels/benchmarks/summarize_module02.py
```

Ubuntu 26.04 需给构建脚本补上环境指南记录的两个 `--nvcc-flag`；Ubuntu 24.04 官方组合不需要这条 workaround。

正式 runner 会给每个 case/implementation 建独立子进程和新缓存。这样某个 JIT 崩溃可被记录为失败，也不会污染后续 case。

## Harness 已实现的契约

- 展开 tiny、large、ragged、causal 等完整 case 矩阵；
- 统一生成输入并把同一输入交给各 implementation；
- 在计时前执行 correctness gate；
- 分离 compile/cold-start 与 steady-state；
- CUDA 正确同步，避免只测到异步 launch；
- 多次采样并输出 median、p10、p90 和样本数；
- 记录 latency，以及可定义时的 GB/s、TFLOPS、峰值显存；
- 捕获 unsupported、OOM、compile/runtime error；
- 保存环境、commit 和完整配置到 JSON。

## 计时解释

冷启动用 wall clock 加显式同步，steady-state 用 CUDA Event；10 次 warm-up 后保留 21 个样本并报告 median/p10/p90。Triton 在隔离 case 中的首次 JIT 约为 0.75–0.94 秒，而稳态是微秒级，因此两个数字必须分开。不要从逻辑延迟里随意减去一个固定常数制造“净 kernel 时间”；纯 device kernel 时间由 Profiler 单独报告。

## 指标定义

- latency：一次逻辑算子调用的 steady-state 时间；
- effective bandwidth：按数学上必须读取/写出的字节数定义，并在报告中写出公式；
- TFLOPS：按明确的 FLOP 约定计算，例如 matmul 常用 `2MNK`；
- speedup：同环境、同 case 下 `baseline_latency / candidate_latency`；
- compile latency：第一次可用调用包含的 JIT 成本，独立于 steady-state。

指标、TOML 和结果 schema 已进入普通 CPU 单元测试。性能数值本身不设跨 GPU 阈值。
