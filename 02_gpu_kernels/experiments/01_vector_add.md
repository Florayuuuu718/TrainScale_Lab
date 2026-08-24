# 实验 01：Vector Add——GPU 为什么会被“搬数据”限制

> 状态：PyTorch/Triton/CUDA C++ 的 FP32/FP16/BF16 correctness、kernel-only 性能与 Profiler 已完成。

## 为什么做

Vector Add 的数学最简单：`out[i] = x[i] + y[i]`。每个元素只做一次加法，却要从显存读两个数、写一个数，因此它适合学习索引、尾块 mask、launch 固定成本和内存带宽。

## 小白名词

- **memory-bound**：速度主要取决于搬数据，而不是做算术。
- **带宽**：单位时间能搬多少字节。
- **block/tile**：一次分给一组 GPU 线程处理的数据块。
- **mask**：最后一个 block 不满时，禁止越界线程读写。
- **effective bandwidth**：按“理论必须搬的字节 ÷ 时间”算出的指标，不一定等于真实 DRAM 带宽。

FP32 每个数 4 字节。一次加法理论至少读 `x/y` 并写 `out`，所以最小数据量是 `3 × N × 4` 字节。

## 一般预期

小 N 时，launch/dispatch 的固定成本占主导；N 增大后，延迟才随数据量增长并逐渐受内存系统限制。非整齐 N 必须通过 mask 保证最后一块正确。

## 跟着做：运行 Vector Add

先打开 [`triton_ops.py`](../trainscale_kernels/triton_ops.py)，找到
`_vector_add_kernel` 和 `vector_add`。kernel 用 `tl.program_id` 确定当前 program
负责的区间，用 `offsets < n_elements` 处理 257 这种不能整除 block 的尾部。
case 来自 [`benchmark_full.toml`](../configs/benchmark_full.toml)。

然后在 WSL Ubuntu 仓库根目录执行：

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k vector_add

.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator vector_add --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/01_vector_add.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/01_vector_add.json
```

应看到两个 shape 各有 PyTorch/Triton 两行 `success`，最后是
`all_cases_passed=True`。表格中的 `speedup` 定义为
`PyTorch latency / Triton latency`：大于 1 才表示 Triton 更快。快速流程通过后，
把 `--samples 5 --warmup 2` 改成 `--samples 21 --warmup 10` 才做正式复现。

CUDA packed/scalar 版本位于 [`kernel_bench.cu`](../cuda/kernel_bench.cu)，其构建和
三方复现放在[实验 07](07_cuda_triton_comparison.md)，避免这里先要求安装 Toolkit。

## 实际方法

使用 CUDA Event，10 次 warm-up，21 组采样，每组内部重复多次。报告中位数和 p90。完整配置和未四舍五入数据见结果 JSON。

## 实际结果

下表使用预分配输出，四条路径都排除 device allocation 和 host-device copy，只计 launch + kernel：

| case | PyTorch | Triton | CUDA scalar | CUDA packed |
|---|---:|---:|---:|---:|
| N=257, FP32 | 10.616 µs | 16.505 µs | **10.039 µs** | 13.256 µs |
| N=4097, FP16 | **8.552 µs** | 16.043 µs | 9.509 µs | 8.713 µs |
| N=4097, BF16 | 15.007 µs | 15.436 µs | **8.660 µs** | 9.168 µs |
| N=1,048,576, FP32 | 16.945 µs | 16.741 µs | 14.502 µs | **11.841 µs** |

全部 16 条路径先过 correctness；CUDA packed 分别是 `float4`、`half2`、`bfloat162`，ragged 尾部由独立边界逻辑处理。早期逻辑算子结果包含输出分配，因此与本表口径不同，不能直接拼速度比。Profiler 的 20 次 large PyTorch/Triton kernel 聚合证据仍保留。

## 理论分析

N 增大约 4080 倍，PyTorch median 只从 8.93 增到 11.11 µs，说明 tiny case 主要支付 launch、dispatch 和输出分配等固定成本。large case 的有效带宽超过外部显存标称能力，不能解释为“显存真的达到 1.1 TB/s”：重复使用同一输入可能命中 cache，而且公式只计算数学最小字节，不是 DRAM transaction 实测值。

N=257 时 `float4` 反而比 scalar 慢，4097 的 FP16/BF16 packed 也没有稳定胜出：向量化会减少指令数量，但 tiny case 的额外类型/边界处理和 launch 噪声可能抵消收益。到 1M FP32 时数据量足够大，`float4` 才以 11.841 µs 成为最快路径。这是“优化必须按规模验证”的直接反例。

## 结论与收尾

三种 dtype、prime/ragged/large shape 都已正确。tiny case 没有统一赢家，1M FP32 的 CUDA `float4` 最快；朴素 Triton 没有明显优势。实验结论是：Vector Add 的核心约束仍是固定 launch 与数据移动，packed load 只有在工作量足以摊薄成本时才更可能获益。
