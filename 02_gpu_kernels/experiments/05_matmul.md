# 实验 05：MatMul——为什么 512 比 509 友好

> 状态：ragged forward/backward、FP32 reference、四候选有限 autotune 与选型记录已完成。

## 为什么做

矩阵乘法是深度学习最重要的计算之一：

`C[M,N] = A[M,K] @ B[K,N]`

每个输出元素要做 K 次乘加。它与 Vector Add 不同，数据可以被反复复用，因此常常是 compute-bound，也是学习 tile、Tensor Core 和 autotune 的核心实验。

## 小白名词

- **tile**：把大矩阵切成小块，放到更快的片上存储重复使用。
- **FLOP**：一次浮点加法或乘法；MatMul 常按 `2MNK` 计数。
- **TFLOPS**：每秒万亿次浮点运算。
- **Tensor Core**：专门加速小矩阵乘加的 GPU 单元。
- **ragged shape**：M/N/K 不能整除 tile 的不规则尺寸。
- **autotune**：在若干 tile/warp 配置中实测选择较好的一个。

## 一般预期

大且整齐的 FP16 矩阵更容易利用 Tensor Core；tiny 矩阵受 launch 影响；ragged shape 需要 mask，尾块线程有一部分做无效工作，也可能触发不同库算法。

## 跟着做：比较 shape，并亲手选择 tile

阅读 [`triton_ops.py`](../trainscale_kernels/triton_ops.py) 的 `_matmul_kernel`、
`matmul_configured`、`matmul` 和 `matmul_backward`。`matmul_configured` 暴露
block M/N/K、group M 和 warp 数；候选不是藏在代码里，而是在
[`matmul_candidates.toml`](../configs/matmul_candidates.toml) 中列出。

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k matmul

# 默认配置：tiny、512 和 ragged 509 的 PyTorch/Triton forward
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator matmul --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/05_matmul_forward.json

# 穷举配置文件中的 4 个候选，并为每个 shape 选 median 最低者
.venv/bin/python 02_gpu_kernels/benchmarks/run_matmul_autotune.py \
  --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/05_matmul_autotune.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/05_matmul_autotune.json
```

autotune 会运行 2 个 shape ×（PyTorch + 4 个候选），终端每条都应为
`success`，末尾是 `all_candidates_passed=True`。结果表中的 `selected` 是该
shape 实测选出的配置，不是跨 shape 永远最优的配置。正式复现改用 `21/10`；
不同 GPU 可以选出不同候选，这正是本实验要观察的现象。

## 实际结果

| M×N×K | PyTorch median / TFLOPS | Triton median / TFLOPS | Triton 相对 PyTorch |
|---|---:|---:|---:|
| 17×31×23 | 20.036 µs / 0.0012 | 25.119 µs / 0.0010 | 0.798× |
| 512³ | 14.636 µs / 18.34 | 27.072 µs / 9.92 | 0.541× |
| 509³ | 27.905 µs / 9.45 | 30.049 µs / 8.78 | 0.929× |

输入为 FP16。correctness 不直接让一个 FP16 实现充当另一个的真值，而是两边都与 FP32 MatMul reference 比较。长 K 会累积舍入误差，K≥256 统一使用 `atol=0.025, rtol=0.02`；这条规则同时约束 PyTorch 与 Triton，不是只为某个实现放宽。独立测试还覆盖 17×31×23、64³ 的 backward。

Profiler 中，PyTorch 512³ 主要是 CUTLASS tensorop kernel，20 次约 228.394 µs；Triton kernel 20 次约 459.432 µs，和逻辑延迟约 2 倍的差距一致。

### 有限 autotune 实测

| shape | 最佳 Triton 配置 | PyTorch | 最佳 Triton | Triton 相对 PyTorch |
|---|---|---:|---:|---:|
| 512³ | M64/N64/K32，group 8，8 warps | 14.650 µs | 22.497 µs | 0.651× |
| 509³ | M32/N64/K32，group 8，4 warps | 31.898 µs | 25.268 µs | **1.262×** |

四个候选在两个 shape 上全部与 FP32 reference 对齐。512³ 候选从 22.497–26.881 µs；509³ 从 25.268–30.047 µs。选型规则只看 correctness 通过后的 21-sample median，不偷偷删除慢候选。

## 理论分析

512 能被常见的 16/32/64/128 tile 整除；509 会留下尾块。若按 32×32 tile 覆盖 509，需要调度到 512 边界，但少量尾块浪费不足以解释所有差距；库还可能因对齐条件选择不同 kernel。实测显示 PyTorch 从 14.64 增到 27.90 µs，Triton 从 27.07 增到 30.05 µs，说明两条实现对 ragged shape 的敏感度不同。

不同 shape 的最佳 tile 不同：64×64 对整齐 512³ 最好，32×64 对 ragged 509³ 最好。较大的 tile 提高数据复用，但尾块可能浪费线程并增加寄存器；这就是 autotune 需要把 shape 放进 key 的理论原因。509³ 上有限搜索超过 PyTorch，只能说明本机这一 shape 的相对结果，不能推广成 Triton MatMul 普遍更快。

## 结论与收尾

MatMul 性能强烈依赖 shape、对齐、累积精度与 tile。有限 autotune 在 512³ 仍落后 PyTorch，在 509³ 则达到 1.262×；这组相反结论正好说明不能用单一方阵或单一配置评价实现。候选、排名、cold JIT 和完整精度证据均已归档。
