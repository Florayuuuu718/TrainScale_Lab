# 实验 03：Softmax——为什么要先减最大值

> 状态：大幅值、ragged、17–4097 行宽的 PyTorch、Triton baseline/policy、CUDA serial/block 扫描已完成。

## 为什么做

Softmax 把一行任意实数变成和为 1 的概率：

`softmax(x_i) = exp(x_i) / sum(exp(x_j))`

它同时包含求最大值、指数和求和，是学习 reduction、数值稳定和算子融合的典型例子。

## 小白名词

- **reduction**：把一组数缩成一个数，例如 max 或 sum。
- **overflow**：数太大，浮点格式无法表示，变成 Inf。
- **数值稳定**：数学等价，但选择更不容易溢出/丢精度的计算形式。
- **row width**：每一行参与 Softmax 的元素个数。

稳定版本先减去一行最大值：

`exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

减同一个常数不改变最终概率，却保证最大的指数是 `exp(0)=1`，显著降低 overflow 风险。

## 一般预期

小行宽主要受固定成本影响；行数和行宽增大后，reduction 与访存占比上升。融合实现可以避免 max、exp、sum、divide 之间的中间 tensor，但行太宽会增加寄存器/shared memory 压力。

## 跟着做：运行稳定 Softmax

阅读 [`triton_ops.py`](../trainscale_kernels/triton_ops.py) 中的
`_softmax_kernel`、`softmax_baseline` 和 `softmax`。重点找三步：求行最大值、
计算 `exp(x - max)`、除以行和。`softmax_baseline` 固定单 warp，`softmax` 根据
行宽选择 warp 数；两者都不是预先保证更快的“答案”。

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k softmax

.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator softmax --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/03_softmax.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/03_softmax.json
```

测试包含 127/509 等 ragged 行宽和大幅值稳定性；benchmark 的两个配置 shape
都应通过 correctness，最后出现 `all_cases_passed=True`。本命令比较 PyTorch 与
默认 Triton policy。若要复现单 warp、policy、CUDA serial/block 的五方表，继续
完成[实验 07](07_cuda_triton_comparison.md)。正式数字使用 `21/10`。

## 实际结果

输入乘以 20，主动覆盖较大正负值：

输入按确定性公式放大约 20 倍，五条路径都使用预分配输出：

| shape | PyTorch | Triton 1 warp | Triton warp policy | CUDA serial | CUDA block |
|---|---:|---:|---:|---:|---:|
| 1×17 | 10.221 | 17.527 | 16.517 | **8.678** | 8.721 |
| 8×127 | 11.821 | 13.887 | 16.800 | 11.908 | **9.028** |
| 32×509 | **8.766** | 14.271 | 14.683 | 25.024 | 10.951 |
| 256×1024 | **9.444** | 16.508 | 15.122 | 61.538 | 12.077 |
| 32×4097 | **9.759** | 16.896 | 18.256 | 180.996 | 10.834 |

单位均为 µs，10 次 warm-up、21 个样本。25 条路径全部通过；GPU tests 还验证 out buffer 和超过 65,536 列时显式拒绝。

## 理论分析

CUDA serial 让一个线程完成整行 max/sum/write，随行宽从 17 增到 4097，延迟从 8.7 增到 181 µs；CUDA block 让 256 个线程分段处理并通过 shared memory reduction，4097 列仍约 10.8 µs。这直接说明 reduction 的并行组织为何重要。

Triton 的简单 warp policy 只在 17 与 1024 列略好于单 warp，在其余行宽反而更慢。它是一次失败但有效的优化假设：`num_warps` 增多会提高并行参与，也会增加调度、同步和资源压力，不能用“更多 warp”替代实测选型。

非 2 次幂的 127 列很重要：自写 kernel 通常把 block 补到下一个 2 次幂，并用 mask 把填充位置设为负无穷，否则 max/sum 会被假数据污染。

## 结论与收尾

五种行宽和两套语言实现都证明了稳定减最大值与 ragged mask 的正确性。CUDA block 相比 serial 在 4097 列快约 16.7×；Triton warp policy 没有普遍加速。结论是 reduction 必须并行，但具体 block/warp 策略仍需按 shape 测量。
