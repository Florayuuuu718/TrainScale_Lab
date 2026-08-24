# 实验 04：LayerNorm——归一化到底做了什么

> 状态：hidden、FP32/FP16、eps 的 forward/backward correctness 与正式性能扫描已完成。

## 为什么做

LayerNorm 广泛用于 Transformer。它对每个样本最后一维计算均值和方差，再把数据缩放到较稳定的分布，最后应用可学习的 weight 和 bias。

## 小白名词与公式

对一行 hidden values：

1. `mean = sum(x) / H`
2. `variance = sum((x-mean)^2) / H`
3. `x_hat = (x-mean) / sqrt(variance+eps)`
4. `y = x_hat * weight + bias`

- **eps**：防止方差接近 0 时除零的小常数。
- **affine**：最后的 weight/bias 可学习变换。
- **rstd**：`1/sqrt(variance+eps)`，反向传播经常保存它。
- **occupancy**：一个 SM 同时驻留多少活跃线程块；它只是影响性能的因素，不是越高越好。

## 一般预期

hidden 很小时固定成本占主导；hidden 增大后 reduction 和访存增加，但并行度也更充分。保存 mean/rstd 会增加少量写入，却能让 backward 少重算。

## 跟着做：先 forward，再做训练所需 backward

实现位于 [`triton_ops.py`](../trainscale_kernels/triton_ops.py) 的
`_layer_norm_forward_kernel`、`_layer_norm_backward_kernel`、`layer_norm` 和
`layer_norm_backward`。测试不仅比较输出 `y`，还比较 `mean`、`rstd`、`dx`、
`dweight` 和 `dbias`。

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k layer_norm

# 第一步：与其他算子相同的 forward 对照
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator layer_norm --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/04_layernorm_forward.json

# 第二步：4 个 dtype/hidden/eps case 的 forward 与 backward
.venv/bin/python 02_gpu_kernels/benchmarks/run_layer_norm_training.py \
  --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/04_layernorm_training.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/04_layernorm_training.json
```

第二个 runner 会打印 `case/forward/pytorch`、`case/forward/triton`、
`case/backward/pytorch`、`case/backward/triton`，总计 16 条路径；末尾必须是
`all_cases_passed=True`。表格按 phase 分行，因此不能把 forward 与 backward
延迟混成一个 speedup。正式复现将两个 runner 都改为 `21/10`。

## 实际结果

| rows×hidden / dtype / eps | phase | PyTorch | Triton | Triton/PyTorch speedup |
|---|---|---:|---:|---:|
| 8×127 / FP32 / 1e-5 | forward | 17.339 | 17.964 | 0.965× |
| 同上 | backward | 18.605 | 31.978 | 0.582× |
| 32×509 / FP16 / 1e-5 | forward | 18.481 | 20.811 | 0.888× |
| 同上 | backward | 17.419 | 31.332 | 0.556× |
| 32×509 / FP16 / 1e-3 | forward | 27.306 | 19.739 | 1.383× |
| 同上 | backward | 19.662 | 31.191 | 0.630× |
| 256×1024 / FP32 / 1e-5 | forward | 9.838 | 21.021 | 0.468× |
| 同上 | backward | 30.429 | 35.649 | 0.854× |

单位均为 µs。每条路径使用自己 forward 保存的 mean/rstd；PyTorch 通过 `native_layer_norm_backward`，Triton 返回 dx/dweight/dbias。8 个 comparison 全部通过 correctness，FP16 backward 最大绝对误差约 0.0065–0.0071，低于统一 0.04 容差。

## 理论分析

PyTorch 的 256×1024 forward 反而最快，说明常见 hidden=1024 的库路径和更高并行度可以超过问题规模增长。Triton backward 使用每行 kernel 并对 dweight/dbias 做 atomic add；rows 增多时原子竞争和清零开销存在，因而四组 backward 都没有胜过 PyTorch。

两组 32×509 只改变 eps，数学工作量不变；独立进程结果中 PyTorch median 却相差较大。不能据此声称 eps 会改变性能，合理读法是当前微秒级实验仍受时钟、调度和进程隔离波动影响；eps 扫描的主要价值是数值正确性。

Backward 不只是求 dx，还要求 dweight/dbias 沿 rows 做 reduction。简单实现若让每一行原子累加参数梯度，会产生竞争；两阶段 reduction 可能更快但需要额外临时空间。这就是为什么 forward 快不代表完整训练算子快。

## 结论与收尾

LayerNorm 的 y、mean、rstd、dx、dweight、dbias 已跨 hidden/dtype/eps 对齐，forward/backward 成本也分开记录。除单个 FP16/eps case 的 forward 外，成熟 PyTorch 通常更快；教学实现的主要后续优化方向是减少参数梯度 atomic contention，而不是继续宣称 forward 已代表完整训练成本。
