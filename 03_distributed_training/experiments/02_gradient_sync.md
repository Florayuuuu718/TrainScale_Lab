# 实验 02：DDP 梯度真的等价于同一个 global batch 吗

## 为什么做

DDP 快不快之前，必须证明它优化的是同一个数学问题。本实验把 global batch 64
切成 4 个 local batch 16，比较 4 rank DDP 与单进程一次读取全部 64 个样本的
梯度和 optimizer step。

## 小白名词

- **local batch**：单个 rank 一次处理的样本数。
- **global batch**：所有 rank 同一步处理的样本总数。
- **gradient AllReduce**：把每 rank 梯度求和并按 world size 平均。
- **replica**：每个 rank 上结构相同、参数应保持一致的模型副本。

## 一般预期

当每 rank local batch 一样大、loss 都使用 mean reduction 时，DDP 平均梯度应与
global batch 单进程 mean loss 梯度一致；浮点加法顺序不同可能产生约 `1e-8`
误差，但不应出现数量级差异。

## 跟着做

阅读 [`worker.py`](../trainscale_distributed/worker.py) 的 `run_gradient`：rank 按
连续区间切 global batch，DDP backward 后 rank 0 再计算完整 batch reference。

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment gradient --world-size 4 \
  --output 03_distributed_training/results/raw/tutorial/02_gradient.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/02_gradient.json
```

必须看到 `all_checks_passed=True`。阅读器会显示 rank 参数差、梯度 reference 差和
step 后参数差；这些数值越接近 0 越好，不是 speedup。

## 实际结果

4 rank 实测：

| 检查 | 最大绝对误差 |
|---|---:|
| rank 之间 step 后参数 | `0` |
| DDP 梯度 vs global-batch reference | `1.1176e-8` |
| DDP step 后参数 vs reference | `1.4901e-8` |

## 理论解释

每个 rank 对 16 个样本的 mean loss 求梯度。DDP reducer 把四份梯度相加再除以 4，
数学上等于 64 个样本 mean loss 的梯度。微小误差来自并行规约改变浮点加法顺序；
FP32 加法不满足严格结合律，因此不要求逐 bit 相等。

## 结论与收尾

在等 local batch、相同初始参数和 SGD 设置下，本机 DDP 更新与 global-batch
reference 对齐。若最后一个 rank 样本数不同，简单平均不再自动等价，不能把本
实验结论无条件推广到不等 batch。
