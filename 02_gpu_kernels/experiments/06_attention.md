# 实验 06：Attention——为什么不想保存完整分数矩阵

> 状态：causal/non-causal correctness 与三组 PyTorch SDPA/Triton fused forward 对照已完成；backward 属于进阶项。

## 为什么做

Attention 先算每个 query 与所有 key 的相似度，再用 Softmax 得到权重，最后加权 value：

`Attention(Q,K,V) = softmax(QK^T / sqrt(D))V`

显式写法会生成 `[heads, sequence, sequence]` 分数矩阵。sequence 翻倍时，这个中间矩阵的元素数变成 4 倍。融合实现把分块结果留在片上存储，避免完整写回显存。

## 小白名词

- **Q/K/V**：query、key、value 三组向量。
- **head**：一组独立 Attention；多头让模型从不同子空间建模。
- **causal mask**：禁止当前位置看到未来 token。
- **SDPA**：PyTorch 的 scaled dot-product attention 接口，可选择融合 backend。
- **materialize**：真正分配并写出一个中间 tensor。
- **O(S²)**：sequence 长度 S 翻倍，中间分数矩阵约变 4 倍。

## 一般预期

SDPA 通常比显式 `matmul → softmax → matmul` 更快、额外显存更少；sequence 越长，避免 O(S²) 中间矩阵越重要。小 sequence 仍可能由固定成本主导。

## 跟着做：运行 causal 与 non-causal Attention

实现位于 [`triton_ops.py`](../trainscale_kernels/triton_ops.py) 的
`_attention_forward_kernel` 和 `attention`。PyTorch reference 使用
`scaled_dot_product_attention`；case 配置同时包含 non-causal 与 causal。

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k attention

.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator attention --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/06_attention.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/06_attention.json
```

pytest 会覆盖 head dim 16/32/64/128、causal/non-causal，还会验证过长 sequence
和非法 head dim 会明确报错。benchmark 运行 3 个配置 case、6 条实现路径，末尾
应是 `all_cases_passed=True`。本模块只验收 fused forward；不要从这些命令推断
Attention backward 已实现。正式复现使用 `21/10`。

## 实际结果

FP16、1 batch，PyTorch 端使用 SDPA：

| heads×S×D | causal | PyTorch median | Triton median | Triton 相对 PyTorch | Triton 最大绝对误差 |
|---|---|---:|---:|---:|---:|
| 2×64×32 | 否 | 16.365 µs | 23.485 µs | 0.697× | 4.88e-4 |
| 8×128×64 | 否 | 13.819 µs | 91.275 µs | 0.151× | 4.88e-4 |
| 8×128×64 | 是 | 17.683 µs | 94.244 µs | 0.188× | 9.77e-4 |

独立测试还以 sequence=33 验证 causal/non-causal，并拒绝当前教学实现不支持的 sequence>256。相对误差在 reference 接近 0 时会被放大，因此验收同时看绝对容差 `atol=3e-2`，三组均通过。

## 理论分析

融合避免完整物化 `[H,S,S]` score matrix 的理论仍成立，但“融合”只是算法结构，不保证实现已经优化。Profiler 显示 PyTorch 使用 flash-attention kernel，20 次约 108.303 µs；教学 Triton kernel 20 次约 1690.004 µs。生产 kernel 对分块、在线 Softmax、Tensor Core、寄存器和调度做了大量优化，朴素融合实现无法仅凭少写中间矩阵追平。

从 S=64 到 S=128，Triton 延迟从 23.5 增到 91.3 µs，接近 Attention 的二次复杂度趋势；PyTorch SDPA 在这个小规模区间仍主要受固定成本和高度优化 kernel 影响。

## 结论与收尾

教学 Triton Attention 已正确，但三组都明显慢于 PyTorch flash SDPA，large case 约慢 6.6 倍。这个负结果很有价值：理解 FlashAttention 的“避免 O(S²) 写回”只是第一步，真正的高性能还依赖硬件友好的分块与流水；不能把“fused”当作“fast”的同义词。
