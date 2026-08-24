# 02 实验索引

这些文档已完成本机正式验收实验。driver 610.88 下，默认 cu129/Triton 环境通过探针和最终 15 项 GPU 测试；14 组通用 forward、41 条 CUDA/Triton 变体、8 个 LayerNorm phase comparison、8 个 MatMul candidate run 与 8 组 Profiler 已落盘。driver 577.05 的 segmentation fault 仍作为历史失败样本保留。

推荐顺序：

1. [Benchmark 方法校准](00_benchmark_protocol.md)
2. [Vector Add：mask 与内存带宽](01_vector_add.md)
3. [ReLU：逐元素融合与 launch 开销](02_relu_fusion.md)
4. [Softmax：稳定 reduction 与行融合](03_softmax.md)
5. [LayerNorm：统计量、仿射与 backward](04_layernorm.md)
6. [MatMul：tiling、autotune 与 Tensor Core](05_matmul.md)
7. [Attention：避免物化中间矩阵](06_attention.md)
8. [CUDA C++ / Triton / PyTorch 对照](07_cuda_triton_comparison.md)
9. [Profiler 与 roofline 总结](08_profiler_roofline.md)

不要只阅读仓库保存的结果。先按 [02 模块入口](../README.md#5-像-01-一样从命令行逐项做实验)
完成一次公共环境准备，再打开每份实验中的“跟着做”章节。每一份都明确给出源码
位置、对应 correctness test、快速命令、预期终端状态、结果阅读命令和正式参数。
建议第一次用 `samples=5, warmup=2`；流程跑通后才用正式的 `21/10`，这样拼错
路径或环境不兼容时不会先等待一轮长实验。

每个实验遵守同一条链：

`数学定义 → 输入域 → reference → correctness → 机制预测 → benchmark → profiler → 有限结论`

新实验复制 [实验模板](experiment-template.md)。大型 trace 放到 `results/raw/`，报告只链接可复现命令和摘要。

## 当前状态

| 实验 | 已获得证据 | 尚未覆盖 |
|---|---|---|
| 00 | 新旧驱动对照、隔离探针、stable/nightly 历史快照与 stable 最终测试 | nightly 仅作兜底 |
| 01 | 三 dtype、ragged、PyTorch/Triton/CUDA scalar/packed、Profiler | 多 GPU 型号属于扩展 |
| 02 | 同形状 Add+ReLU forward/backward、融合性能与 Profiler | broadcast 与更多 dtype |
| 03 | 17–4097 行宽、Triton 两策略、CUDA serial/block | 更多生产级策略属于扩展 |
| 04 | y/mean/rstd/dx/dweight/dbias、dtype/eps、forward/backward | 两阶段参数梯度优化属于扩展 |
| 05 | ragged backward、FP32 reference、四候选有限 autotune、Profiler | 更大搜索空间属于扩展 |
| 06 | 4 种 head dim、causal/non-causal、显式错误边界与 Profiler | backward 属于进阶项 |
| 07 | CUDA 13.0 构建、9 case/41 路径统一对照 | PyTorch extension ABI 不在本实验范围 |
| 08 | 四类算子的 PyTorch/Triton Profiler 与 roofline 解释 | Nsight Compute 深层计数器 |
