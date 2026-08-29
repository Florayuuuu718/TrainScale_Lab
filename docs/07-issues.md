# 07 — FSDP2 / Tensor Parallel 验收清单

| ID | 验收项 | 完成证据 | 当前状态 |
|---|---|---|---|
| 07-01 | 冻结范围、配置与硬件门 | 严格 TOML；2/4 rank；4 GPU/NCCL gate | 已完成 |
| 07-02 | 状态显存模型 | 参数/梯度/Adam/activation 下界及边界声明 | 已完成（本地） |
| 07-03 | 最小 FSDP2 correctness | 真实 `fully_shard`、DTensor placement、reference 一步更新 | 已完成（CPU/Gloo）；待 CUDA 复验 |
| 07-04 | FSDP2 wrap/shard 实验 | root/layer wrap 的吞吐、峰值显存与 trace | 已实现；待 4 GPU |
| 07-05 | 可信 OOM→可运行实验 | 配置驱动的显存转折；不能形成时为 unavailable | 待 GPU，非强制造 OOM |
| 07-06 | distributed checkpoint/resume | DCP 分片文件及恢复后下一步一致 | 已完成（CPU/Gloo）；待 CUDA 复验 |
| 07-07 | 最小 TP correctness | 自定义与原生 Colwise/Rowwise；2/4 rank shape/placement | 已完成（CPU/Gloo）；待 CUDA 复验 |
| 07-08 | attention/MLP TP 实验 | 输出/梯度/更新；吞吐、显存和 AllReduce trace | correctness 已完成；性能待 4 GPU |
| 07-09 | 并行策略选择树 | DDP/FSDP2/TP 的瓶颈、代价和选择条件 | 已完成 |
| 07-10 | 2D TP×FSDP/DP | 仅在 07A/07B 暴露明确需求后开启 | 未启用（可选） |
| 07-11 | 模块发布验收 | 本地 gates、GPU artifacts、SHA-256、全仓测试 | 本地 gates 已完成；GPU 待统一租卡 |

## 本地验收结论

- 自定义 MLP/attention TP 在 2/4 rank 下完成输出、梯度分片和一步更新对齐。
- 真实 FSDP2 参数在 forward 前即为 `Shard(0)` DTensor；SGD momentum 的一步更新与 reference 对齐，DCP 恢复后下一步完全一致。
- 一次 AdamW 探针暴露了近零 bias 在首步符号更新下的敏感性，因此 correctness probe 改用 SGD momentum。这个变化用于获得数值稳定的分片语义判据，不应被解释为 AdamW 不支持 FSDP2。
- PyTorch 2.8 CPU/Gloo 的默认 reduce-scatter 缩放未产生预期的全局平均梯度；探针与正式 benchmark 已显式设置 gradient divide factor 为 world size，并要求 artifact 记录所用 API。
- 原生 TP 的 loss 不应再按 TP world size 缩放；输入和输出均复制时，框架 autograd 已处理 placement 语义。教学版自定义 TP 因显式 collective 的梯度行为仍需该缩放。两者均由 reference 一步更新验证。

## 尚不能宣称

- 不能用 CPU/Gloo 结果宣称 CUDA 显存节省或 NCCL 加速。
- 不能在没有 profiler trace 时宣称 FSDP2/TP 通信与计算发生有效重叠。
- 不能用理论显存下界替代 allocator peak；临时 full parameter、collective buffer 和保存的 activation 必须实测。
- 不能因单次 OOM 或单次最快结果给出策略结论；正式性能采用三次独立作业的中位数，并报告相对极差。

## 不阻塞 v1.0 的扩展

- 2D TP×DP/FSDP、pipeline parallel；
- 8 GPU、多节点和弹性 checkpoint；
- 自动并行搜索及完整生产框架复刻。
