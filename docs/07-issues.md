# 07 · FSDP2 / Tensor Parallel 验收清单

07 使用 06 的同一 Transformer 和训练口径，先解释显存瓶颈，再选择切分方式。只有
“配置成功启动”而没有 reference 对照、显存分解和通信证据，不能标记完成。

| ID | 验收项 | 依赖 | 完成证据 | 状态 |
|---|---|---|---|---|
| 07-01 | 冻结 07A/07B 范围、DeviceMesh 与硬件门 | 06 完成 | README、环境、2/4 GPU 能力矩阵 | 未开始 |
| 07-02 | 状态显存模型与单卡 baseline | 07-01 | 参数/梯度/optimizer/activation 估算与实测 | 未开始 |
| 07-03 | 最小 FSDP2 correctness | 07-02 | 输出/loss/梯度/一步更新 reference | 未开始 |
| 07-04 | FSDP2 wrap/shard 实验 | 07-03 | AllGather/ReduceScatter、显存/吞吐对照 | 未开始 |
| 07-05 | 可信 OOM → 可运行实验 | 07-03 | 配置驱动 OOM；不足时明确 unavailable | 未开始 |
| 07-06 | distributed checkpoint/resume | 07-03 | 分片状态保存、恢复下一步一致 | 未开始 |
| 07-07 | 最小 Tensor Parallel correctness | 07-01..02 | Colwise/Rowwise、placement、局部 shape | 未开始 |
| 07-08 | attention/MLP TP 实验 | 07-07 | 通信、显存、吞吐与 scaling efficiency | 未开始 |
| 07-09 | 并行策略选择树 | 07-02..08 | DDP/FSDP2/TP 的瓶颈、代价和选择条件 | 未开始 |
| 07-10 | 2D TP×FSDP/DP | 07-04, 07-08 | 4 GPU 可选证据或 unavailable | 未开始（可选） |
| 07-11 | 模块与 v1.0 发布验收 | 07-01..09 | correctness、报告、acceptance、全路线审计 | 未开始 |

## 不阻塞 v1.0 的扩展

- 07-10 的 2D 组合并行（硬件不足时）；
- Pipeline Parallel、8 GPU、多节点和弹性 checkpoint；
- 自动并行搜索、完整 TorchTitan/Megatron 功能复刻。

