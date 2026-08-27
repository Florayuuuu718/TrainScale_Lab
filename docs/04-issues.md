# 04 · NCCL Performance Lab 验收清单

04 必须把 collective 曲线与 03 的 DDP workload 连接起来，不能只提交 `nccl-tests`
截图。状态只使用“未开始、进行中、已完成、阻塞”。

| ID | 验收项 | 依赖 | 完成证据 | 状态 |
|---|---|---|---|---|
| 04-01 | 冻结范围、环境、拓扑和硬件门 | 03 完成 | ENVIRONMENT、GPU/NUMA/topology JSON、2 GPU 最低门 | 进行中：本地契约完成，待多 GPU 证据 |
| 04-02 | 公共 artifact envelope 与统计工具 | 04-01 | schema/hash/percentile/unavailable CPU tests | 已完成 |
| 04-03 | 固定 `nccl-tests` 构建与解析 | 04-01 | pinned commit、可复现 build、stdout parser tests | 进行中：代码/测试完成，待 Linux 实编译 |
| 04-04 | 四种 collective correctness/smoke | 04-02..03 | AllReduce/AllGather/ReduceScatter/Broadcast 小配置 | 进行中：runner/config 完成，待 2 GPU smoke |
| 04-05 | 消息大小正式扫描 | 04-04 | latency/algbw/busbw、三次重复、结构化结果 | 未开始 |
| 04-06 | world size 与拓扑对照 | 04-05 | 同 NUMA/跨 NUMA或等价可控路径实验 | 未开始 |
| 04-07 | 03 DDP bridge 与 GPU timeline | 04-05 | 同环境复跑、实际梯度/bucket 大小、trace 摘要 | 进行中：payload/runner 完成，待 2/4 GPU trace |
| 04-08 | 参数单变量实验 | 04-06..07 | 一个 NCCL/拓扑变量，固定其余条件 | 未开始 |
| 04-09 | 教程、失败记录和结果汇总 | 04-04..08 | experiments、raw hash、summary JSON | 进行中：教程/聚合器完成，待正式结果 |
| 04-10 | 模块发布验收 | 04-01..09 | ruff/mypy/pytest、GPU gates、acceptance JSON | 未开始 |

## 不阻塞完成的扩展

- 8 GPU、NVLink 主机、多节点或 InfiniBand；
- NCCL 全源码阅读和算法强制选择的完整矩阵；
- 跨云厂商或跨 GPU 型号排行榜。
