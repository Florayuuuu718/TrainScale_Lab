# GPU/NCCL 性能对照实验

## 问题

在相同 GPU、world size、dtype、消息大小、warm-up 和测量次数下，教学版 Python ring 与
PyTorch/NCCL AllReduce 的延迟和有效带宽差多少？差距如何随消息大小变化？

## 控制变量

- 单机 world size 2/4，FP32；
- 算法 `centralized`、`ring` 与 `torch`，形成最小实现、结构优化、生产库三层对照；
- 每个组合独立 torchrun job，固定 3 次重复并取中位数；
- 每次测量前同步，耗时取所有 rank 中的最大值；
- 小消息到 64 MiB，并包含 04 DDP payload 10,494,976 bytes；
- 所有性能样本先通过与 NCCL reference 的 correctness 检查。

## 证据与解释边界

每个 job 保存命令、日志和逐 rank JSON，总 artifact 保存文件哈希。报告应画消息大小—延迟与
消息大小—bus bandwidth 曲线，不只报告跨尺寸平均值。短消息主要暴露启动与 Python 调度开销，
大消息平台区才更接近链路吞吐；本实验不能证明 NCCL 内部采用了与教学实现相同的 schedule。

优化假设是 ring 消除 centralized 的根节点集中收发瓶颈，尤其在 4 rank 与大消息时更有优势。
验收不强制“ring 必须更快”：若 Python 调度、逐轮同步或节点 P2P 限制抵消理论优势，应将其作为
瓶颈结论，并用 NCCL 曲线说明生产实现还做了哪些教学版没有覆盖的优化。

若租用节点禁用 GPU P2P，应结合 04 的 transport/topology 证据解释，不把结果外推到 NVLink、
多节点或其他 GPU。
