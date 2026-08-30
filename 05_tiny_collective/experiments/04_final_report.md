# 05 最终实验报告：为什么 Ring 比中心化方案更可扩展

## 先学什么

AllReduce 的数学目标很简单：每个 rank 最终得到所有 rank 张量之和。难点是怎样组织通信。

- **Centralized**：所有 rank 把数据发给 root，root 求和后再广播。代码直观，但 root 的收发量随 world size 增长并形成热点。
- **Ring**：把数据切块，先 ReduceScatter，再 AllGather。每个 rank 每轮只和相邻 rank 通信，通信负载更均匀。
- **NCCL**：不是“另一条数学公式”，而是生产实现，包含拓扑发现、协议/算法选择、分块、融合和 GPU kernel 优化。

对于大小为 `N` 的数据和 `p` 个 rank，教学 ring 每 rank 的总收发量约为 `2N(p-1)/p`，但有 `2(p-1)` 轮。它减少热点，却增加轮数；因此大消息更能体现带宽优势，小消息可能被轮次和启动开销支配。

术语解释见 [分布式训练与通信术语表](../../docs/concepts/distributed-systems-glossary.md)。

## 运行前预测

- world size 较小时 centralized 简单直接，不一定明显落后；
- world size 增大后 root 热点会限制 centralized，大消息 ring 更有优势；
- Python ring 在小消息上可能很慢，因为每轮都有解释器、P2P 和同步开销；
- NCCL 应在多数正式 case 上更稳健，但不能预先断言它在每个消息点都最快。

## 正确性实验

CPU/Gloo 覆盖 world size 2/3/4，元素数 5/7/16/17，以及 centralized/ring，共 24 个 case，全部通过。5、7、17 不能总被 world size 整除，专门验证 ragged chunks；这比只测整除长度更能证明调度实现正确。

正确性先回答“是不是同一个 AllReduce”，性能再回答“代价如何”。两者不能互相替代。

## GPU 对照结果

4 GPU 代表值：

| 消息 | Centralized | Ring | PyTorch/NCCL |
|---:|---:|---:|---:|
| 10,494,976 bytes | 3.035 GB/s | 5.888 GB/s | 9.257 GB/s |
| 64 MiB | 3.069 GB/s | 7.617 GB/s | 9.317 GB/s |

在 64 MiB 时，教学 ring 是 centralized 的 2.48×，说明均匀分摊通信确实缓解了 root 热点；NCCL 又比教学 ring 高约 22.3%，说明生产性能不仅取决于“是不是 ring”，还取决于实现层次。

2 GPU 的 64 MiB ring 与 NCCL 都约为 7.57 GB/s。此时 ring 只有两端、轮数较少，Python 教学实现还能接近库实现；world size 增至 4 后，更多同步、P2P 调度和 Python 开销被放大。

8-byte 等小消息的相对极差很高，不参与算法排名。一般规律是：在固定启动延迟占主导时，吞吐指标会因为分母太小而失真，应优先读 latency。

## 和一般结论对照

实验支持“ring 比单 root 方案更容易扩展”，但不支持“ring 总是最快”。真实 collective 选择还取决于消息大小、GPU 数、拓扑、协议和库版本。04 的 Auto/Tree/Ring 对照也表明，强制单一算法不能成为跨环境建议。

本章真正得到的是一个分析框架：先画消息流和轮数，再算每 rank 通信量，最后用实测解释固定开销与带宽平台。这个框架可以继续用于 AllGather、ReduceScatter、MoE AllToAll 和多节点通信。

机器可读摘要见 [`../results/module05_final_summary.json`](../results/module05_final_summary.json)。
