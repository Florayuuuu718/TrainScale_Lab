# 07 最终实验报告：什么时候选择 DDP、FSDP2 或 TP

## 先区分三个问题

- **DDP** 复制完整模型，每张卡处理不同数据，再 AllReduce 梯度。它解决吞吐扩展，不减少每卡模型状态。
- **FSDP2** 切分参数、梯度和 optimizer state，在计算前 AllGather 所需参数、反向后 ReduceScatter 梯度。它主要解决训练状态显存。
- **TP** 切分单层矩阵或 attention heads，输入常常复制、局部计算后通过 collective 合并。它解决单层参数/activation 放不下或数据并行 local batch 过小的问题。

三者不是从低级到高级的排行榜。应先识别瓶颈，再选择引入哪一种通信。

术语解释见 [分布式训练与通信术语表](../../docs/concepts/distributed-systems-glossary.md)。

## 运行前预测

- FSDP2/TP 的局部参数或状态应随 world size 增大而缩小；
- allocator 峰值不会严格按 `1/world_size` 下降，因为还包含 activation、临时参数和 buffer；
- tiny 模型的计算量不足以摊薄新增 collective，因此分片很可能比 DDP 慢；
- 更细的 FSDP wrap 可能进一步降低峰值，但会增加参数重建次数和通信代价。

## 正确性证据

CUDA/NCCL 的 2/4 GPU FSDP2 和原生 TP preflight 全部通过。FSDP2 最大一步更新误差 `1.49e-8`，DCP 恢复后下一步误差为 0；原生 TP 最大误差 `5.96e-8`。FSDP2 artifact 同时记录 `Shard(0)` placement 和 world-size gradient divide factor。

PyTorch 2.8 的 CPU/Gloo FSDP2 探针没有匹配全局 reference，即使显式设置 divide factor 仍不可用；同一代码在 CUDA/NCCL 下误差约 `1e-8`。因此 CPU 结果保留为后端兼容性限制，CUDA/NCCL preflight 才是本章 GPU gate。不能通过放宽容差把后端差异藏掉。

## FSDP2：状态切小了，为什么总显存只降一点

4 GPU medium：

| 策略 | 吞吐 | 峰值显存 | 相对 DDP 显存 |
|---|---:|---:|---:|
| DDP | 3,216 samples/s | 48.5 MiB | baseline |
| FSDP2 root wrap | 1,901 samples/s | 44.0 MiB | -9.37% |
| FSDP2 layer wrap | 1,143 samples/s | 39.2 MiB | -19.31% |

理论上 4 卡可把持久化分片状态降到约 1/4，但 allocator peak 还包含 activation、临时 full parameter、collective buffer 和固定框架开销。这个模型太小，持久状态不是峰值显存的主体，所以实测只降 9%–19%。layer wrap 更省峰值，却引入更多 AllGather/ReduceScatter，吞吐代价更高。

这正是理论下界与实测 peak 的区别：理论用于预测趋势，allocator measurement 用于决策。

## TP：局部参数变小了，为什么没有加速

| TP world size | 局部参数 | 吞吐 |
|---:|---:|---:|
| 1 reference | 0.500 MiB | 9,554 samples/s |
| 2 | 0.251 MiB | 1,367 samples/s |
| 4 | 0.126 MiB | 1,351 samples/s |

分片完全按预期缩小，但 tiny MLP 的本地 GEMM 很短，collective 和分布式调度远大于节省的计算；4 卡甚至没有优于 2 卡。Profiler 也观察到 DDP 的 AllReduce、FSDP2 的 AllGather/ReduceScatter、TP 的 AllReduce，和各自算法一致。

## 可迁移的选择顺序

1. 模型和目标 batch 能放入单卡：先建立 DDP baseline。
2. 参数、梯度、optimizer state 是显存瓶颈：比较 FSDP2 root/layer wrap 的峰值与通信代价。
3. 单层本身放不下，或 DP 导致 local batch 太小：在高速机内互联上考虑 TP。
4. 只有两个瓶颈同时存在且 GPU 数足够，才考虑 TP×DP/FSDP；2D 并行不是本项目 v1.0 gate。

当前 preset 无法在 24 GiB 4090D 上自然制造可信 OOM，因此没有用无关进程抢显存伪造“DDP OOM、FSDP 可运行”。这项边界被记录为 unavailable，不影响已完成的 correctness、显存和通信代价比较。

机器可读摘要见 [`../results/module07_final_summary.json`](../results/module07_final_summary.json)。
