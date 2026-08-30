# 分布式训练与高性能通信术语表

这不是需要背诵的字典。第一次遇到一个词时，先读“直观理解”，运行对应实验后再回来读
“在本项目中怎样验证”。

## 进程与设备

| 术语 | 直观理解 | 在本项目中怎样验证 |
|---|---|---|
| process | 一个正在运行的 Python 程序，有独立内存 | 03 用 `torchrun` 启动多个 worker |
| rank | 一个进程在通信组中的编号，从 0 开始 | 03 检查每个 rank 的数据和参数 |
| world size | 同一通信组中 rank 的总数 | `world_size=4` 表示四个参与进程 |
| local rank | 当前主机内的进程编号，通常绑定一张 GPU | 单机四卡中 local rank 0–3 对应 GPU 0–3 |
| backend | 执行分布式通信的后端 | CPU 常用 Gloo，NVIDIA GPU 性能实验使用 NCCL |
| process group | 能够互相参与 collective 的 rank 集合 | 03 的第一个实验显式初始化和销毁 group |

一张 GPU 通常对应一个训练进程，但“rank”和“GPU”不是同一个概念。CPU/Gloo 也可以有多个
rank；TP 或二维并行还可能建立不同的子通信组。

## Collective

collective 是一组 rank 共同参加的通信操作。所有 rank 必须以兼容的顺序和张量形状调用，
否则可能报错或一直等待。

| 操作 | 最终结果 | 常见用途 |
|---|---|---|
| Broadcast | root 的数据发送给所有 rank | 分发初始化状态 |
| Reduce | 多个输入被求和等操作后放到 root | 教学版 centralized AllReduce 的前半段 |
| AllReduce | 所有输入归约，每个 rank 都得到结果 | DDP 梯度同步 |
| AllGather | 每个 rank 的不同分片被拼到所有 rank | FSDP 计算前重建参数、TP 收集输出 |
| ReduceScatter | 先归约，再把结果切分给不同 rank | FSDP 梯度分片、Ring AllReduce 前半段 |
| AllToAll | 每个 rank 向每个 rank 发送不同分片 | MoE token dispatch 等扩展场景 |

AllReduce 可以实现成 Reduce + Broadcast，也可以实现成 ReduceScatter + AllGather。数学结果
相同，但热点、通信轮数、临时内存和拓扑适应性不同，这正是 05 要验证的内容。

## 性能指标

| 术语 | 含义 | 常见误区 |
|---|---|---|
| latency | 一次操作从开始到完成的时间 | 小消息不应只用 GB/s 排名 |
| throughput | 单位时间处理的样本或字节 | 必须说明 global 还是 per-rank 口径 |
| `algbw` | 有效载荷大小除以操作时间 | 不等于链路真实搬运量 |
| `busbw` | 按 collective 通信量模型归一化的带宽 | 只能在明确公式和同环境下比较 |
| warm-up | 不进入统计的预热步骤 | 首次编译、缓存和初始化会污染结果 |
| median | 多次测量排序后的中间值 | 比单次最快值更稳健，但仍要报告波动 |
| relative range | `(max-min)/median` | 范围过大时不能给高精度 speedup |
| speedup | baseline 时间或吞吐与新方案的比值 | 必须保证 workload 和 batch 语义一致 |
| scaling efficiency | speedup 除以设备数增长倍数 | strong 与 weak scaling 不能混算 |

通信时间常用 `T ≈ α × rounds + bytes / bandwidth` 建立直觉。`α` 表示每轮启动和同步等
固定开销；它能解释为什么很多小 collective 比少量大 collective 更贵，但不能替代实测。

## Scaling、Bucket 与 Overlap

- **Strong scaling**：固定 global workload，增加 GPU。每张卡的计算越来越少，固定通信成本
  更难摊薄。
- **Weak scaling**：每张卡保持相似 workload，增加 GPU 时 global workload 同时增大。它回答
  系统能否随资源扩展处理更多工作。
- **Gradient bucket**：把多个参数梯度放进一块连续 buffer 后统一通信，减少 collective 次数。
- **Async collective**：发起通信后先得到 handle，稍后等待完成。它只是创造并行机会。
- **Overlap**：通信 kernel 与 backward compute kernel 的时间区间真实重叠。只能从 GPU
  timeline 证明；发生 overlap 也不保证总 step 更快。

06 的 1 MiB 实验正好展示了这个区别：更小 bucket 产生了真实 overlap，但 collective 数量
增加，最终吞吐下降。

## DDP、FSDP2 与 TP

| 方法 | 切分什么 | 主要解决什么 | 新增通信 |
|---|---|---|---|
| DDP | 数据；模型完整复制 | 训练吞吐 | 梯度 AllReduce |
| FSDP2 | 参数、梯度、optimizer state | 训练状态显存 | 参数 AllGather、梯度 ReduceScatter |
| TP | 单层矩阵、hidden dimension 或 attention heads | 单层容量与计算 | 层内 AllReduce/AllGather 等 |

- **DTensor**：同时描述 tensor 的全局形状、设备 mesh 和 placement；本地进程只持有自己的分片。
- **DeviceMesh**：用一维或多维网格组织设备，是 TP/FSDP 组合的坐标系。
- **Placement**：`Shard(dim)` 表示沿某维切分，`Replicate()` 表示每个 rank 都有完整副本。
- **DCP**：Distributed Checkpoint。每个 rank 保存分片和 metadata，恢复时按新进程组重建状态。
- **wrap policy**：决定 FSDP 按整个模型还是按层切分。更细粒度可能降低峰值，却增加通信次数。

选择顺序不是“DDP < FSDP2 < TP”。模型能放下时先用 DDP；训练状态成为显存瓶颈时考虑
FSDP2；单层本身放不下时才考虑 TP。07 用 tiny 模型得到“分片更省但更慢”的结果，正是在
验证选择必须由瓶颈驱动。

## 拓扑与软件栈

- **PCIe/PHB/NODE/SYS**：`nvidia-smi topo -m` 描述 GPU 间经过的硬件路径。标签只提供假设，
  不能直接当作性能结论。
- **NVLink**：GPU 间专用高速互联；并非所有消费级或云主机都有。
- **NUMA**：CPU 和内存存在不同 locality。跨 NUMA 访问可能经过 CPU interconnect。
- **NCCL algorithm/protocol**：Ring、Tree 和 Simple、LL 等内部选择。Auto 是默认基线，强制
  设置主要用于诊断。
- **driver/runtime/toolkit**：驱动决定硬件支持上限，PyTorch wheel 带 CUDA runtime，`nvcc`
  属于 toolkit。三者版本号不必完全相同。

04 的参考主机没有 NVLink，也未观察到稳定的 PHB/NODE/SYS 排名。这不表示拓扑普遍无影响，
只表示该环境、该消息范围和 NCCL 路径下的差异不足以支持推广。

## 正确性与证据

- **reference**：更简单、可信但不一定快的实现，用来判断优化版本的数值结果。
- **correctness gate**：性能实验前必须通过的数值、状态或数据覆盖检查。
- **artifact**：包含配置、环境、commit、指标和 raw hash 的结构化结果。
- **trace**：Profiler 导出的时间线，用来解释 kernel 和 collective 发生的顺序。
- **SHA-256**：文件内容摘要。下载前后相同，说明文件没有被截断或悄悄改变。
- **unavailable**：当前平台或后端不具备能力；不是 0 性能，也不等于其他后端失败。

可靠结论的顺序始终是：correctness 通过 → 测量协议固定 → 重复值可接受 → 机制证据吻合 →
写明适用边界。缺少其中任何一项，都应缩小措辞，而不是扩大结论。
