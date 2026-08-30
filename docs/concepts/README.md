# 概念文档

概念文档回答“为什么”，阶段 README 回答“怎么运行”，源码回答“具体怎么实现”。

基础概念：

- [PyTorch 训练基础](pytorch-training-basics.md)：synthetic dataset、DataLoader、batch、模型、loss、梯度、optimizer、epoch、validation 和 overfit test。
- [checkpoint 完整状态](../checkpoint-contract.md)：保存与恢复一次训练需要哪些状态。
- [分布式训练与通信术语表](distributed-systems-glossary.md)：rank、collective、带宽、bucket、
  overlap、DDP/FSDP2/TP、拓扑和 artifact 的直观解释。

04–07 的系统概念直接与实验报告合并，避免把公式和证据拆成两套重复文档：

- [通信曲线与 DDP](../../04_nccl_benchmark/experiments/06_final_report.md)：`α + bytes/bandwidth`、
  `algbw`/`busbw`、拓扑与 strong/weak scaling；
- [Centralized 与 Ring](../../05_tiny_collective/experiments/04_final_report.md)：轮数、通信量、
  root 热点和生产实现差异；
- [Bucket 与 Overlap](../../06_training_engine/experiments/06_final_report.md)：reducer 生命周期、
  collective 粒度、AMP 状态机和“overlap 不等于加速”；
- [DDP、FSDP2 与 TP](../../07_parallelism/experiments/01_final_report.md)：复制状态、分片状态、
  切分单层和策略选择边界。

建议先运行一次 CPU smoke test，再阅读概念文档，然后打开对应源码对照。只读概念而不运行代码，很难形成训练系统的直觉。
