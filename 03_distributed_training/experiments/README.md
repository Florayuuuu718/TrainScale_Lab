# 03 实验索引

按顺序执行，不要从 scaling 图开始：

1. [Process group 与 rank](00_process_group.md)
2. [DistributedSampler 数据分片](01_distributed_sampler.md)
3. [DDP 梯度同步等价性](02_gradient_sync.md)
4. [Rank-0 checkpoint 与精确恢复](03_checkpoint_resume.md)
5. [CPU strong/weak scaling](04_cpu_scaling.md)
6. [NCCL 单/多 GPU scaling](05_nccl_scaling.md)
7. [DDP communication Profiler](06_ddp_profiler.md)

每份都包含为什么做、名词、一般预期、源码入口、终端命令、预期输出、实际结果、
理论解释和有限结论。第一次写 `results/raw/tutorial/`；只有固定配置、完整环境和
正确性门均通过后，才把摘要归档为正式结果。

