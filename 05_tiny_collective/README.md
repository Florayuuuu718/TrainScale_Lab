# 05 · TinyCollective

> 状态：尚未实现。本目录是教学版 collective 实现的唯一入口。

计划用 `torch.distributed.send/recv` 实现 Naive AllReduce 与 Ring AllReduce，并与 `torch.distributed.all_reduce` 对照正确性、通信量和性能。

进入本模块前，请先完成 [04 · NCCL Benchmark](../04_nccl_benchmark/README.md)。
