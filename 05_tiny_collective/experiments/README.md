# 05 实验导航

按“先画消息流，再证明数学正确，最后测性能”的顺序学习：

1. [实验协议](00_protocol.md)：冻结指标、重复和 artifact 规则；
2. [算法与通信量](01_algorithms.md)：centralized 与 ring 的轮次、chunk 和复杂度；
3. [CPU/Gloo correctness](02_cpu_correctness.md)：用 ragged chunks 验证调度；
4. [GPU/NCCL 对照](03_gpu_comparison.md)：区分算法结构与生产实现；
5. [最终报告](04_final_report.md)：用 2/4 GPU 结果和一般通信规律对照。

不要跳过 correctness 直接看 GB/s，也不要用 8-byte 等噪声主导的结果给算法排名。
