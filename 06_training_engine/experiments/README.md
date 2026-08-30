# 06 实验导航

本章沿着 reducer 生命周期逐步增加复杂度：

1. [架构边界](00_architecture.md)：哪些能力复用 01/03，哪些由 06 新增；
2. [Reducer 演进](01_reducer_evolution.md)：bulk → per-parameter → bucket sync/async；
3. [本地正确性](02_local_correctness.md)：global-batch reference、unused 参数和 accumulation；
4. [GPU 消融](03_gpu_ablation.md)：一次只改变策略、bucket、AMP 或 accumulation；
5. [Overlap timeline](04_overlap_timeline.md)：区分 launch candidate 与真实 kernel overlap；
6. [AMP、累积和 checkpoint](05_amp_accum_checkpoint.md)：验证状态机而非只验证启动；
7. [最终报告与 1 MiB 延伸](06_final_report.md)：理解“真实 overlap 仍可能更慢”。

正确阅读方式是把 benchmark 与 profiler 配对：前者回答是否更快，后者解释为什么。
