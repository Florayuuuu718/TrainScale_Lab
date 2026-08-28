# 04 Experiments

按顺序推进，每份实验都先写预测，再运行命令：

1. [00 · Benchmark protocol](00_benchmark_protocol.md)：冻结版本、状态和重复口径；
2. [01 · Collective curves](01_collective_curves.md)：四种 collective 的消息曲线；
3. [02 · Topology pairs](02_topology_pairs.md)：解释 pair01/pair02/world4；
4. [03 · DDP bridge](03_ddp_bridge.md)：把 03 梯度载荷映射到通信曲线；
5. [04 · Multi-GPU campaign](04_multi_gpu_campaign.md)：一次租卡完成 smoke、三次正式
   运行、timeline、下载校验和关机；
6. [05 · Scaling stability follow-up](05_scaling_stability_followup.md)：修复过短测量窗口，
   用五次长窗口运行决定 speedup 是否可以正式报告。

实验 05 只在短窗口 scaling 的重复性不足时触发；它不要求重跑 NCCL 曲线或 profiler。
