# 03 · Distributed Training 验收清单

03 采用“本机正确性主线 + 云端单机多 GPU 性能扩展”边界。硬件不足必须写
`unavailable`，不能把未运行项标 success，也不能用同一 GPU 的多个 rank 伪造
多 GPU scaling。

| ID | 验收项 | 证据 | 状态 |
|---|---|---|---|
| 03-01 | 冻结环境与 torchrun/Gloo/NCCL 契约 | 环境 JSON、WSL 路线、一进程一 GPU 边界 | 已完成 |
| 03-02 | rank/process-group 语义 | 2 rank AllReduce/Broadcast 与 rank JSON | 已完成 |
| 03-03 | DistributedSampler | 4 rank coverage、padding、set_epoch | 已完成 |
| 03-04 | DDP 梯度等价 | 4 rank 对 global-batch reference，误差进入 JSON | 已完成 |
| 03-05 | 参数一致性与 checkpoint | rank-0 writer、连续/恢复最终参数一致 | 已完成 |
| 03-06 | CPU strong/weak scaling | 1/2/4 rank、最慢 rank 计时、speedup/efficiency | 已完成 |
| 03-07 | NCCL DDP 能力门 | RTX 5060 world=1 forward/backward/计时/显存 | 已完成 |
| 03-08 | 多 GPU 硬件边界 | world=2/4/8 自动检测并记录 unavailable 原因 | 已完成 |
| 03-09 | DDP communication Profiler | 2 rank 均捕获 5 次 Gloo AllReduce | 已完成 |
| 03-10 | 测试、教程与结果归档 | CPU CI、Linux torchrun 集成、00–06 教程、SHA-256 汇总 | 已完成 |
| 03-11 | 云端单机四卡实验 | 4×4090D，1/2/4 GPU strong/weak 各重复三次 | 已完成 |
| 03-12 | 云资源教程与止费闭环 | 选型、bundle、环境路线、smoke、哈希下载、关机 | 已完成 |

## 云端扩展验收结果

- 同一 `gpu_scaling.toml` 已补采 1/2/4 GPU，8 GPU 因实例只有四卡保留 unavailable；
- 环境、GPU 型号、驱动、PCIe/NUMA 拓扑和无 NVLink 边界均已归档；
- 三次运行经脚本校验 config、commit、环境和 case 集合后取中位数；
- 本地 `scaling_nccl_sm120.json` 与云端 `scaling_nccl_4x4090d.json` 分开保存；
- strong 四卡 speedup `0.510×`，weak 四卡 `2.018×` / efficiency `50.4%`；
- 20-step 窗口的抖动与小模型通信占比进入教程，未夸大为生产性能结论。

这些扩展不会改变本地已经证明的 DDP correctness。03 现在可以宣称完成单机
1/2/4 GPU 教学性能验收；8 GPU、NVLink 主机和真实大模型仍是后续可选扩展。

历史本机发布证据见
[`module03_acceptance_sm120.json`](../03_distributed_training/results/module03_acceptance_sm120.json)：
Windows `35 passed, 2 skipped`，WSL 03 测试 `10 passed`（含真实 2 rank Gloo），
单 GPU NCCL smoke 通过。最终本地+云端发布证据见
[`module03_acceptance.json`](../03_distributed_training/results/module03_acceptance.json)。
