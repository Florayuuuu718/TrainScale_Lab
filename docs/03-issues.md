# 03 · Distributed Training 验收清单

03 采用“本机可执行主线 + 多 GPU 硬件扩展”边界。硬件不足必须写
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

## 换到多 GPU 机器后的扩展验收

- 用同一 `gpu_scaling.toml` 补采 2/4/8 GPU strong/weak throughput；
- 记录 GPU 型号、互连拓扑、功耗、NCCL/driver 和 CPU/NUMA；
- 计算真实 speedup/efficiency，并用 Profiler/NCCL 证据解释拐点；
- 不覆盖本机单 GPU JSON，使用带硬件标识的新文件名。

这些扩展不会改变当前机器已经证明的 DDP correctness，但在宣称“03 完成了多 GPU
性能验收”前必须补齐。

本机发布证据见
[`module03_acceptance_sm120.json`](../03_distributed_training/results/module03_acceptance_sm120.json)：
Windows `35 passed, 2 skipped`，WSL 03 测试 `10 passed`（含真实 2 rank Gloo），
单 GPU NCCL smoke 通过。
