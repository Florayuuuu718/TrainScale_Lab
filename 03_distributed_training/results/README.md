# 03 结果契约

## 正式结果

| 文件 | 内容 |
|---|---|
| `environment_sm120.json` | WSL/PyTorch/Gloo/NCCL/GPU 数量 |
| `gloo_semantics_cpu.json` | 2 rank 环境变量、AllReduce、Broadcast |
| `sampler_sharding_cpu.json` | 4 rank、两个 epoch 的索引与 coverage |
| `gradient_equivalence_cpu.json` | 4 rank DDP 对 global-batch reference |
| `checkpoint_resume_cpu.json` | 2 rank 连续/部分/恢复对照 |
| `scaling_cpu.json` | CPU 1/2/4 rank strong/weak scaling |
| `scaling_nccl_sm120.json` | NCCL world=1 实测与 2/4/8 unavailable |
| `ddp_profile_cpu.json` | 2 rank Gloo communication rows 与 trace 路径 |
| `module03_summary.json` | 所有正式源文件 SHA-256 与总门 |
| `module03_acceptance_sm120.json` | Windows/WSL 测试与多 GPU 硬件边界 |

`results/raw/` 保存练习 JSON、`.pt` checkpoint 和 Profiler trace，进入 Git ignore。
正式 JSON 只保存小型摘要；失败、unavailable 和硬件数量是有效证据，不能删除后
只保留最快 case。

## 每条 scaling 记录至少包含

- mode、world size、backend、device；
- local/global batch、warm-up、measured steps；
- 最慢 rank elapsed、global samples/s；
- speedup、scaling efficiency；
- CUDA 时的 peak allocated memory；
- success/failed/unavailable 和不可用原因；
- 环境、配置、commit 和工作区状态。

本机只有一张 GPU，所以 `scaling_nccl_sm120.json` 的 2/4/8 world size 没有
latency/throughput 字段。这是正确的缺失表达，不是未记录的 0。
