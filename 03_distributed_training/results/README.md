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
| `scaling_nccl_4x4090d.json` | 云端三次 1/2/4 GPU 结果的中位数、离散程度与源哈希 |
| `ddp_profile_cpu.json` | 2 rank Gloo communication rows 与 trace 路径 |
| `evidence/cloud_4x4090d/` | 云端环境、拓扑、smoke 和三次未聚合正式 JSON |
| `module03_summary.json` | 本地与云端正式源文件 SHA-256 与总门 |
| `module03_acceptance_sm120.json` | 历史本地 Windows/WSL 单卡验收 |
| `module03_acceptance.json` | 最终 03 验收与云端 1/2/4 GPU 结论 |

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
latency/throughput 字段。这是正确的缺失表达，不是未记录的 0。随后云端四卡结果
补齐 1/2/4 GPU；8 GPU 仍保持 `unavailable`。

云端正式值不是挑选最好的一次。`aggregate_scaling_runs.py` 要求三次运行的 config、
commit、环境、case 集合和干净 worktree 一致，并对每个 case 的吞吐、最慢 rank
时间和显存取中位数；speedup/efficiency 从中位吞吐重新计算。原下载压缩包 SHA-256
为：

```text
63b0bb1efc17313cfd9df381afe67281d9daa2eb634a4fe570861ca7f3077e54
```

正式四卡结论限定为这次短 synthetic MLP：strong 多卡因工作粒度太小而变慢；weak
四卡中位 speedup `2.018×`、efficiency `50.4%`。它证明真实 NCCL 多卡执行和拓扑
影响，不是生产模型或其他 GPU 主机的绝对性能排名。
