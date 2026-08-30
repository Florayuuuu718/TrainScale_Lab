# 06 · Mini Training Engine + Reducer Lab 验收清单

06 的新增价值是 reducer、bucketing 和 overlap。若某项工作只是重复 01/03 且没有
新的实验问题，应复用或建立适配层，而不是重新实现。

| ID | 验收项 | 依赖 | 完成证据 | 状态 |
|---|---|---|---|---|
| 06-01 | 冻结 engine/reducer 范围与共享边界 | 04/05 | 架构说明、明确复用和非目标 | 已完成 |
| 06-02 | Tiny Transformer 与单卡 baseline | 06-01 | 两种规模、overfit/一步更新/显存基线 | 已完成 |
| 06-03 | bulk AllReduce reference | 06-02 | 对 global-batch reference 的梯度与更新 | 已完成 |
| 06-04 | per-parameter hook reducer | 06-03 | 次数、顺序、None gradient correctness | 已完成 |
| 06-05 | deterministic gradient buckets | 06-04 | ownership/offset/view 单元测试 | 已完成 |
| 06-06 | bucketed synchronous reducer | 06-05 | 多 bucket size、数值与通信次数 | 已完成 |
| 06-07 | async bucket 与 handle 管理 | 06-06 | optimizer 前完成、异常快速失败 | 已完成 |
| 06-08 | backward/communication overlap | 06-07 | GPU timeline 直接证明 overlap | 已完成：1 MiB 延伸观察到真实 overlap |
| 06-09 | AMP、accumulation、`no_sync` | 06-06..08 | overflow/skip/micro-step 同步契约 | 已完成 |
| 06-10 | checkpoint/resume 与 artifact | 06-02..09 | 恢复下一步一致、配置/状态完整 | 本地已完成 |
| 06-11 | DDP 消融与 profiler 报告 | 06-03..10 | bulk/per-param/bucket/overlap/DDP 正式对照 | 已完成 |
| 06-12 | 模块发布验收 | 06-01..11 | CPU/GPU gates、报告、acceptance JSON | 已完成 |

最终实验同时给出一个重要负结果：1 MiB bucket 的真实 overlap 中位数为 2.676%，但吞吐
下降 7.31%。它证明 overlap 发生，不证明优化收益；launch 次数必须与隐藏时间一起分析。

## 不阻塞完成的扩展

- 通用 callback/plugin 生态、弹性训练和动态 membership；
- 完整复刻 DDP reducer 的所有稀疏/unused parameter 能力；
- 自动 bucket 搜索或生产级通信调度器。
