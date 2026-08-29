# 06 · Mini Training Engine + Reducer Lab 验收清单

06 的新增价值是 reducer、bucketing 和 overlap。若某项工作只是重复 01/03 且没有
新的实验问题，应复用或建立适配层，而不是重新实现。

| ID | 验收项 | 依赖 | 完成证据 | 状态 |
|---|---|---|---|---|
| 06-01 | 冻结 engine/reducer 范围与共享边界 | 04/05 | 架构说明、明确复用和非目标 | 已完成 |
| 06-02 | Tiny Transformer 与单卡 baseline | 06-01 | 两种规模、overfit/一步更新/显存基线 | CPU overfit/更新完成，GPU 显存待租卡 |
| 06-03 | bulk AllReduce reference | 06-02 | 对 global-batch reference 的梯度与更新 | 已完成 |
| 06-04 | per-parameter hook reducer | 06-03 | 次数、顺序、None gradient correctness | 已完成 |
| 06-05 | deterministic gradient buckets | 06-04 | ownership/offset/view 单元测试 | 已完成 |
| 06-06 | bucketed synchronous reducer | 06-05 | 多 bucket size、数值与通信次数 | 本地正确，GPU 消融待租卡 |
| 06-07 | async bucket 与 handle 管理 | 06-06 | optimizer 前完成、异常快速失败 | 已完成 |
| 06-08 | backward/communication overlap | 06-07 | GPU timeline 直接证明 overlap | runner 完成，待租卡 |
| 06-09 | AMP、accumulation、`no_sync` | 06-06..08 | overflow/skip/micro-step 同步契约 | accumulation 完成，AMP/overflow runner 完成，待租卡 |
| 06-10 | checkpoint/resume 与 artifact | 06-02..09 | 恢复下一步一致、配置/状态完整 | 本地已完成 |
| 06-11 | DDP 消融与 profiler 报告 | 06-03..10 | bulk/per-param/bucket/overlap/DDP 正式对照 | runner 完成，待租卡 |
| 06-12 | 模块发布验收 | 06-01..11 | CPU/GPU gates、报告、acceptance JSON | 本地门完成，待 GPU gates |

## 不阻塞完成的扩展

- 通用 callback/plugin 生态、弹性训练和动态 membership；
- 完整复刻 DDP reducer 的所有稀疏/unused parameter 能力；
- 自动 bucket 搜索或生产级通信调度器。
