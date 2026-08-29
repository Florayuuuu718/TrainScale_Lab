# 05 · TinyCollective 验收清单

05 先证明数学与调度正确，再比较性能。教学实现慢于 NCCL 是正常结果，不得通过
删除不利 case 或降低 correctness 容差来美化曲线。

| ID | 验收项 | 依赖 | 完成证据 | 状态 |
|---|---|---|---|---|
| 05-01 | 冻结算法、P2P 与超时契约 | 03 完成 | README、消息/tag/chunk 规则 | 已完成 |
| 05-02 | centralized reduce + broadcast | 05-01 | 2/3/4 rank CPU correctness 与 trace | 已完成 |
| 05-03 | ring reduce-scatter | 05-01 | 每轮 owner/邻居与 reference 对齐 | 已完成 |
| 05-04 | ring all-gather | 05-03 | chunk 恢复完整且 trace 可验证 | 已完成 |
| 05-05 | ring all-reduce 组合 | 05-03..04 | 与 `dist.all_reduce` 数值对照 | 已完成 |
| 05-06 | ragged/非整除 chunk | 05-05 | world=3、padding 或 uneven chunk tests | 已完成 |
| 05-07 | 异步 P2P 与生命周期 | 05-05 | handle、tag、顺序和 timeout tests | 已完成 |
| 05-08 | 通信量与轮数推导 | 05-02..07 | 公式、CPU tests、trace 自动核对 | 已完成 |
| 05-09 | GPU/NCCL 性能对照 | 04 完成，05-05 | 同消息/dtype/world size 正式曲线 | 待统一租卡 |
| 05-10 | 模块发布验收 | 05-01..09 | correctness matrix、报告、acceptance JSON | 本地门完成，待 05-09 |

## 不阻塞完成的扩展

- 自适应选择 tree/ring、分层拓扑或任意 ReduceOp；
- 生产级容错、拥塞控制和多节点优化；
- 把 TinyCollective 宣称为 NCCL 替代品。
