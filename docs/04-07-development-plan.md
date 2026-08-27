# 04–07 开发总纲

这份文档是 04–07 的开发驾驶舱。它只回答阶段边界、依赖、共同完成定义和当前顺序；
具体命令与源码导航留在各模块 README，逐项状态留在对应验收清单。

## 不变的项目目标

TrainScale Lab 面向 ML Systems、AI Infrastructure 和 Distributed Training 初学者。
它不是链接合集，也不是对 PyTorch/NCCL 的简单配置封装。每个阶段必须完成：

```text
提出问题 → 预测 → 最小 reference → 正确性门 → 可信测量
        → 找到瓶颈 → 只改一个变量 → 复测 → 解释与归档
```

跑通框架 API、贴一张 profiler 截图或得到更快数字，都不能单独算完成。如果优化没有
变快，可以通过验收，但必须证明结果正确、测量可信并解释失败原因。

## 阶段关系

```text
03 DDP
├── 04 NCCL 性能 + DDP 通信桥接 ─┐
└── 05A CPU/Gloo collective ──────┼→ 06 reducer/overlap → 07 FSDP2/TP
             05B GPU 对照 ←──────04┘
```

- 04 用通信曲线和 GPU timeline 解释 03 的 scaling 现象；
- 05A 完成 03 后即可在 CPU 开发，05B 使用 04 的 GPU benchmark 契约；
- 06 复用 01/03，不重新实现通用训练框架；
- 07 复用 06 的 Tiny Transformer，避免更换 workload 后无法归因。

## 每阶段共同完成定义

1. README 写清问题、最小命令、源码地图、预期现象和边界；
2. CPU CI 覆盖配置、公式、schema 和可离线验证的 correctness；
3. GPU/multi-GPU 测试独立 opt-in，不伪装成普通 CPU CI；
4. smoke 与 formal 配置分离，正式测量至少三次；
5. 结果记录 commit、dirty state、环境、拓扑、配置哈希和原始证据哈希；
6. 报告包含运行前预测、一个变量的消融、Profiler/通信证据和失败记录；
7. acceptance 汇总所有可执行 gate，并把缺失硬件标为 `unavailable`；
8. 下一阶段只有在当前必需项关闭后才能成为唯一 active milestone。

## 推荐实施顺序

| 顺序 | 工作 | 目的 |
|---:|---|---|
| 1 | 04 环境门、公共 schema、解析器和 CPU 测试 | 先冻结实验与证据口径 |
| 2 | 04 多 GPU 正式实验与 DDP bridge | 解释 03，得到 06 bucket 输入 |
| 3 | 05A CPU/Gloo correctness | 不消耗云 GPU 完成算法调试 |
| 4 | 05B GPU/NCCL 对照 | 在冻结实现上集中租卡测量 |
| 5 | 06 Tiny Transformer 与 reducer 演进 | 建立真正新增的训练系统能力 |
| 6 | 06 多 GPU overlap 正式实验 | 用 timeline 验证优化机制 |
| 7 | 07A FSDP2 | 先解决状态显存 |
| 8 | 07B TP | 再解决单层切分与扩展 |
| 9 | 07C 2D 并行 | 仅在硬件和前序结果需要时进行 |

## 防止跑偏的范围闸门

- 不因参考项目有某项功能就自动加入范围；
- 不同时开发 04–07 多个正式阶段，05A 的 CPU 算法练习除外；
- 不为“代码更像框架”而增加无实验用途的抽象；
- 不把 8 GPU、多节点、PP、弹性容错和自动并行作为默认 v1.0 门槛；
- 不重写 01–03 已封存结果来追求目录统一；
- 每个新增 feature 必须能指出它要验证的瓶颈、指标和 correctness gate。

## 开发入口

- [04 README](../04_nccl_benchmark/README.md) · [04 验收清单](04-issues.md)
- [05 README](../05_tiny_collective/README.md) · [05 验收清单](05-issues.md)
- [06 README](../06_training_engine/README.md) · [06 验收清单](06-issues.md)
- [07 README](../07_parallelism/README.md) · [07 验收清单](07-issues.md)
- [公共 benchmark 契约](../benchmarks/README.md)
