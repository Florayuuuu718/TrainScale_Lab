# 07 · FSDP2 / Tensor Parallel

> 状态：规划已冻结，尚未实现。本目录是模型切分与组合并行的唯一入口。

07 只在模型或训练状态确实形成单卡显存/扩展瓶颈后引入切分。它继续使用 06 的
Tiny Transformer、训练口径和 artifact 契约，关注“为什么选择这种并行策略”，
而不是收集一组能够启动的框架配置。

## 07A · FSDP2：解决状态显存

1. 分解参数、梯度、optimizer state 和 activation 显存；
2. 建立单卡 baseline 与可控 OOM case；
3. 用 `fully_shard` 实现最小 FSDP2；
4. 观察参数 AllGather 和梯度 ReduceScatter；
5. 比较不同 shard placement/wrap 粒度；
6. 使用 distributed checkpoint 保存和恢复分片状态；
7. 验证单步数值、恢复一致性、峰值显存和吞吐。

“单卡 OOM”必须由配置和显存估算构造，不能通过同时运行无关进程人为挤占显存。
如果硬件不足以形成可信 OOM 对照，应记录为 `unavailable`，并保留较小规模的语义验证。

## 07B · Tensor Parallel：解决单层计算/参数切分

1. 建立 DeviceMesh 与 mesh dimension 命名；
2. 对 attention/MLP 分别实现 Colwise/Rowwise Parallel；
3. 明确 head、hidden size 与 world size 的整除契约；
4. 对照未切分 reference 的输出、loss、梯度和一步更新；
5. 记录 DTensor placement、通信和局部 shape；
6. 比较显存、吞吐和扩展效率。

## 07C · 组合并行：可选正式门槛

- 4 GPU 环境优先尝试 `TP=2 × DP/FSDP=2` 的二维 DeviceMesh；
- 比较纯 DDP、纯 FSDP2、纯 TP 与 2D 组合；
- 说明每个 mesh 维度引入的通信；
- 硬件不足时明确记录 `unavailable`。

07C、Pipeline Parallel、8 GPU 和多节点都不是默认 v1.0 阻塞项。只有 07A/07B 的
结果暴露出明确需要时，才继续扩大范围。

## 正确性与验收证据

- 未切分 reference 与分片实现的输出、loss、梯度和一步更新对齐；
- FSDP2 checkpoint/resume 后下一步一致；
- 理论状态显存与实测峰值显存的差异得到解释；
- 至少两个模型规模，不能只用一个“刚好能跑”的 shape；
- 吞吐、step time、peak memory、collective/timeline 和 scaling efficiency；
- 最终给出 DDP/FSDP2/TP/2D 的并行策略选择树。

## 范围边界

不实现生产级自动并行搜索、完整 TorchTitan/Megatron 功能、多节点弹性恢复或任意
Transformer 架构。框架仅用于提供 primitive；核心交付仍是最小实现、测量、解释和
一次可复现优化。

进入本模块前，请先完成 [06](../06_training_engine/README.md)。逐项开发与验收见
[07 验收清单](../docs/07-issues.md)。
