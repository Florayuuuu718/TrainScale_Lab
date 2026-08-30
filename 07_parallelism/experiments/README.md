# 07 实验记录

推荐先读 [最终实验报告](01_final_report.md)，再用本页核对实验口径与判定标准。

## 实验问题

本阶段只比较两类不同瓶颈：FSDP2 是否通过切分训练状态缓解显存压力，以及 TP 是否通过切分单层 hidden/head 维度解决单层容量或计算压力。DDP 是对照，不把三者混成“谁总是更快”的排行榜。

## 本地发现

自定义 TP 的 MLP 与 attention 在 2/4 rank CPU/Gloo 下均通过。最大绝对误差约 `1.2e-7`，局部 weight shape 随 world size 缩小，输出保持 replicated。MLP 的 Colwise→Rowwise 组合每个 forward 只在 Rowwise 输出处执行一次 AllReduce。

真实 FSDP2 探针复用了 06 的 small Tiny Transformer。参数 placement 为 `Shard(0)`；SGD momentum 一步更新相对全局 reference 的最大误差约 `1.5e-8`；保存 DCP 后，连续执行下一步与恢复后执行下一步的误差为 0。该结果验证 API 语义与恢复路径，不是显存或吞吐结果。

PyTorch 2.8 的 CPU/Gloo 实测暴露出默认 reduce-scatter 梯度缩放与 reference 不一致，因此探针和正式 FSDP2 benchmark 都显式把 gradient divide factor 设置为 data-parallel world size。该设置固定的是“各 rank loss 均为 local mean 时取全局平均梯度”的实验契约，不能用放宽误差替代。

原生 `parallelize_module` MLP 的 `fc1` 为 output-dimension shard，`fc2` 为 input-dimension shard，最终输出 replicated；一步更新最大误差约 `8.9e-8`。早期错误地将 loss 除以 TP world size 会导致更新偏差，这一失败证据说明 TP 与 DDP 的 loss scaling 不能机械套用。

## GPU 实验读法

- DDP/FSDP 使用相同 per-rank batch，因此 global batch 随 world size 增长；比较吞吐扩展时同时报告 batch 语义。
- TP 的输入 batch 在 ranks 间复制，TP reference 固定为 world size 1；比较的是单个 MLP core 的切分代价，不等同于完整 Transformer 训练吞吐。
- 每个正式条件为 20 warmup、100 measured steps、3 个独立 torchrun 作业。计时取每步最慢 rank，再汇总 p50/p95；跨作业取中位数和相对极差。
- 峰值显存取所有 ranks 的最大值；DTensor 参数字节按 `to_local()` 统计。
- profiler 使用独立的 5-step 作业，只解释通信事件，不参与性能数字。

## 判定标准

FSDP2 只有在 correctness 通过、峰值显存明显下降且吞吐代价被量化后，才能说“解决了状态显存瓶颈”。TP 只有在 reference 对齐、placement/shape 正确、AllReduce trace 可见且容量或吞吐目标明确时，才能说“切分有价值”。如果模型本来就很小，分片更慢是合理结果，不应为了展示优化而隐藏。
