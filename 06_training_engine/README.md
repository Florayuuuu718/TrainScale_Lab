# 06 · Mini Training Engine + Gradient Reducer Lab

> 状态：规划已冻结，尚未实现。本目录是迷你分布式训练引擎和 reducer 实验的唯一入口。

06 不重新发明 01 的单卡训练循环，也不重复 03 的 DDP 教程。它复用已经证明的
训练、分布式和证据契约，集中实现 PyTorch DDP 隐藏起来的梯度同步调度：何时发起
collective、怎样分 bucket，以及通信怎样与 backward 重叠。

## 本阶段要回答的问题

- bulk、per-parameter 和 bucketed gradient synchronization 有什么代价？
- autograd hook 何时触发，怎样保证每个梯度只同步一次？
- 异步 collective 何时真正与 backward 重叠？
- AMP、gradient accumulation 和 `no_sync` 怎样改变同步边界？
- 自己的 reducer 与 PyTorch DDP 相比缺少哪些能力？

## 复用边界

- 从 01 复用配置、seed、AMP、累积、指标和 checkpoint 语义；
- 从 03 复用 rank/process group、sampler、launcher、最慢 rank 计时和 scaling 口径；
- 从 04 选择有意义的 bucket size 和消息区间；
- 从 05 复用通信量推导；TinyCollective 只作为可选 reference backend。

01–03 已封存的代码和结果不为统一目录结构而重写。06 开始建立供 06/07 使用的
共享包或稳定适配层，避免继续复制训练循环、launcher 和 artifact 代码。

## 开发顺序

1. 建立可扩缩的 Tiny Transformer 和单卡数值 baseline；
2. 实现 backward 完成后一次 bulk AllReduce reference；
3. 实现 per-parameter hook 同步；
4. 实现 deterministic bucket assignment 与 bucket view；
5. 实现 bucket AllReduce；
6. 实现 async bucket 和 handle 生命周期管理；
7. 用 timeline 证明 backward/communication overlap；
8. 加入 AMP、accumulation、`no_sync` 和 unused/None gradient 边界；
9. 与 PyTorch DDP 做正确性、吞吐、显存和 timeline 对照；
10. 完成 checkpoint/resume、消融汇总和 acceptance。

## 必须测试的不变量

- 同一 global batch 下，单卡 reference、手写 reducer 和 DDP 的梯度/更新在容差内一致；
- 每个参数属于且只属于一个 bucket，bucket offset 不重叠；
- optimizer step 前所有异步 handle 已完成；
- accumulation 的非同步 micro-step 不发起 gradient collective；
- AMP unscale、overflow/skip 与 collective 顺序明确；
- 任一 rank 的 bucket 顺序不一致时快速失败，而不是静默挂死；
- checkpoint 恢复后的下一步与连续训练一致。

## 验收实验

- bulk、per-parameter、bucket、bucket+overlap、PyTorch DDP；
- 至少两种模型规模、三种 bucket size；
- FP32 与 AMP，accumulation off/on；
- samples/s、step time p50/p95、peak memory、collective 次数和通信占比；
- timeline 中可见的 overlap 区间；
- 每次只改变一个 reducer/precision/accumulation 变量。

## 范围边界

本模块不追求完整 callback 生态、弹性容错、任意模型自动并行或生产级 reducer。
优化没有带来加速也可以通过验收，但必须保留正确性、可信测量和瓶颈解释。

进入本模块前，应完成 04 的通信测量和 05 的算法正确性主线。逐项开发与验收见
[06 验收清单](../docs/06-issues.md)。
