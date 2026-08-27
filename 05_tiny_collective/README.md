# 05 · TinyCollective

> 状态：规划已冻结，尚未实现。本目录是教学版 collective 算法的唯一入口。

05 的目标不是替代 NCCL，而是用最小可读实现回答“AllReduce 内部发生了什么”。
开发时先在 CPU/Gloo 上证明调度和数学正确，再把 GPU/NCCL 作为性能对照。

## 本阶段要回答的问题

- centralized 与 ring 算法各需要多少轮、每个 rank 发送多少数据？
- reduce-scatter 和 all-gather 怎样组合成 ring all-reduce？
- 为什么所有 rank 的调用顺序稍有不同就可能死锁？
- 浮点加法顺序为什么改变误差，但不一定表示算法错误？
- 为什么教学版算法即使复杂度正确，仍可能远慢于 NCCL？

## 两条开发路线

### A · CPU/Gloo 正确性主线

完成 03 后即可开始，不必等待多 GPU：

1. centralized reduce + broadcast；
2. ring reduce-scatter；
3. ring all-gather；
4. 组合 ring all-reduce；
5. 每轮 sender、receiver、chunk owner 的确定性 debug trace；
6. 从阻塞 P2P 演进到明确管理生命周期的异步 P2P。

### B · GPU 性能对照

依赖 04 的测量契约和真实多 GPU 环境：

- 与 `torch.distributed.all_reduce` 对照；
- 使用相同 dtype、消息大小、world size、warm-up 和重复次数；
- 解释 Python 调度、同步、额外复制和拓扑带来的差距。

## 正确性矩阵

- world size 2/3/4；
- FP32，以及环境支持时的 FP16/BF16；
- 空间较小、非 2 次幂和不能被 world size 整除的元素数量；
- in-place/out-of-place 语义；
- SUM reference、容差和不同归约顺序；
- tag 冲突、调用次序、异步 handle 生命周期和超时诊断。

第一版允许对非整除 chunk 显式拒绝，但不能以此完成阶段验收；正式版本应使用
uneven chunk 或 padding/unpadding，并通过 world size 3 的测试。

## 最终证据

- centralized/ring 的轮数和每 rank 通信量推导；
- debug trace 与推导一致；
- correctness matrix 全部可执行项通过；
- TinyCollective 与 PyTorch/NCCL 的消息大小–延迟曲线；
- 至少一个“理论复杂度正确但实现仍慢”的 profiler 解释。

## 范围边界

TinyCollective 是 reference implementation，不承诺生产性能、故障恢复或任意网络
拓扑优化。06 可以把它作为可选教学 backend，但不能把它默认包装成高性能训练通信层。

CPU 正确性路线依赖 [03](../03_distributed_training/README.md)；GPU 性能路线依赖
[04](../04_nccl_benchmark/README.md)。逐项开发与验收见
[05 验收清单](../docs/05-issues.md)。
