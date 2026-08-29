# Centralized 与 Ring 推导

设 rank 数为 `P`、tensor 元素数为 `N`。

## Centralized reduce + broadcast

每个非根 rank 向根发送 `N`，再从根接收 `N`。根节点串行处理两组 `P-1` 次传输：

- 非根通信量：`2N`；
- 根节点收发总量：`2(P-1)N`；
- 根节点是带宽和调度瓶颈。

它的价值是容易读懂和验证，不是扩展性。

## Ring all-reduce

reduce-scatter 与 all-gather 各执行 `P-1` 轮。若能整除，每个 rank 每轮发送 `N/P`：

- 每 rank 轮数：`2(P-1)`；
- 每 rank 通信量：`2(P-1)N/P`；
- 所有 rank 总发送量：`2(P-1)N`。

不能整除时，各 rank 发送量会因 chunk 大小相差少量元素；实现直接传递 uneven chunk，不使用
补零。`ring_volume()` 从真实 schedule 求最小/最大值并核对总量闭式公式。

复杂度正确不等于实现快。Python 循环、每轮 tensor 操作、通用 P2P 调用与同步都可能让该实现
远慢于融合并针对拓扑优化的 NCCL。
