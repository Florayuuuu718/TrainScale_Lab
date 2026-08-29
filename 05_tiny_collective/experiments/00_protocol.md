# TinyCollective 协议与不变量

## 数据与 rank 契约

- 单机 `torch.distributed` process group，每个进程一个 rank；
- 第一版只实现 SUM，输入为连续 tensor，返回 shape 与 dtype 不变；
- `chunk_sizes(N, P)` 将余数分配给低编号 chunk，因此任意 `N >= 0` 都有确定划分；
- 两个 ring phase 使用不同 tag 区间，同一步的 send tag 按发送 chunk 生成，recv tag 按接收
  chunk 生成。相邻 rank 的两者必须严格匹配。

## P2P 生命周期

每轮把 `isend` 和 `irecv` 一起提交，再等待全部 handle。接收 buffer 只能在 `wait()` 后读取，
同一个 exchange 不允许重复 wait。超时由 process-group 与外层 launcher 双重限制；失败必须留下
命令和 stderr，不能静默跳过。

## correctness oracle

相同输入先用 `torch.distributed.all_reduce(SUM)` 得到 reference，再执行教学算法并比较最大误差。
trace 同时验证轮数、事件数和跨 rank tag。浮点归约次序可能不同，因此使用固定 `atol/rtol`，
但不得为通过失败 case 临时放宽容差。
