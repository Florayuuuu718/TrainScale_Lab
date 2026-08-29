# Reducer 演进与预测

## Bulk

backward 全部完成后 flatten 所有非 None gradient，只发起一次 AllReduce。启动开销低，但没有
通信计算重叠。

## Per-parameter

每个 autograd hook 立即同步对应 gradient。ready 时间最早，但 collective 数量等于参与训练的
参数 tensor 数，通常被启动开销主导。

## Bucket sync / async

参数按反向注册顺序确定性装桶，每个参数唯一归属，offset 不重叠。sync bucket 在 hook 中等待；
async bucket 保存 handle，optimizer 前统一 wait。预测存在中间 bucket cap：比小 bucket 少启动，
又比大 bucket 更早 ready。预测是否成立由正式消融决定。

每个 rank 在训练前 all-gather bucket plan digest。静态模型不一致会快速失败；动态 unused 集合
跨 rank 不一致不属于第一版支持范围。
