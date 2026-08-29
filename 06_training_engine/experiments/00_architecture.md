# Engine 与 Reducer 边界

训练 step 只管理 forward/backward、accumulation、AMP、clip 和 optimizer 顺序；reducer 只管理
gradient readiness 与 collective。手写 reducer 必须在 optimizer 前 `finish_backward()`，否则
`assert_complete()` 失败。DDP 路径使用其原生 `no_sync()`，不伪装成手写 reducer。

模型采用无 dropout 的 encoder-only Tiny Transformer。small 用于快速 correctness，medium 用于
GPU 消融并由 07 直接复用，避免切换 workload 后失去可比性。
