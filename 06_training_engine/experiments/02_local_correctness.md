# CPU/Gloo Correctness

固定 global batch 为 8，由两个 rank 等分。单进程 reference 对完整 batch 做一次 update；分布式
路径分别运行 bulk、per-parameter、bucket sync、bucket async 和 DDP，并测试 accumulation 1/2。
模型包含一个 unused parameter，用于验证 None gradient 不被错误更新。

2026-08-29 启用 unused parameter 后的本地结果为 10/10 成功。最大 gradient error
`2.09e-7`，最大 parameter update error `1.19e-7`。bulk、per-parameter、1 KiB bucket 的
collective 次数分别为 1、18、10；累积从 1
改为 2 后 collective 次数不变，说明非同步 micro-step 没有额外通信。

bucket async 的 launch 时间早于 backward-complete 事件，但这只是调度证据，不是 CUDA overlap
证据。
