# 实验 00：先冻结通信 benchmark 协议

## 预测

小消息更容易受 launch/synchronization 延迟影响；大消息更可能接近链路带宽上限。
这是运行前假设，不是当前机器结论。

## 固定变量

- `nccl-tests v2.19.7` / commit `1a65d7f...`；
- 同一主机、GPU 顺序、driver、CUDA/NCCL 和环境；
- smoke 最大 1 MiB，formal 为 8 B–256 MiB；
- formal 每个 case warm-up 5 次、测量 20 次，整套运行三次；
- stdout 原文与结构化 JSON 同时保存。

编译参数也属于环境证据。本机 Ubuntu 26.04/GCC 15/CUDA 13.0 默认构建遇到
`rsqrt` 头文件冲突；`-U_GNU_SOURCE -D_DEFAULT_SOURCE` 越过它后又遇到 pthread
clock API 声明错误，因此本机实编译明确未通过。这些参数只用于复现诊断，不应复制到
推荐的租卡 Ubuntu 22.04/24.04 环境，也不把替换系统编译器变成 04 的前置项目。

## 状态语义

- `success`：进程成功、表格可解析、可用错误计数全为 0；
- `failed`：命令、超时、解析或 correctness 失败；
- `unavailable`：缺 Linux、binary 或足够 GPU，没有任何性能值。

正式发布只聚合三次 clean worktree、同 commit/config/environment 且 row 集合完全
一致的 `success` 结果。中位数不用于掩盖抖动；报告还要保留三次原值。
