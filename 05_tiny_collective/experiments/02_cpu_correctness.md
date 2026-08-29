# CPU/Gloo correctness 实验

## 目的

在不租 GPU 的情况下隔离验证算法数学、ragged chunk、tag 和异步 handle 生命周期。

## 固定矩阵

- world size：2、3、4；
- element count：5、7、16、17；
- algorithm：centralized、ring；
- dtype：FP32；reference：PyTorch Gloo AllReduce SUM。

共 24 个 case。每个 case 必须满足所有 rank 数值通过，且 trace 事件数符合协议。centralized
根节点为 `2(P-1)` 个事件，非根为 2；ring 每个 rank 为 `2(P-1)`。

## 当前本地结果

2026-08-28 在 WSL/Linux + PyTorch 环境运行，24/24 case 成功。结果写入被 Git 忽略的
`results/raw/cpu_correctness.json`。正式发布 artifact 应在干净 commit 上重新生成，避免把 dirty
工作树结果当作最终可复现证据。
