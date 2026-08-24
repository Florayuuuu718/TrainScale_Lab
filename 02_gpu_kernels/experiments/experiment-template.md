# 实验标题

> 状态：计划 / 进行中 / 已完成 / 阻塞

## 单一问题

本实验只回答什么问题？

## 数学定义与输入域

写出输入、输出、维度、dtype、layout、数值范围和明确不支持的情况。

## 实现对照

列出 PyTorch eager/compile/库实现、Triton baseline/optimized、CUDA C++（适用时）。说明比较是否语义等价。

## Correctness gate

列出 shape/dtype/layout、reference、容差、gradient 和错误边界。先定义通过条件，再运行性能测试。

## 对象特点与机制预测

估算必须读写的字节数、FLOPs、算术强度、kernel launch 数和中间张量；据此预测瓶颈。

## 运行前假设

明确哪些 case 可能获益、持平或变慢，以及原因。

## 环境与配置

记录 GPU、driver、Python、PyTorch、CUDA、Triton、Toolkit、commit、配置和 profiler 可用性。

## 复现命令

~~~bash
# 完整命令
~~~

## 原始结果

写明 `results/raw/` 位置、结果 schema 和失败记录。

## 汇总结果

报告 median 和离散程度；适用时报告 GB/s、TFLOPS、peak memory 和 compile latency。

## Profiler 证据

记录 kernel 名称、launch、memory/compute 指标和工具限制。没有 profiler 证据时明确标注。

## 完整推理链

输入特点 → 理论机制 → 实测 → profiler → 是否支持预测。

## 有限结论与一般预期

区分本机直接证据、通常预期和仍需验证的推断。

## 限制、失败点与下一步

保留 unsupported、OOM、没有加速的 shape 和可能的替代解释。
