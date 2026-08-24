# 实验 08：Profiler 与 Roofline——从数字走向解释

> 状态：PyTorch/Triton 的 Vector Add、融合 ReLU、MatMul、Attention 代表性 Profiler 已完成；Nsight Compute 指标属于后续增强。

## 为什么做

只看到“快/慢”不能说明原因。Profiler 告诉我们 GPU 实际启动了什么 kernel、设备时间多少；Roofline 用算术强度判断一个算子更可能受带宽还是算力限制。

## 小白名词

- **Profiler**：记录 CPU 调度、GPU kernel、时间和内存活动的工具。
- **device time**：GPU 设备执行活动的时间。
- **算术强度**：FLOPs ÷ 搬运字节数。
- **Roofline**：性能上限取“峰值算力”和“带宽×算术强度”中较小者。
- **compute-bound**：算术单元先成为瓶颈。
- **memory-bound**：内存系统先成为瓶颈。
- **aggregate row**：Profiler 把同名事件汇总后的行；嵌套行会重叠，不能全部相加当 wall time。

## 一般预期

Vector Add 每个元素只有一次加法却至少搬 12 字节，算术强度很低，通常 memory-bound。MatMul 的 tile 可以重复利用 A/B 数据，M/N/K 足够大时算术强度高，通常更接近 compute-bound。

## 跟着做：生成 Profiler 摘要和 trace

先完成实验 00 的环境探针和 correctness test。Profiler 入口
[`profile_triton_comparison.py`](../benchmarks/profile_triton_comparison.py)固定比较
Vector Add、融合 ReLU、MatMul 和 Attention 的 PyTorch/Triton 版本。

```bash
mkdir -p 02_gpu_kernels/results/raw/tutorial/08_traces

.venv/bin/python 02_gpu_kernels/benchmarks/profile_triton_comparison.py \
  --iterations 5 \
  --trace-directory 02_gpu_kernels/results/raw/tutorial/08_traces \
  --output 02_gpu_kernels/results/raw/tutorial/08_profiler.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/08_profiler.json
```

终端末尾应显示结果路径和 `cases=8`。结果阅读器打印每个 case 的 device-time
最大聚合行；Chrome/Perfetto 可打开 trace 目录下的 JSON 查看时间线。聚合行可能
嵌套，不能把所有 `device_time_total_us` 直接相加当 wall time。流程通过后用
`--iterations 20` 复现正式报告；Profiler 本身有开销，所以不要拿 profile 时间
替代实验 01–07 的 CUDA Event latency。

## 实际结果

| case | PyTorch 主要 device kernel（20 次） | Triton device kernel（20 次） | 读法 |
|---|---:|---:|---|
| Vector Add 1M FP32 | 117.097 µs | 219.267 µs | 朴素 Triton 单 kernel 本身更慢 |
| Add + ReLU 1M FP32 | add 115.018 µs + ReLU 99.985 µs | fused 219.892 µs | 融合减少 launch/分配，逻辑延迟仍更好 |
| MatMul 512³ FP16 | CUTLASS 228.394 µs | 459.432 µs | 成熟 tensorop kernel 约快 2 倍 |
| Attention H8/S128/D64 FP16 | flash 108.303 µs | 1690.004 µs | 算法都融合，工程优化程度差异巨大 |

表内都是相同 kernel 行的聚合 device time，除以 20 才是每次量级。大型 trace 在 ignored 的 `results/raw/`，可提交摘要见 [triton_profiler_sm120_cu129.json](../results/triton_profiler_sm120_cu129.json)。Profiler 的嵌套行不能全相加当作 GPU wall time。

## 理论分析

Vector Add 的算术强度约为 `1 FLOP / 12 bytes = 0.083 FLOP/byte`，即使 GPU 算力很强，也没有足够计算去隐藏数据移动。重复输入可能命中 cache，因此 effective bandwidth 高于外部显存带宽；这不改变它低算术强度的性质。

512³ MatMul 约有 2.68 亿 FLOPs，但 A/B/C 总数学数据量只有约 1.57 MB（忽略内部读写），tile 复用让算术强度远高于 Vector Add。Profiler kernel 名称中的 tensorop 支持它使用了矩阵乘加路径；Attention 的 flash kernel 名称也证实 PyTorch 走了专用 fused backend。

ReLU 对照展示了一个容易误读的现象：Triton fused kernel 的 device time 与两个 eager kernel 的量级相近，但逻辑 median 仍从 23.295 降到 19.752 µs。Profiler 与 benchmark 测的是不同边界，融合省下的调度、分配和依赖不能只看一行 kernel 时间。

## 结论与收尾

Profiler 已把“逻辑算子 latency”和“纯 GPU kernel time”分开，也解释了为什么 Vector Add、MatMul 和 Attention 需要不同优化方向。当前证据足以完成教程级跨算子分析；没有 Nsight Compute 的 DRAM、occupancy、stall 指标，所以 memory/compute-bound 仍是由算术强度、kernel 类型和相对表现共同支持的判断，不冒充完整硬件瓶颈证明。
