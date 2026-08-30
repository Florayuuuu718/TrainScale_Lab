# 07 结果与证据

公开的紧凑结果见 [`module07_final_summary.json`](module07_final_summary.json)，完整解释见
[`../experiments/01_final_report.md`](../experiments/01_final_report.md)。`memory-estimate.json` 是
本地理论估算；`module07-acceptance.json` 已用正式 GPU artifacts 汇总为 `complete`，并保留
CPU/Gloo FSDP2 unavailable 与 CUDA/NCCL success 的区别。

`raw/` 保存 rank JSON、DCP、日志和 profiler trace，由 Git 忽略。CPU/Gloo unavailable 记录与
CUDA/NCCL success 记录必须同时保留，因为它们描述不同 backend capability，不能互相覆盖。
