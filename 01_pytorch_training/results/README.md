# M1 结果目录

这里保存适合 Git 的小型结果：`m1_summary.json` 是总表；三个 `*_curve.svg` 是学习曲线；两个 `*_ablation.json` 是 FP32/AMP/compile 对照；`dataloader_workers.json` 是 workers 测量；`profiler_summary.json` 是可信 CPU activity 摘要；其他 `*_profiler_summary.json` 保留 CUDA profiler 失败尝试产生的 CPU-only 事件证据。

`raw/` 保存逐 epoch JSONL、checkpoint 和 Chrome trace，由 `.gitignore` 排除，因为可重建且可能很大。人类可读解释位于[实验目录](../experiments/README.md)。数字必须连同配置、环境、控制变量和适用边界阅读。
