# 01 结果目录

这里保存适合 Git 的小型、可审阅结果；大型 trace、checkpoint 和逐 epoch 日志位于忽略的 `raw/`，因为它们能由命令重建且体积较大。

| 文件 | 回答的问题 |
|---|---|
| `summary.json` | 01 训练与性能实验的机器可读总表 |
| `synthetic_*` | 最小训练链是否学习、CPU/CUDA 是否数值对齐 |
| `cifar10_curve.svg` | CIFAR 子集学习曲线 |
| `dataloader_workers.json` | 短 synthetic 管线的 worker 固定成本 |
| `dataloader_image_workers.json` | 真实 JPEG、长 epoch 下的 worker 稳态趋势 |
| `cifar10_modes_wsl.json` | WSL2 完整 CIFAR-10 的 FP32/AMP/compile 对照 |
| `profiler_summary.json` | synthetic CPU activity |
| `cifar10_cuda_profiler_wsl_cu129.json` | PyTorch 2.12.1+cu129 的真实 CUDA device-time 验收 |

读取数字时必须同时查看对应实验的对象特点、控制变量和边界。尤其注意：Profiler 聚合行存在父子嵌套，device-time 行数不是 kernel 数，聚合值求和也不是 GPU wall time。

开发阶段使用旧 PyTorch/CUPTI 组合产生的失败数据不属于学习主线；若遇到类似问题，请直接查看[实验 06 的排障提示](../experiments/06_profiler.md)并确认使用锁定环境。人类可读的完整解释位于[实验目录](../experiments/README.md)。
