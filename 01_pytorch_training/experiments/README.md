# M1 实验索引

测试回答“代码的重要性质是否仍成立”，实验回答“为什么在固定环境和控制变量下出现这个结果”。每份报告都按“概念 → 对象特点 → 机制预测 → 实测 → 有限结论 → 一般结论”组织。

推荐学习顺序：

1. [Synthetic CPU/CUDA FP32 baseline](01_synthetic_baseline.md)：证明完整训练链路能学习，并理解小模型为何可能不适合 GPU。
2. [梯度累积与 checkpoint resume](02_accumulation_and_resume.md)：理解有效 batch 和必须保存的训练状态。
3. [DataLoader workers 吞吐](03_dataloader_workers.md)：学习控制变量、重复测量和平台边界。
   - [可选扩展：真实 JPEG 解码与长 epoch](03_cont_dataloader_workers.md)：验证复杂数据管线下 workers 的一般趋势，并区分首 epoch 与稳态。
4. [FP32、AMP 与 compile](04_fp32_amp_compile.md)：同时解释固定成本、稳态吞吐、显存和正确性。
   - [可选扩展：长 workload AMP/compile](04_cont_amp_compile_wsl.md)：计算 break-even，并分析组合优化为何不能简单相加。
5. [CIFAR-10 CNN baseline](05_cifar10_baseline.md)：从合成规则走向真实图像分类。
6. [PyTorch Profiler](06_profiler.md)：从 CPU 调度走到真实 CUDA kernel activity。
   - [可选扩展：真实 CNN CUDA Profiler 验收](06_cont_cuda_profiler_wsl.md)：学习 device-time 字段与聚合边界。

原生 Windows 的 Triton/compile 或旧 PyTorch/CUPTI 兼容问题不是学习步骤；对应排障提示已经放回实验 04 和 06。Windows + NVIDIA GPU 学习者应先按 [WSL2 Ubuntu 教程](../../docs/getting-started/wsl2-gpu.md)建立正式环境。

新实验请复制 [`experiment-template.md`](experiment-template.md)。不要只保存截图或最好的一次数字；必须保存配置、环境、重复测量、正确性门和结论边界。
