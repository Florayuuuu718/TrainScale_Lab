# M1 实验索引

测试回答“代码的重要性质是否仍成立”，实验回答“在固定环境和控制变量下发生了什么”。推荐按顺序阅读：

1. [Synthetic CPU/CUDA FP32 baseline](01_synthetic_baseline.md)：先证明完整训练链路能学习。
2. [梯度累积与 checkpoint resume](02_accumulation_and_resume.md)：理解有效 batch 和完整训练状态。
3. [DataLoader workers 吞吐](03_dataloader_workers.md)：学习控制变量、重复测量和平台边界。
4. [FP32、AMP 与 compile 消融](04_fp32_amp_compile.md)：同时看正确性、吞吐和显存。
5. [CIFAR-10 CNN baseline](05_cifar10_baseline.md)：从合成数据走向真实图像。
6. [PyTorch Profiler](06_profiler.md)：从 operator 时间定位开销。
7. [失败：CUDA Profiler/CUPTI](07_failure_gpu_profiler.md)：区分训练 runtime 与 profiling 工具链。
8. [失败：Windows torch.compile/Triton](08_failure_compile_windows.md)：理解优化特性的环境边界。

每份报告均包含问题、假设/控制变量、完整命令、结构化结果、分析和知识总结。新实验请复制 [`experiment-template.md`](experiment-template.md)，不要只保存终端截图或挑选最好的一次数字。
