# 失败实验：CUDA 训练成功，但 GPU Profiler activity 缺失

## 现象

CUDA baseline 可以正常训练，且：

```text
torch=2.11.0+cu128
torch.version.cuda=12.8
torch.cuda.is_available()=True
GPU=NVIDIA GeForce RTX 5060 Laptop GPU
```

但运行 PyTorch Profiler CUDA activity 时出现：

```text
CUPTI_ERROR_INVALID_DEVICE (2)
CUPTI initialization failed - CUDA profiler activities will be missing
```

程序仍导出 trace，但其中只有 CPU activity，不能用于 CUDA kernel 时间分析。

## 已排除

- 不是 CPU-only PyTorch wheel：版本包含 `+cu128`；
- 不是 CUDA 完全不可用：GPU smoke training 已成功；
- 不是模型无法前向/反向：synthetic CUDA 5 epochs 完成；
- CPU Profiler 路径可产生 event summary。

## 当前判断

训练 runtime 与 CUPTI profiling 是不同能力。前者只要求驱动能执行 CUDA 程序，后者还依赖 CUPTI、驱动/工具兼容和分析权限。当前 Windows、driver 577.05、PyTorch 2.11 cu128 环境下 CUPTI 初始化失败。

## 处理决定

- 保留 CPU Profiler trace 和解释；
- 不把缺失 CUDA activity 的 trace 表述为 GPU profile；
- 不为了本实验盲目降级驱动；
- 后续先尝试更新到满足目标 PyTorch/CUDA 矩阵的驱动，再在 Linux/WSL2 或原生 Linux 环境复测；
- 进入 M2 前仍需建立可用的 CUDA profiler 工具链。

## 学到什么

“GPU 代码能跑”不等于“所有 CUDA 开发工具都可用”。可靠实验必须区分 runtime、compiler (`nvcc`) 和 profiler (CUPTI/Nsight) 三条工具链，并把失败作为结果记录，而不是只保留成功截图。
