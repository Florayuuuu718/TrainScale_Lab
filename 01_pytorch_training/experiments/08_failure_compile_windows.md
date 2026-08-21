# 失败实验：CIFAR-10 torch.compile 缺少 Triton

## 现象

同一 benchmark 中 FP32 eager 和 AMP eager 均完成；`torch.compile` 在 CIFAR CNN 首次执行时失败并写入：

```text
TritonMissing: Cannot find a working triton installation
```

结构化失败状态保存在 [`cifar10_ablation.json`](../results/cifar10_ablation.json)，前两个成功 variant 没有因此丢失。

## 为什么 synthetic 看似成功

极小 MLP 的图可能走了不依赖完整 Triton kernel 的路径，同时出现 `triton not found` 警告。CNN 卷积工作量触发 Inductor 的 GPU 代码生成后才暴露缺失。因此不能仅凭一个小模型完成就宣布 compile 工具链可用。

## 当前结论与处理

当前 PyTorch 2.11 cu128 Windows wheel 能进行 CUDA eager/AMP 训练，但本机没有可供该 compile 路径使用的 Triton。M1 不额外引入非官方 Windows Triton 包，也不把 compile 失败误归因于 nvcc；后续应在官方支持良好的 Linux/WSL2 环境建立 compile 基线。

可靠实验的做法是：记录环境和异常类型、保留成功对照、明确能力边界，再决定是否更换运行环境。
