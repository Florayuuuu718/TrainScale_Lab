# 实验 04：FP32、AMP 与 torch.compile 消融

## 问题与控制变量

固定数据、seed、模型、batch、optimizer、scheduler 和 epochs，只改变 precision 或 compile。synthetic 用来验证数值链路，CIFAR-10 CNN 用来形成更有意义的 GPU 工作量。

```powershell
.venv\Scripts\python -m trainscale_training.benchmark --config 01_pytorch_training/configs/synthetic_cuda.toml --output 01_pytorch_training/results/synthetic_ablation.json
.venv\Scripts\python -m trainscale_training.benchmark --config 01_pytorch_training/configs/cifar10_ablation.toml --output 01_pytorch_training/results/cifar10_ablation.json
```

## 实测结果

| workload / variant | 状态 | steady samples/s | peak CUDA bytes | final valid acc |
|---|---|---:|---:|---:|
| synthetic FP32 eager | 完成 | 37,621 | 18,127,872 | 0.8929 |
| synthetic AMP eager | 完成 | 20,917 | 18,118,656 | 0.8929 |
| synthetic FP32 compile | 完成 | 40,475 | 18,127,872 | 0.8929 |
| CIFAR FP32 eager | 完成 | 6,839 | 142,372,352 | 0.3320 |
| CIFAR AMP eager | 完成 | 7,370 | 70,057,984 | 0.3506 |
| CIFAR FP32 compile | 失败 | — | — | — |

CIFAR AMP 相对 FP32 的稳态吞吐约提高 7.8%，峰值 allocated memory 约减少 50.8%。短短 3 epoch 中的 accuracy 差异不能解释为 AMP 提高泛化能力；它只表明数值链路没有明显异常。

synthetic 太小，autocast/GradScaler 等固定开销会抵消收益，短时吞吐也波动明显，所以不能用它给三种方案做普适排名。compile 在小 MLP 上完成，但运行时警告找不到 Triton；在 CNN 真正触发 Inductor GPU 代码生成时失败。因此“命令没报错”也不等于完整优化路径可用。

结构化结果：[synthetic](../results/synthetic_ablation.json)、[CIFAR-10](../results/cifar10_ablation.json)。compile 失败分析见[实验 08](08_failure_compile_windows.md)。

## 知识总结

- AMP 的收益与 Tensor Core 可利用程度和激活显存有关，必须在实际 workload 上测；
- compile 是 JIT，要区分首次编译与 steady-state；
- 性能开关后仍要验证 loss/accuracy；
- benchmark harness 应保存失败状态和部分成功结果，不能因一个 variant 失败而丢失全部实验。
