# 实验 02：梯度累积与 checkpoint resume

## 问题 A：梯度累积是否等价于有效大 batch

reference 使用 batch 64；累积配置使用 micro-batch 16、`accumulation_steps=4`，有效 batch 同为 64。每个 micro-batch 的 loss 除以 4，累积完再执行一次 optimizer step。

复现：

```powershell
.venv\Scripts\python -m trainscale_training.train `
  --config 01_pytorch_training/configs/synthetic_accumulation.toml
```

实测 5 epoch 的 loss 和 accuracy 与 reference batch 64 在显示精度下完全一致，global optimizer step 同为 35。自动测试进一步在单次更新层面对全部参数执行 `torch.testing.assert_close`。

```powershell
.venv\Scripts\pytest `
  01_pytorch_training/tests/test_training.py::test_accumulation_matches_effective_batch_update -v
```

结论：在相同样本顺序、等大小 micro-batch、正确缩放 loss 且只在有效 batch 末尾 step 的条件下，累积更新与大 batch reference 一致。包含 BatchNorm、随机数据增强或不等大小 micro-batch 时，需要重新分析等价边界。

## 问题 B：resume 是否延续同一训练轨迹

先运行到 epoch 3：

```powershell
.venv\Scripts\python -m trainscale_training.train `
  --config 01_pytorch_training/configs/synthetic_cpu.toml `
  --epochs 3 `
  --output-dir 01_pytorch_training/results/raw/resume_demo
```

再从 `last.pt` 恢复到 epoch 5：

```powershell
.venv\Scripts\python -m trainscale_training.train `
  --config 01_pytorch_training/configs/synthetic_cpu.toml `
  --epochs 5 `
  --resume 01_pytorch_training/results/raw/resume_demo/last.pt `
  --output-dir 01_pytorch_training/results/raw/resume_demo
```

| 路径 | epoch 4 train/valid loss | epoch 5 train/valid loss | epoch 5 valid accuracy |
|---|---|---|---:|
| 连续 1→5 | 0.443353 / 0.433336 | 0.261855 / 0.325891 | 0.892857 |
| 1→3，resume 4→5 | 0.443353 / 0.433336 | 0.261855 / 0.325891 | 0.892857 |

两条路径一致。自动测试还比较 resume 下一步的 loss 和全部更新后参数，而不是只检查文件能否打开。

## 为什么 checkpoint 要保存完整状态

仅保存 model weights 会丢失 momentum、scheduler 位置、AMP scale、epoch/step 和 RNG。模型看似恢复了，但下一步更新已经不是原训练轨迹。当前 schema 见 [`docs/checkpoint-contract.md`](../../docs/checkpoint-contract.md)。

## 知识总结

- 梯度累积改变 micro-batch 显存需求，不应意外改变有效 batch 的梯度尺度；
- `optimizer.step()` 和 `GradScaler.update()` 应在有效 batch 边界执行；
- checkpoint 正确性的强证据是“恢复后下一步与连续训练一致”，不是“文件存在”。
