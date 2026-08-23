# 实验 03：DataLoader workers 吞吐

## 问题

Windows 上给每个 synthetic 样本增加 1 ms 模拟预处理延迟后，增加 `num_workers` 是否提高完整遍历吞吐？

## 概念与机制

`num_workers` 决定 DataLoader 使用多少子进程并行执行 `Dataset.__getitem__`。worker 可以让读取、解码和增强与训练重叠，但也引入进程启动、序列化、IPC、预取和额外内存开销。Windows 默认使用 spawn，短 epoch 中这些固定成本尤其明显。

## 配置与复现

- 512 samples、128 features、batch 32；
- workers 0/1/2/4；
- 每种配置重复 3 次；
- 只迭代 DataLoader，不执行模型 forward/backward。

```powershell
.venv\Scripts\python -m trainscale_training.benchmark_workers `
  --workers 0 1 2 4 --samples 512 --input-dim 128 `
  --batch-size 32 --delay-ms 1 --repeats 3 `
  --output 01_pytorch_training/results/dataloader_workers.json
```

## 实测结果

| workers | median samples/s | min | max | 相对 workers=0 |
|---:|---:|---:|---:|---:|
| 0 | 584.77 | 572.80 | 587.80 | 1.00x |
| 1 | 171.62 | 163.17 | 172.22 | 0.29x |
| 2 | 199.72 | 195.66 | 201.27 | 0.34x |
| 4 | 163.48 | 161.09 | 175.68 | 0.28x |

原始小型 JSON：[`dataloader_workers.json`](../results/dataloader_workers.json)。

## 分析

workers=0 最快。Windows 使用 spawn 创建进程，短数据集的进程启动、序列化和 IPC 成本超过并行预处理收益。2 workers 比 1 稍好，但 4 workers 又下降，说明 worker 不是越多越快。

本结论不能推广到真实图片解码、长 epoch、Linux 或 GPU 被数据饿住的训练。每个新数据管线都应重新扫描 workers。

为验证其中两个关键边界，新增了[实验 03（补充）：真实 JPEG 解码与长 epoch 下的 workers](03_cont_dataloader_workers.md)。补充实验把首 epoch 与预热后的稳态吞吐分开，避免把 worker 启动成本误当成长训练的持续成本。

## 知识总结

- DataLoader 性能实验输出 samples/s，不输出模型 accuracy；
- 改变 workers 时必须固定数据、batch、delay 和重复次数；
- 报告 median 和波动范围比挑选一次最快值更可信；
- 性能参数没有脱离平台和 workload 的“最佳默认值”。

## 有限结论与一般预期

本实验只支持“短小 Windows synthetic 管线中 0 worker 最快”。一般复杂图像管线通常包含可并行 I/O/解码工作，长训练还能摊薄启动成本；这一预期已由[真实 JPEG 长 epoch 补充实验](03_cont_dataloader_workers.md)在当前机器上得到进一步支持，但仍不能替代目标平台扫描。
