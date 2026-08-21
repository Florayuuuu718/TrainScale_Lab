# 实验 03：DataLoader workers 吞吐

## 问题

Windows 上给每个 synthetic 样本增加 1 ms 模拟预处理延迟后，增加 `num_workers` 是否提高完整遍历吞吐？

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

## 知识总结

- DataLoader 性能实验输出 samples/s，不输出模型 accuracy；
- 改变 workers 时必须固定数据、batch、delay 和重复次数；
- 报告 median 和波动范围比挑选一次最快值更可信；
- 性能参数没有脱离平台和 workload 的“最佳默认值”。
