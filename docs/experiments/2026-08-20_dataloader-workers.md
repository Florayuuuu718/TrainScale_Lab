# DataLoader workers 对吞吐的影响

## 问题与假设

只改变 `num_workers` 时，带有每样本 1 ms 模拟预处理延迟的数据集会从并行加载中获益；但在 Windows 的 `spawn` 启动方式下，worker 启动与进程间传输会使短实验出现明显固定开销，因此 worker 并非越多越快。

## 固定条件

- dataset：离线 synthetic classification；
- samples：512；input dim：128；batch size：32；
- 每样本模拟预处理延迟：1 ms；
- workers：0/1/2/4；每项重复 3 次；
- 主指标：完整遍历的 median samples/s；
- 当前规范结果：[`01_pytorch_training/results/dataloader_workers.json`](../../01_pytorch_training/results/dataloader_workers.json)。本页保留早期运行记录，阶段内的[实验 03](../../01_pytorch_training/experiments/03_dataloader_workers.md)为最终解释。

## 复现命令

```powershell
.venv\Scripts\python -m trainscale_training.benchmark_workers `
  --workers 0 1 2 4 `
  --samples 512 --input-dim 128 --batch-size 32 --delay-ms 1 --repeats 3 `
  --output 01_pytorch_training/results/dataloader_workers.json
```

## 结果

环境：Windows 10.0.26200、Python 3.11.15、PyTorch 2.11.0+cpu。

| num_workers | median samples/s | min | max | 相对 workers=0 |
|---:|---:|---:|---:|---:|
| 0 | 640.58 | 637.79 | 641.54 | 1.00x |
| 1 | 206.48 | 204.02 | 210.56 | 0.32x |
| 2 | 232.95 | 209.69 | 233.44 | 0.36x |
| 4 | 232.63 | 230.01 | 233.53 | 0.36x |

## 结论与限制

本配置下 `num_workers=0` 最快。即使每样本加入 1 ms 延迟，512 个样本的工作量仍不足以摊薄 Windows `spawn`、worker 启动与进程间传输成本；2 个 worker 比 1 个略快，但 4 个没有继续提升。因此短小、廉价的 synthetic pipeline 应默认 0 worker，不能照搬 Linux/GPU 图像训练的常见 worker 数。

该实验只测完整 epoch wall time，并用 sleep 模拟 CPU 侧预处理，不能直接代表真实图像解码。后续真实数据集必须重新扫描 workers，并同时观察 CPU、GPU 利用率和 batch 等待时间。
