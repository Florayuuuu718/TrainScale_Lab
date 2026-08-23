# 实验 03（补充）：真实 JPEG 解码与长 epoch 下的 workers

## 问题

实验 03 的短 synthetic 数据管线中，worker 的进程启动和 IPC 成本超过了并行收益。换成磁盘 JPEG 读取、真实解码与随机图像增强，并把 epoch 拉长后，增加 `num_workers` 是否能提高预热后的稳态吞吐？

## 概念与为什么需要补充实验

“数据集更长”本身不会让单个样本更快。它的作用是摊薄 worker 启动等固定成本；多 worker 能否真正提速，仍取决于每个样本的 I/O、解码和变换工作量，以及 CPU、磁盘和进程通信的瓶颈。因此本实验同时改变两个与原实验边界直接相关的条件：

- 用磁盘 JPEG 读取、解码、`RandomResizedCrop`、随机翻转和归一化代替 `sleep`；
- 用较长 epoch，并启用 `persistent_workers`，把首 epoch 和预热后的稳态 epoch 分开报告。

## 配置与复现

脚本可使用已有 JPEG 目录，也可生成确定性的本地 JPEG 样本。指定 `--prepare-images` 时会覆盖同名的 `sample_*.jpg` 编号文件，建议使用专用目录。生成图片只用于建立可复现的磁盘输入，耗时不计入吞吐。

```bash
.venv/bin/python -m trainscale_training.benchmark_image_workers `
  --image-root 01_pytorch_training/data/worker_jpegs `
  --prepare-images 512 --source-size 160 --image-size 128 `
  --workers 0 1 2 4 --samples 16384 --batch-size 64 `
  --warmup-epochs 1 --timed-epochs 3 `
  --output 01_pytorch_training/results/dataloader_image_workers.json
```

控制变量包括 JPEG 文件、每 epoch 样本数、batch size、图像尺寸、增强、预取因子和随机种子。`samples` 可以大于 JPEG 文件数：数据集循环映射同一批文件，以延长 epoch，而不生成大量重复文件。

## 指标解释

- `first_epoch_samples_per_second`：包含 worker spawn 和首次迭代开销，更接近短任务体验；worker 按 0/1/2/4 顺序扫描，因此不把它解释为严格的操作系统冷缓存结果；
- `median_steady_samples_per_second`：完成预热后多个完整 epoch 的中位吞吐，是长训练的主要比较指标；
- min/max：用于判断结果波动，不能只挑最快一次；
- 本实验只测数据管线，不执行模型 forward/backward，因此不能直接证明 GPU 利用率会提高。

## 实测结果

环境：WSL2 Ubuntu、Python 3.11.16、PyTorch 2.12.1+cu129。原始结果：[`dataloader_image_workers.json`](../results/dataloader_image_workers.json)。

| workers | first epoch samples/s | median steady samples/s | min | max | 相对 workers=0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1345.80 | 1501.21 | 1418.17 | 1558.33 | 1.00x |
| 1 | 1661.43 | 1682.65 | 1631.03 | 1715.60 | 1.12x |
| 2 | 3211.73 | 3296.58 | 3124.76 | 3323.52 | 2.20x |
| 4 | 5867.77 | 5743.17 | 5535.80 | 5776.04 | 3.83x |

## 分析

结果支持“较重且持续运行的数据管线通常能从并行 worker 获益”。在本机和本次 0/1/2/4 扫描范围内，稳态吞吐随 workers 单调提升：1 worker 为 1.12 倍，2 workers 为 2.20 倍，4 workers 为 3.83 倍。这说明 JPEG 解码和图像增强具有足够的可并行 CPU 工作，也补足了短 synthetic 实验中未能展示的常见训练场景。

首 epoch 与稳态结果也不能混为一谈。1 worker 的稳态只比 0 worker 高约 12%，而 2/4 workers 的并行度才明显隐藏了解码与增强等待。4 workers 的首 epoch 为 5867.77 samples/s，稳态中位数为 5743.17 samples/s；这里首轮并未更慢，说明进程启动成本已被长 epoch 摊薄。min/max 仍提醒我们必须报告中位数和波动范围。当前扫描尚未出现 4 workers 之后的拐点，因此不能声称 4 是最佳值，更不能据此断言 workers 会无限单调提速；若机器资源允许，应继续测 8、16 workers，直到吞吐持平或下降。

## 有限结论与一般结论

有限结论是：在本次 Ubuntu、512 个 JPEG 循环构成的 16,384 samples/epoch、128×128 增强管线中，4 workers 的稳态吞吐约为 0 worker 的 3.83 倍；扫描尚未出现拐点，所以不能断言 4 是最佳值。

一般结论是：workers 只有在可并行的 I/O、解码或增强足够重、且运行足够长时才容易带来收益；增加过多又会受到 CPU 核数、磁盘、内存、IPC 与上下文切换限制。生成 JPEG、重复文件访问和文件缓存仍不等于大型生产数据集，也未包含模型 forward/backward。每个新数据管线都应在目标机器重新扫描 workers，并联合 GPU 利用率与 batch 等待时间判断。
