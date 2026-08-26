# 实验 01：DistributedSampler 会不会漏样本或重复训练

## 为什么做

即使梯度同步完全正确，如果所有 rank 都读取整个数据集，相当于重复计算；如果
分片漏样本，训练分布又被改变。必须直接记录索引，不能只看到 loss 下降就认为
数据管线正确。

## 小白名词

- **shard**：一个 rank 负责的数据子集。
- **shuffle seed**：决定可复现乱序的种子。
- **`set_epoch(epoch)`**：让同一 seed 在不同 epoch 产生不同顺序。
- **padding duplicate**：数据量不能整除 world size 时，sampler 为等长 shard
  补入的重复索引；它与错误地让每 rank 读取全数据不同。

## 一般预期

256 个样本、4 rank 时每 rank 应得 64 个，合并后完整覆盖 0–255 且无 padding。
epoch 0 与 epoch 1 的 shard 顺序应不同。

## 跟着做

阅读 [`worker.py`](../trainscale_distributed/worker.py) 的 `run_sampler`，注意
`sampler.set_epoch(args.epoch)` 在读取索引前调用。

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment sampler --world-size 4 \
  --output 03_distributed_training/results/raw/tutorial/01_sampler.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/01_sampler.json
```

应看到 `set_epoch_changed_order: True`，两个 epoch 的 `coverage=True`、
`padding_duplicates=0`、`samples_per_rank=[64,64,64,64]`。

## 实际结果

本机结果完全符合预期：两个 epoch 都覆盖 256 个唯一索引，没有缺失和 padding，
每 rank 64 个；调用 `set_epoch(1)` 后顺序发生变化。

## 理论解释

DistributedSampler 先生成一个全局确定性排列，再按 rank 步进切片。所有 rank
必须使用相同 seed 和 epoch，否则 shard 可能重叠或遗漏。`set_epoch()` 不改变
样本集合，只改变每个 epoch 的随机排列；忘记调用时每个 epoch 会重复同一顺序。

## 结论与收尾

当前 256/4 是整除情形，证明无错误重复。非整除数据允许 sampler padding，因此
教程的测试函数单独统计 coverage 和 padding，不能把任何重复都武断判错。
