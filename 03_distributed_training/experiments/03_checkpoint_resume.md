# 实验 03：为什么普通 DDP checkpoint 只让 rank 0 写

## 为什么做

DDP 每个 rank 都有完整模型副本。如果所有 rank 同时写同一路径，会竞争和损坏
文件；只保存模型又会丢失 optimizer 和进度。我们比较连续 4 epoch 与“训练 2
epoch → 保存 → 新 torchrun job 恢复到 epoch 4”。

## 小白名词

- **rank-0 writer**：只有全局 rank 0 执行普通 `torch.save`。
- **barrier**：所有 rank 到达后才继续，用来约束保存/结束顺序。
- **resume state**：模型、optimizer、next epoch 和 seed，而不只是权重。
- **continuous reference**：不中断训练得到的最终状态。

## 一般预期

连续与恢复路线的 epoch 2/3 loss 应相同，最终参数最大误差应为 0 或接近浮点容差；
continuous/partial/resumed 三次保存各自都只能有 1 个 writer。

## 跟着做

阅读 [`worker.py`](../trainscale_distributed/worker.py) 的 `run_train` 与
[`run_correctness.py`](../benchmarks/run_correctness.py) 的 `run_checkpoint`。

```bash
.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment checkpoint --world-size 2 \
  --output 03_distributed_training/results/raw/tutorial/03_checkpoint.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/03_checkpoint.json
```

runner 会在临时目录创建连续、部分、恢复三个 torchrun job，结束后只保留摘要。
终端应为 `checkpoint: success` 和 `all_checks_passed=True`。

## 实际结果

- 每条路线的 rank 参数最大差：`0`；
- 连续 vs 恢复最终参数最大差：`0`；
- 恢复从 epoch 2 开始；
- 三次 checkpoint writer 数均为 1；
- 连续 loss 为 `1.4241 → 1.3818 → 1.3428 → 1.3070`；
- 恢复后的 epoch 2/3 loss 精确复现 `1.3428 / 1.3070`。

## 理论解释

DDP backward 后所有 replica 参数更新一致，所以普通未分片 DDP 只需保存一份
`module.state_dict()`。optimizer 决定下一步更新，next epoch 决定 sampler 的
`set_epoch()`，缺任何一项都可能让恢复轨迹改变。barrier 不能替代完整状态，
它只保证进程执行顺序。

## 结论与收尾

本实验验证同 world size、确定性 synthetic 数据和 SGD 的精确恢复。未来 FSDP
参数被分片时，checkpoint 策略会不同；不要把“DDP rank 0 写一份”直接照搬到 07。
