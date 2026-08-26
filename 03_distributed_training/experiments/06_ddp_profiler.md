# 实验 06：Profiler 怎样看到 DDP 的 AllReduce

## 为什么做

吞吐下降只告诉我们“慢了”，Profiler 才能确认通信调用是否出现、每 rank 调用
次数是否一致，以及等待发生在哪一侧。本实验用 2 rank Gloo，避免把单 GPU NCCL
world=1 当成通信证据。

## 小白名词

- **trace**：按时间记录 CPU operator 的事件文件。
- **key averages**：按 operator 名称聚合后的 count 和累计 CPU time。
- **`gloo:all_reduce`**：Gloo 后端的梯度规约事件。
- **嵌套行**：高层 DDP forward 和底层 operator 可能包含彼此，不能把所有行相加。

## 一般预期

测量 5 个 backward step 时，两个 rank 都应捕获 5 次 AllReduce；每 rank 时间可能
不同，因为一个 rank 早到同步点后等待另一个 rank。

## 跟着做

```bash
mkdir -p 03_distributed_training/results/raw/tutorial/06_traces

.venv/bin/python 03_distributed_training/benchmarks/run_profile.py \
  --world-size 2 --steps 5 \
  --trace-directory 03_distributed_training/results/raw/tutorial/06_traces \
  --output 03_distributed_training/results/raw/tutorial/06_profile.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/06_profile.json
```

终端应显示 `ranks=2` 和 `all_checks_passed=True`。trace JSON 可用 Chrome trace
viewer 或 Perfetto 打开；提交 Git 的只是摘要，trace 留在 ignored raw 目录。

## 实际结果

| rank | `gloo:all_reduce` count | 聚合 CPU total |
|---:|---:|---:|
| 0 | 5 | 2,863.36 µs |
| 1 | 5 | 3,159.88 µs |

两个 rank 也都记录了 5 次 `DistributedDataParallel.forward`。

## 理论解释

DDP autograd hook 在梯度 ready 后触发 bucket AllReduce。这个微型模型的参数可被
少量 bucket 覆盖，因此每 step 观察到一次 Gloo AllReduce。rank 1 累计时间更高
可能包含等待；key averages 是聚合 operator 时间，不是端到端 wall time。

## 结论与收尾

Profiler 证明实际执行了通信，并支持 CPU scaling 中同步开销的解释。它不提供
NCCL 带宽或多 GPU overlap 结论；04 会用 nccl-tests 深挖 collective 性能。
