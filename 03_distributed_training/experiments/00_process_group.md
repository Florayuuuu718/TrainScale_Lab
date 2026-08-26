# 实验 00：torchrun、rank 与 process group

## 为什么先做这个实验

DDP 的错误经常不是模型数学错误，而是进程没有加入同一个 group、rank 配错，或
collective 调用顺序不一致。先用最小张量验证通信语义，比直接启动长训练更容易
定位问题。

## 小白名词

- **process/worker**：一个独立 Python 进程；每个进程有自己的模型对象。
- **rank**：进程在整个 worker group 中的唯一编号，范围 `0..world_size-1`。
- **local rank**：进程在当前机器上的编号，GPU 路线通常映射到本机 GPU。
- **world size**：参与当前 group 的进程总数。
- **process group**：collective 的参与者集合和通信后端。
- **AllReduce**：每个 rank 提供值，规约后每个 rank 都得到同一个结果。
- **Broadcast**：一个源 rank 的值发送给所有 rank。

## 一般预期

两 rank 的 rank 值分别是 0 和 1，求和应为 1；rank 0 广播 42 后两个进程都应
看到 42。任何 rank 文件缺失、退出码非 0 或值不一致都应失败。

## 跟着做

源码入口是 [`launcher.py`](../benchmarks/launcher.py) 和
[`worker.py`](../trainscale_distributed/worker.py) 的 `run_semantics`。

```bash
mkdir -p 03_distributed_training/results/raw/tutorial

.venv/bin/python 03_distributed_training/benchmarks/check_environment.py

.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment semantics --world-size 2 \
  --output 03_distributed_training/results/raw/tutorial/00_semantics.json

.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/00_semantics.json
```

终端应出现 `semantics: success`、结果路径和 `all_checks_passed=True`。JSON 中应
有 rank 0/1 两条记录，不是父进程假装出的一个 world size 字段。

## 实际结果

本机 2 rank Gloo 全部通过：rank 集合 `{0,1}`，两个 rank 的 world size 均为 2，
AllReduce rank sum 均为 1，Broadcast 均为 42。

## 理论解释

torchrun 父进程只负责创建 worker 并设置 `RANK/WORLD_SIZE/LOCAL_RANK` 等环境
变量。`init_process_group("gloo")` 让 worker 通过 rendezvous 找到彼此。AllReduce
和 Broadcast 是阻塞语义检查点：所有参与 rank 必须按相同顺序进入，否则可能
等待而不是抛出漂亮的 Python 异常。

## 结论与收尾

本实验只证明本机两 rank Gloo 通路与基本 collective 正确，不证明 DDP 梯度、
数据分片或 GPU/NCCL。下一步检查每个 rank 到底拿到了哪些样本。
