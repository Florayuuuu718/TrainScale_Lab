# 05 · TinyCollective

> 状态：已完成。24 个 CPU/Gloo correctness case 与 2/4 GPU centralized/ring/NCCL
> 对照均通过，正式结论和边界已归档。

TinyCollective 不是 NCCL 的替代品。它用最小、可读的 P2P 实现拆开 AllReduce，让初学者先实现
centralized 与 ring，再测量 Python 调度、同步和额外内存操作造成的性能差距。

## 开始前先建立联系

04 已经画出了 AllReduce 曲线，但 NCCL 会自动隐藏算法细节。05 暂时放下生产性能，用可读的
send/recv 把“谁把哪个 chunk 发给谁”展开，再回到 NCCL 对照工程差距。第一次遇到
ReduceScatter、AllGather 或 ragged chunk 时，可先查
[术语表](../docs/concepts/distributed-systems-glossary.md)。

## 已实现内容

- centralized reduce + broadcast；
- ragged ring reduce-scatter + all-gather + all-reduce；
- 每一步显式记录 sender、receiver、chunk 与 send/recv tag；
- `batch_isend_irecv` handle 的一次性 `wait()` 生命周期；
- 2/3/4 rank CPU/Gloo correctness matrix；
- 2/4 GPU centralized、ring 与 `torch.distributed.all_reduce` 的同条件对照 runner。

源码入口：

- `trainscale_collective/schedule.py`：纯调度与通信量模型；
- `trainscale_collective/algorithms.py`：centralized/ring 实现；
- `trainscale_collective/worker.py`：分布式 correctness 与 benchmark worker；
- `benchmarks/run_correctness.py`：CPU/Gloo 验收；
- `benchmarks/run_gpu_comparison.py`：GPU/NCCL 对照。

## 本地 correctness gate

完整 2/3/4-rank Gloo runner 建议在 Linux/WSL CPU 环境运行；Windows 可先运行 schedule、
通信量和配置单元测试。若 Windows pytest 显示集成项被跳过，使用
[Linux/Gloo 正确性门教程](../docs/getting-started/linux-gloo-validation.md)补跑，不需要 GPU。

```bash
python 05_tiny_collective/benchmarks/run_correctness.py \
  --config 05_tiny_collective/configs/cpu_correctness.toml \
  --output 05_tiny_collective/results/raw/cpu_correctness.json
```

当前配置覆盖 24 个 case：world size 2/3/4，元素数 5/7/16/17，以及 centralized/ring。
5、7、17 专门验证不能被 world size 整除的 uneven chunk。本 gate 只证明调度和数学正确，
不证明 CUDA/NCCL 可用或性能合理。

## GPU gate（已完成，可复现）

在 4 GPU Linux 节点、仓库干净且依赖检查通过后运行：

```bash
python 05_tiny_collective/benchmarks/run_gpu_comparison.py \
  --config 05_tiny_collective/configs/gpu_comparison.toml \
  --raw-directory /root/trainscale-results/module05/raw \
  --output /root/trainscale-results/module05/gpu-comparison.json
```

runner 会保留每个 job 的命令、stdout/stderr 与 rank JSON，并在总 artifact 中记录 SHA-256。
消息矩阵包含 04 的 10,494,976-byte DDP gradient payload。正式结论只使用重复实验中位数；
若短消息噪声较大，应如实报告，不把教学 Python ring 的慢解释成 NCCL 算法本身慢。

## 本章结论

4 GPU、64 MiB 时 centralized/ring/NCCL 分别为 3.069/7.617/9.317 GB/s。实验验证了
ring 对 root 热点的缓解，也验证了“算法结构正确”与“生产实现高效”之间仍有巨大工程空间。
小消息波动很大，不用于排名。

先按 [实验导航](experiments/README.md) 画消息流、做 correctness 和写预测；运行后再读
[最终报告](experiments/04_final_report.md)。四卡操作见
[JupyterLab 教程](../docs/getting-started/jupyterlab-4gpu.md)，机器可读摘要见
[results/module05_final_summary.json](results/module05_final_summary.json)。

## 阶段边界

- CPU/Gloo 是开发门，GPU/NCCL 是性能门，两者不可互相替代；
- 第一版仅支持单机、SUM 和连续 tensor；
- 不承诺故障恢复、任意拓扑优化、生产性能或多节点能力；
- 06 可以把它作为可选教学 backend，但默认训练路径仍使用 PyTorch/NCCL。

推导与实验设计见 [`experiments/`](experiments/)，配置见 [`configs/`](configs/)，逐项验收见
[`docs/05-issues.md`](../docs/05-issues.md)。

学完后进入 [06 · Mini Training Engine](../06_training_engine/README.md)：05 的 collective
将不再是孤立算法，而会被放进 backward 和 optimizer step 的真实生命周期。
