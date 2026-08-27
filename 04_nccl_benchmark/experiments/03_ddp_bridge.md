# 实验 03：把 03 的 DDP workload 映射到 NCCL 曲线

本地先计算模型载荷：

```bash
python 04_nccl_benchmark/benchmarks/plan_ddp_bridge.py \
  --config 04_nccl_benchmark/configs/ddp_bridge.toml
```

当前 03 MLP 有 2,623,744 个参数，FP32 梯度载荷为 10,494,976 bytes，约 10.01 MiB。
这是由模型和 dtype 推导的理论载荷；实际 bucket、ready 顺序和 collective 时间必须由
多 GPU trace 验证。

租卡时运行：

```bash
python 04_nccl_benchmark/benchmarks/run_ddp_bridge.py \
  --config 04_nccl_benchmark/configs/ddp_bridge.toml \
  --raw-directory 04_nccl_benchmark/results/raw/rental/ddp_bridge \
  --output 04_nccl_benchmark/results/raw/rental/ddp_bridge.json
```

报告把约 10 MiB 所在的 AllReduce 曲线区域、DDP logging bucket 信息和每 rank timeline
并列分析。通信 microbenchmark 只能提供通信上限，不能单独解释 forward/backward、
进程调度和最慢 rank 等训练开销。

