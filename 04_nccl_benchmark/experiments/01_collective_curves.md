# 实验 01：四种 collective 的消息曲线

先运行 smoke：

```bash
python 04_nccl_benchmark/benchmarks/run_collectives.py \
  --config 04_nccl_benchmark/configs/nccl_smoke.toml \
  --binary-directory /root/autodl-tmp/nccl-tests/build \
  --output 04_nccl_benchmark/results/raw/rental/nccl_smoke.json
```

smoke 四个 case 全部 `success` 后，才把配置换成 `nccl_formal.toml`。报告按消息大小
画 time/algbw/busbw，并分别解释 AllReduce、AllGather、ReduceScatter 和 Broadcast；
不能把不同 collective 的 busbw 归一化公式混为一谈。

