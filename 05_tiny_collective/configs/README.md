# 05 Configs

- `cpu_correctness.toml`：2/3/4 rank Gloo、ragged element counts、centralized/ring 数值与 trace。
- `gpu_comparison.toml`：2/4 GPU 上固定相同 dtype、消息、warm-up 和迭代，对照 centralized、
  ring 与 `torch.distributed.all_reduce`；包含 04 的 10.01 MiB DDP payload。

CPU 配置是开发门，GPU 配置只在所有 CPU correctness case 通过后运行。
