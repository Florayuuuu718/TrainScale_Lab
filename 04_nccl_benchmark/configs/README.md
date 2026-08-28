# 04 Configs

- `nccl_smoke.toml`：2 GPU、四种 collective、最大 1 MiB，只验证构建、调度、解析和正确性。
- `nccl_formal.toml`：2/4 GPU、8 B–256 MiB，正式运行三次；pair 名称必须结合实际拓扑解释。
- `ddp_bridge.toml`：复用 03 的 MLP 维度，在 2/4 GPU 上采集 DDP 通信 timeline。
- `ddp_scaling_long.toml`：保持 03 的模型、batch 和 1/2/4 GPU 口径，只把 warm-up/
  measured window 延长到 200/5000 steps，供五次稳定性补测使用。

配置使用物理可见设备编号。runner 会把每个 case 的 `devices` 写入
`CUDA_VISIBLE_DEVICES`，再以 `-g world_size` 启动单进程多 GPU `nccl-tests`。
不要在运行之间改变设备顺序、GPU 时钟、软件环境或配置。
