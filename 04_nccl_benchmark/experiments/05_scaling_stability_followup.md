# 实验 05：DDP scaling 长窗口稳定性补测

第一次 4×RTX 4090 D 采集复用了 03 的短配置：5 warm-up steps、20 measured steps。
由于这个 MLP 很小，每个 case 的测量窗口只有约 19–49 ms。2/4 GPU 吞吐重复较稳定，
但 single-GPU strong/weak baseline 的三次相对极差约为 23%/50%。因此原始结果可以证明
真实执行与总体趋势，却不能支撑高精度 speedup/scaling-efficiency 结论。

## 单变量修正

补测保持以下变量不变：

- Linear–ReLU–Linear 模型维度、seed、SGD 和学习率；
- strong/weak batch 语义；
- NCCL backend、1/2/4 world size 和一进程一 GPU；
- 同一次 campaign 内的主机、镜像、驱动、PyTorch 和设备顺序。

只修改测量协议：warm-up 从 5 增加到 200 steps，measured window 从 20 增加到
5000 steps，并执行五次独立重复。旧的短窗口结果不与新结果混合聚合。

## 租卡入口

在干净提交上运行：

```bash
python 04_nccl_benchmark/benchmarks/run_ddp_scaling_campaign.py \
  --config 04_nccl_benchmark/configs/ddp_scaling_long.toml \
  --output-directory 04_nccl_benchmark/results/raw/rental/ddp-scaling-long \
  --summary-output 04_nccl_benchmark/results/raw/rental/ddp-scaling-long.json \
  --repetitions 5 \
  --stability-threshold 0.05 \
  --warning-threshold 0.10 \
  --timeout-seconds 1800
```

runner 保存每次结构化 JSON 及 stdout/stderr，再以五次吞吐中位数聚合。稳定性指标为：

```text
(maximum throughput - minimum throughput) / median throughput
```

## 决策门

- `<= 5%`：measurement quality passed，可以报告中位数 speedup/efficiency；
- `5%–10%`：保留区间和中位数，但结论注明波动；
- `> 10%`：不报告精确扩展效率，只报告吞吐范围和未解决限制。

正确性与测量质量分开记录：即使所有进程和参数同步正确，稳定性门也可以失败。失败时
runner 仍会写出 artifact，并以退出码 2 提醒不要把不稳定数值包装成精确结论。

租卡前可在 CPU 环境检查配置和将要执行的五条命令，不启动训练：

```bash
python 04_nccl_benchmark/benchmarks/run_ddp_scaling_campaign.py \
  --config 04_nccl_benchmark/configs/ddp_scaling_long.toml \
  --output-directory /tmp/trainscale-ddp-scaling-long \
  --summary-output /tmp/trainscale-ddp-scaling-long.json \
  --dry-run
```
