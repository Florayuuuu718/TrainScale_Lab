# 03 · Distributed Training

> 状态：03 已完成。CPU/Gloo 的进程语义、数据分片、梯度同步、checkpoint/resume、
> 1/2/4 rank scaling 与通信 Profiler 均已实测；本地单 GPU NCCL 基线通过，并在
> AutoDL 单机 4×RTX 4090D 上完成三次 1/2/4 GPU strong/weak scaling。8 GPU 因
> 租用实例只有 4 张卡继续标为 `unavailable`，没有伪造数据。

03 从 01 的单进程训练循环出发，把同一个模型复制到多个进程：每个 rank 读取
不同样本、独立做 forward/backward，再由 DDP 在 backward 中同步梯度。这里学习
的不只是 `torchrun` 命令，而是怎样证明“没有重复训练、所有 rank 参数一致、恢复
结果一致、吞吐统计口径正确”。

## 1. 我们正在做什么

```text
global batch
   ├── rank 0: shard 0 → model replica → local gradients ─┐
   ├── rank 1: shard 1 → model replica → local gradients ─┼→ AllReduce/average
   └── rank N: shard N → model replica → local gradients ─┘
                                                        ↓
                              every rank applies the same optimizer step
```

完成本模块后，应能解释：

- process、rank、local rank、world size 和 process group 分别是什么；
- Gloo 与 NCCL 为什么用于不同教学/性能路线；
- `DistributedSampler` 为什么必须每个 epoch 调用 `set_epoch()`；
- DDP 梯度平均怎样等价于同一 global batch 的单进程梯度；
- 为什么只有 rank 0 写普通 DDP checkpoint，但所有 rank 都要参与 collective；
- strong scaling、weak scaling、speedup 和 scaling efficiency 的口径；
- 为什么 CPU 多 rank 变慢不能推出“多 GPU DDP 也没用”。

## 2. 目录地图

```text
03_distributed_training/
├── README.md                       # 本教程入口
├── ENVIRONMENT.md                  # Gloo/NCCL/torchrun 环境门
├── trainscale_distributed/
│   ├── contract.py                 # TOML、sampler 与 scaling 公式
│   └── worker.py                   # 每个 torchrun rank 执行的真实 DDP 代码
├── benchmarks/
│   ├── launcher.py                 # 隔离 torchrun job，收集每 rank JSON
│   ├── run_correctness.py          # 语义/sampler/梯度/checkpoint
│   ├── run_scaling.py              # CPU/GPU strong/weak scaling
│   ├── run_profile.py              # 两 rank Gloo/DDP Profiler
│   ├── aggregate_scaling_runs.py   # 三次云端结果校验与中位数汇总
│   └── show_distributed_results.py # 初学者终端结果表
├── configs/                        # correctness、smoke、formal 配方
├── tests/                          # CPU 契约与 Linux 两 rank 集成测试
├── experiments/                    # 00–07：命令、结果、理论与边界
└── results/
    ├── *.json                      # 本机与云端正式摘要
    ├── evidence/cloud_4x4090d/     # 四卡环境、拓扑、smoke 与三次原始 JSON
    └── raw/                        # trace/checkpoint/练习结果，Git 忽略
```

## 3. 环境与公共准备

正式路线在 WSL2 Ubuntu/Linux 的 `/home/...` 仓库运行。进入前应完成 01 与 02，
继续使用根 `.venv`，不要为 03 再混装一套 PyTorch：

```bash
cd ~/projects/TrainScale_Lab
uv sync --extra cu129 --extra dev --python 3.11
mkdir -p 03_distributed_training/results/raw/tutorial

.venv/bin/python 03_distributed_training/benchmarks/check_environment.py
```

只要 `cpu_distributed_ready=True`，没有 GPU 也能完成实验 00–04 和 06。GPU 路线
还应看到 `nccl_available=true`、`cuda_available=true`。详细安装和排错见
[环境教程](ENVIRONMENT.md)。

## 4. 像 01/02 一样逐项亲手做

每份实验都按“读源码 → 运行 correctness → 快速实验 → 查看结果 → 正式复现”
组织。第一次将结果写入 Git 忽略的 `results/raw/tutorial/`，不要覆盖正式 JSON：

| 顺序 | 实验 | 快速入口 |
|---|---|---|
| 00 | [torchrun 与 rank 语义](experiments/00_process_group.md) | 2 rank Gloo semantics |
| 01 | [DistributedSampler](experiments/01_distributed_sampler.md) | 4 rank、epoch 0/1 |
| 02 | [DDP 梯度等价](experiments/02_gradient_sync.md) | 4 rank 对 global-batch reference |
| 03 | [Checkpoint/resume](experiments/03_checkpoint_resume.md) | 2 rank 连续/中断/恢复 |
| 04 | [CPU strong/weak scaling](experiments/04_cpu_scaling.md) | smoke TOML |
| 05 | [NCCL 与多 GPU scaling](experiments/05_nccl_scaling.md) | 自动检测 GPU 数量 |
| 06 | [DDP Profiler](experiments/06_ddp_profiler.md) | 2 rank Gloo trace |
| 07 | [租云 GPU 完成四卡实验](experiments/07_cloud_4gpu.md) | 选型、部署、三次测量、校验与关机 |

查看任意结果：

```bash
.venv/bin/python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/tutorial/02_gradient.json
```

终端中的 `success` 只表示该 case 的检查通过；`unavailable` 表示硬件不足且没有
运行，绝不等价于 0 samples/s。Bash 的续行反斜杠 `\` 表示下一行仍属于同一条
命令，终端提示符 `$` 不需要输入。

## 5. 正确性契约

性能实验前必须依次证明：

1. torchrun 提供的 `RANK/WORLD_SIZE/LOCAL_RANK` 与进程组一致；
2. collective 的所有 rank 都进入同一调用，AllReduce/Broadcast 值正确；
3. sampler 完整覆盖数据集，padding 重复与真实重复分开统计；
4. DDP 梯度与同一 global batch 的单进程 reference 在容差内一致；
5. optimizer step 后所有 rank 参数一致；
6. 只有 rank 0 写 checkpoint，恢复后的最终参数与连续训练一致；
7. benchmark 使用最慢 rank 的 elapsed time 计算 global throughput。

这些门失败时，不能继续解释 scaling 数字。DDP 构造、forward、backward 都可能
包含同步点；任一 rank 提前退出或少调用一次 collective 都可能让其他 rank 等待。

## 6. Scaling 契约

- **Strong scaling**：global batch 固定，world size 增大后每 rank batch 变小；
- **Weak scaling**：每 rank batch 固定，global batch 随 world size 增大；
- **Speedup**：`throughput(N) / throughput(1)`；
- **Scaling efficiency**：`speedup / N`；
- 每个 rank 使用相同模型/优化器/步数，先 warm-up，再测 steady-state；
- GPU 使用一进程一 GPU，`nproc-per-node` 不能大于可见 GPU 数；
- CPU ranks 共享同一主机资源，只用于理解同步与争用，不能作为 GPU scaling 代理；
- 不同机器、GPU、网络和功耗状态的绝对吞吐不能混在一条曲线排名。

## 7. 本机与云端结果总览

环境为 WSL2、Python 3.11.16、PyTorch 2.12.1+cu129、Gloo/NCCL 可用、Windows
driver 610.88、RTX 5060 Laptop GPU ×1。

| 证据 | 本机结论 |
|---|---|
| 2 rank 进程语义 | rank/world size、AllReduce rank sum、Broadcast 全部正确 |
| 4 rank sampler | 256 个样本完整覆盖、无 padding、`set_epoch()` 改变顺序 |
| 4 rank 梯度同步 | 梯度最大误差 `1.12e-8`，step 后参数误差 `1.49e-8` |
| 2 rank resume | 连续/恢复最终参数误差 `0`；每次恰好 1 个 checkpoint writer |
| CPU strong 1/2/4 | `27.1k / 21.2k / 13.7k samples/s`，共享 CPU 下多 rank 变慢 |
| CPU weak 1/2/4 | `13.0k / 13.5k / 15.3k samples/s`，4 rank efficiency `29.5%` |
| NCCL world=1 | strong `171.7k`、weak `118.0k samples/s`，只证明单 GPU路线可运行 |
| NCCL world=2/4/8 | `unavailable`：本机只有 1 张 GPU，不提供虚假吞吐 |
| 2 rank Profiler | 两个 rank 均捕获 5 次 `gloo:all_reduce` |

云端环境为 AutoDL 单机 4×RTX 4090D、Python 3.12.3、PyTorch 2.8.0+cu128、
driver 580.76.05。GPU0/1 与 GPU2/3 分属两个 NUMA 域，组间为 `SYS`，无 NVLink。
三次正式运行取中位数：

| mode | world 1 | world 2 | world 4 | 四卡 speedup / efficiency |
|---|---:|---:|---:|---:|
| strong，global batch=256 | 252.6k | 151.7k | 128.9k | `0.510× / 12.8%` |
| weak，per-rank batch=128 | 126.6k | 145.1k | 255.4k | `2.018× / 50.4%` |

Strong 多卡变慢不是失败：小模型在每 rank batch 降到 64 后计算粒度过小，而梯度
AllReduce、进程调度和跨 NUMA 同步仍在。正式窗口也只有 20 step，单卡三次相对
极差达 15%–18%，所以结论限定为教学机制，不把它包装成 4090D 性能榜单。完整过程
与理论分析见[云端四卡实验](experiments/07_cloud_4gpu.md)。

机器可读结果总门见 [`module03_summary.json`](results/module03_summary.json)，最终发布
检查见 [`module03_acceptance.json`](results/module03_acceptance.json)；原本的
[`module03_acceptance_sm120.json`](results/module03_acceptance_sm120.json) 保留为本地
单卡阶段的历史验收。短 benchmark 会受频率、温度和后台负载影响；教程保留原始值
和离散程度，但结论只依赖数量级和机制。

## 8. 完成定义

- [x] [03 验收清单](../docs/03-issues.md)中的本机可执行项全部关闭；
- [x] Gloo/CPU 多进程语义、sampler、梯度和 checkpoint 有真实测试；
- [x] strong/weak scaling 口径、最慢 rank 计时和派生公式进入代码与测试；
- [x] NCCL 单 GPU DDP 路线真实运行；
- [x] 多 GPU 不可用 case 保留明确原因，不写 0 或猜测值；
- [x] 通信 Profiler、正式 JSON、SHA-256 汇总和小白复现命令均归档；
- [x] 云端 1/2/4 GPU 共三次 strong/weak 正式实验完成并内容寻址；
- [x] 云主机选型、私有仓库上传、环境快慢路线、下载校验与关机止费进入教程；
- [x] 8 GPU 继续作为明确的可选扩展，不影响 03 单机四卡教学验收完成。

继续阅读：[环境](ENVIRONMENT.md) → [配置](configs/README.md) →
[测试](tests/README.md) → [实验](experiments/README.md) → [结果](results/README.md)。
