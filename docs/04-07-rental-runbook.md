# 04–07 统一四卡租用实验 Runbook

> 本 runbook 是已经完成 campaign 的工程速查表。第一次租卡请使用更细的
> [JupyterLab 一站式教程](getting-started/jupyterlab-4gpu.md)，它解释界面、命令、预期输出、
> 下载和关机。下面命令用于熟悉 shell 的学习者复跑。

## 硬件决定

租单机 **4×同型号 NVIDIA GPU**，优先复用上一次的 4×RTX 4090 D 24 GiB 主机和镜像。当前正式配置的最大 world size 都是 4；8 GPU、多节点和 2D 并行不是 v1.0 gate。租 8 卡只会增加费用并改变拓扑，现有 runner 不会使用额外四卡。

推荐条件：Ubuntu 22.04、可用 NCCL、至少 50 GiB 可写空间、足够保存 Chrome trace。启动后固定：

```bash
export PROJECT_ROOT=${PROJECT_ROOT:-/root/TrainScale_Lab}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_ROOT=/root/trainscale-results/rental-0407
mkdir -p "$RUN_ROOT"/{environment,module04,module05,module06,module07}
```

若仓库目录带平台生成的后缀，先用 `find /root -maxdepth 3 -name pyproject.toml -type f` 找到它，
再重新设置 `PROJECT_ROOT`。

同一次 campaign 不升级 PyTorch/CUDA/NCCL，不更换 GPU 顺序，不同时运行无关 GPU 作业。若不能复用原主机，必须把新拓扑视为新环境，不能把新旧吞吐直接混合聚合。

## 0. 启动后先做环境与代码门

```bash
cd "$PROJECT_ROOT" || exit 1
git status --short
git rev-parse HEAD | tee "$RUN_ROOT/environment/project-git-commit.txt"
nvidia-smi | tee "$RUN_ROOT/environment/nvidia-smi.txt"
nvidia-smi topo -m | tee "$RUN_ROOT/environment/topology.txt"
python --version | tee "$RUN_ROOT/environment/python.txt"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())" \
  | tee "$RUN_ROOT/environment/torch.txt"
df -h | tee "$RUN_ROOT/environment/disk.txt"

python 04_nccl_benchmark/benchmarks/check_environment.py \
  --require-multi-gpu \
  --output "$RUN_ROOT/environment/module04-environment.json"

python -m pytest -q \
  05_tiny_collective/tests \
  06_training_engine/tests \
  07_parallelism/tests
```

必须满足：仓库是准备好的 clean commit、四卡都空闲、`cuda_device_count=4`、NCCL 可用、测试通过。任一失败就先排障，不开始性能采集。

## 1. 先复跑本地语义门

这些任务主要使用 CPU/Gloo，成本很低，但能确认租用镜像和当前 commit 没有兼容性漂移。

```bash
python 05_tiny_collective/benchmarks/run_correctness.py \
  --config 05_tiny_collective/configs/cpu_correctness.toml \
  --output "$RUN_ROOT/module05/cpu-correctness.json"

python 06_training_engine/benchmarks/run_single_device_baseline.py \
  --config 06_training_engine/configs/local_baseline.toml \
  --output "$RUN_ROOT/module06/local-baseline.json"

python 06_training_engine/benchmarks/run_reducer_correctness.py \
  --config 06_training_engine/configs/local_correctness.toml \
  --raw-directory "$RUN_ROOT/module06/local-correctness-raw" \
  --output "$RUN_ROOT/module06/local-correctness.json"

python 07_parallelism/benchmarks/estimate_memory.py \
  --output "$RUN_ROOT/module07/memory-estimate.json"
python 07_parallelism/benchmarks/run_tp_correctness.py \
  --config 07_parallelism/configs/local_correctness.toml \
  --raw-directory "$RUN_ROOT/module07/tp-local-raw" \
  --output "$RUN_ROOT/module07/tp-local.json"
python 07_parallelism/benchmarks/run_fsdp2_capability.py \
  --raw-directory "$RUN_ROOT/module07/fsdp2-local-raw" \
  --output "$RUN_ROOT/module07/fsdp2-local.json"
python 07_parallelism/benchmarks/run_native_tp_capability.py \
  --raw-directory "$RUN_ROOT/module07/native-tp-local-raw" \
  --output "$RUN_ROOT/module07/native-tp-local.json"
```

## 2. 正式性能任务

按 04→05→06→07 顺序执行。每个命令结束后立刻检查退出码和 artifact `status`；失败时停止该模块后续任务，保留日志，不覆盖目录重跑。

### 04：只补长窗口 DDP 稳定性

此前 collective、拓扑和 DDP bridge 已采集，本轮必需补测是 5 次长窗口 strong/weak、world size 1/2/4 campaign：

```bash
python 04_nccl_benchmark/benchmarks/run_ddp_scaling_campaign.py \
  --config 04_nccl_benchmark/configs/ddp_scaling_long.toml \
  --output-directory "$RUN_ROOT/module04/ddp-scaling-long" \
  --summary-output "$RUN_ROOT/module04/ddp-scaling-long.json" \
  --repetitions 5 \
  --stability-threshold 0.05 \
  --warning-threshold 0.10 \
  --timeout-seconds 1800
```

退出码 2 表示测量质量未达门槛，不等同于 correctness 失败。此时保留五次范围和中位数，但不能报告高精度 speedup/efficiency。若换了主机，只额外做一次 NCCL smoke 验证链路，不自动重跑上一轮全部正式曲线。

### 05：TinyCollective 与 NCCL 对照

```bash
python 05_tiny_collective/benchmarks/run_gpu_comparison.py \
  --config 05_tiny_collective/configs/gpu_comparison.toml \
  --raw-directory "$RUN_ROOT/module05/gpu-raw" \
  --output "$RUN_ROOT/module05/gpu-comparison.json"
```

矩阵为 2/4 GPU × centralized/ring/torch，共 6 个条件、18 个独立 torchrun 作业；每个作业内部测 8 个消息大小。教学版 Python ring 变慢是有效结果，不应包装为 NCCL 算法优劣。

### 06：Reducer ablation 与 overlap timeline

```bash
python 06_training_engine/benchmarks/run_gpu_ablation.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/ablation-raw" \
  --output "$RUN_ROOT/module06/ablation.json"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/profile-raw" \
  --output "$RUN_ROOT/module06/overlap-profile.json"

python 06_training_engine/benchmarks/run_amp_overflow_probe.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/amp-overflow-raw" \
  --output "$RUN_ROOT/module06/amp-overflow.json"
```

性能矩阵为 20 个 targeted 条件×3 次，即 60 个作业；profile 另有 bucket-sync、bucket-async、DDP 三个 4 卡作业。overflow probe 另有 2/4 卡×bucket-async/DDP 四个作业，必须同时验证 GradScaler scale 下降、step skip 和参数逐位不变。只有 CUDA trace 中 NCCL 区间与 backward kernel 实际重叠，才能宣称 overlap。

### 07：FSDP2 / TP correctness、显存、吞吐与 trace

```bash
python 07_parallelism/benchmarks/run_gpu_parallelism.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --raw-directory "$RUN_ROOT/module07/gpu-raw" \
  --output "$RUN_ROOT/module07/gpu-parallelism.json"

python 07_parallelism/benchmarks/run_gpu_profiles.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --correctness-artifact "$RUN_ROOT/module07/gpu-parallelism.json" \
  --raw-directory "$RUN_ROOT/module07/profile-raw" \
  --output "$RUN_ROOT/module07/gpu-profiles.json"
```

第一条命令先运行 4 个 2/4 卡 FSDP2/原生 TP correctness preflight；全部通过后才执行 11 个条件×3 次，即 33 个性能作业。profile 另有 DDP、layer-wrap FSDP2、TP 三个 4 卡作业。

当前 small/medium preset 明显不足以在 24 GiB 4090 D 上自然构成 OOM。不要用其他进程抢显存制造 OOM；若没有新增、经过本地验证的大模型 preset，就把 07-05 记录为 `unavailable`，这不阻塞当前 v1.0 的 FSDP2/TP 语义和性能比较。2D parallel 保持关闭。

## 3. 汇总与关机门

```bash
python 06_training_engine/benchmarks/summarize_module06.py \
  --baseline "$RUN_ROOT/module06/local-baseline.json" \
  --correctness "$RUN_ROOT/module06/local-correctness.json" \
  --gpu-ablation "$RUN_ROOT/module06/ablation.json" \
  --overlap-profile "$RUN_ROOT/module06/overlap-profile.json" \
  --amp-overflow "$RUN_ROOT/module06/amp-overflow.json" \
  --output "$RUN_ROOT/module06/acceptance.json"

python 07_parallelism/benchmarks/summarize_module07.py \
  --memory "$RUN_ROOT/module07/memory-estimate.json" \
  --tp-correctness "$RUN_ROOT/module07/tp-local.json" \
  --fsdp2-capability "$RUN_ROOT/module07/fsdp2-local.json" \
  --native-tp-capability "$RUN_ROOT/module07/native-tp-local.json" \
  --gpu-parallelism "$RUN_ROOT/module07/gpu-parallelism.json" \
  --gpu-profiles "$RUN_ROOT/module07/gpu-profiles.json" \
  --output "$RUN_ROOT/module07/acceptance.json"

# 可选：复现“有 overlap 但更慢”的 1 MiB bucket 延伸实验
python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --bucket-cap-mb 1.0 \
  --raw-directory "$RUN_ROOT/module06/extension-1m/profile-raw" \
  --output "$RUN_ROOT/module06/extension-1m/overlap-profile-1m.json"

find "$RUN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$RUN_ROOT/SHA256SUMS"
tar -C /root/trainscale-results -czf /root/trainscale-rental-0407.tar.gz rental-0407
sha256sum /root/trainscale-rental-0407.tar.gz \
  > /root/trainscale-rental-0407.tar.gz.sha256
```

关机前必须确认：四个模块的目标 artifact 都存在、失败日志没有被覆盖、`SHA256SUMS` 已生成、压缩包和外层 SHA-256 已复制到本地并抽查可解压。

## 数量与时间预算

| 模块 | 正式 GPU 工作量 | 必需结论 |
|---|---:|---|
| 04 | 5 次 campaign，内部共 30 个 1/2/4 卡 case | 单卡 baseline 稳定性和可报告精度 |
| 05 | 18 个作业，48 条聚合消息曲线 | Python collective 与 NCCL 的代价差异 |
| 06 | 60 个性能作业 + 3 个 profile + 4 个 overflow probe | reducer、bucket、AMP、accumulation、overlap |
| 07 | 4 个 preflight + 33 个性能作业 + 3 个 profile | FSDP2/TP correctness、显存和吞吐 |

总计约 155 个 torchrun/case 启动。按当前 tiny workload，建议为 **2 小时目标窗口、3 小时费用余量** 做准备；实际以网络、进程启动和 profiler trace 写盘速度为准。不要为了卡在整点内删除重复次数或缩短测量窗口。
