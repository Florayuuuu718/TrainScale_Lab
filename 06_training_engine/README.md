# 06 · Mini Training Engine + Gradient Reducer Lab

> 状态：本地 CPU/Gloo 实现与 correctness gate 已完成；CUDA/NCCL 消融和 overlap timeline
> 待统一租卡实验。

06 不重写 01 的通用训练循环，也不把 03 的 DDP 再包装一次。它用同一个 Tiny Transformer
逐步实现 bulk、per-parameter、bucketed synchronous 和 bucketed asynchronous gradient
reducer，再与 PyTorch DDP 比较。

## 已实现内容

- 供 06/07 复用的 small/medium Tiny Transformer；
- deterministic reverse-order bucket ownership、offset、flat buffer 和 plan digest；
- bulk、per-parameter hook、bucket sync、bucket async reducer；
- accumulation 非同步 micro-step、AMP scale/unscale、gradient clipping 和 optimizer 边界；
- unused/None gradient、异步 handle wait 和跨 rank bucket-plan 预检；
- 复用 01 checkpoint schema 的稳定 adapter；
- CPU/Gloo global-batch reference correctness runner；
- 2/4 GPU targeted ablation 与 4 GPU CUDA profiler runner。

源码入口：

- `trainscale_engine/model.py`：Tiny Transformer 与模型规模；
- `trainscale_engine/bucket.py`：bucket plan 和 digest；
- `trainscale_engine/reducer.py`：四种手写 reducer；
- `trainscale_engine/engine.py`：accumulation、AMP、clip、step 生命周期；
- `trainscale_engine/worker.py`：torchrun correctness/benchmark/profile worker。

## 本地 gate

```bash
python 06_training_engine/benchmarks/run_single_device_baseline.py \
  --config 06_training_engine/configs/local_baseline.toml \
  --output 06_training_engine/results/raw/local-baseline.json

python 06_training_engine/benchmarks/run_reducer_correctness.py \
  --config 06_training_engine/configs/local_correctness.toml \
  --raw-directory 06_training_engine/results/raw/local-correctness \
  --output 06_training_engine/results/raw/local-correctness.json

python 06_training_engine/benchmarks/summarize_module06.py \
  --baseline 06_training_engine/results/raw/local-baseline.json \
  --correctness 06_training_engine/results/raw/local-correctness.json \
  --output 06_training_engine/results/raw/module06-local-acceptance.json
```

矩阵是五种策略 × accumulation 1/2，共 10 个 2-rank Gloo case，并启用一个故意未参与
forward 的参数。每个 rank 的同步梯度和 optimizer update 都与单进程 global-batch reference
比较。CPU gate 证明数学与状态机，不证明 NCCL 性能或真实 CUDA overlap。

## 统一租卡 gate

先检查 20 个单变量条件，不启动 GPU：

```bash
python 06_training_engine/benchmarks/run_gpu_ablation.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --output /tmp/module06-plan.json \
  --dry-run
```

正式性能与 timeline：

```bash
python 06_training_engine/benchmarks/run_gpu_ablation.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory /root/trainscale-results/module06/ablation-raw \
  --output /root/trainscale-results/module06/ablation.json

python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory /root/trainscale-results/module06/profile-raw \
  --output /root/trainscale-results/module06/overlap-profile.json

python 06_training_engine/benchmarks/run_amp_overflow_probe.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory /root/trainscale-results/module06/amp-overflow-raw \
  --output /root/trainscale-results/module06/amp-overflow.json
```

性能矩阵采用 targeted one-factor ablation：20 个条件 × 3 次，而不是 720 个组合。AMP
overflow probe 另测 2/4 卡的 bucket-async 与 DDP，验证 scale 下降、step skip 和参数不变。Profiler
只比较 bucket sync、bucket async、DDP。所有命令、日志、rank JSON、Chrome trace 都持久保存并
写入 SHA-256。

## 解释边界

- async collective 在 backward 结束前 launch 只是 overlap candidate；只有 CUDA/NCCL timeline
  中通信区间与 backward kernel 真正重叠，才能宣称 overlap；
- per-parameter 理论上最早发起通信，但 collective 数量和启动开销可能使它最慢；
- bucket cap 过小增加启动开销，过大推迟首个 collective，最优值依赖模型和链路；
- 自研 reducer 不支持稀疏梯度、动态图、跨 rank 动态 unused 分歧和生产级故障恢复；
- 优化没有加速仍可验收，但必须保留正确性、变异度和 profiler 证据。

实验说明见 [`experiments/`](experiments/)，逐项状态见
[`docs/06-issues.md`](../docs/06-issues.md)。
