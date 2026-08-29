# 07 — 从 DDP 到 FSDP2 与 Tensor Parallel

07 不以“成功启动某个框架配置”为完成标准，而是回答：模型为什么需要切分，应该切分训练状态还是单层计算，代价是什么？本模块复用 06 的 Tiny Transformer，先实现最小分片，再用 reference 对齐、显存分解、通信事件和吞吐实验验证选择。

## 当前状态

本地阶段已经可验证：

- 解析 DDP/FSDP2 的参数、梯度、Adam 状态和 activation 显存下界；
- 实现可处理非整除长度的教学版张量分片、重建和分片 checkpoint；
- 实现 MLP Colwise→Rowwise TP，以及按 head 切分的 attention；
- 在 2/4 rank CPU/Gloo 下对齐输出、梯度分片和一步参数更新；
- 使用真实 `fully_shard` 验证 DTensor `Shard(0)`、一步更新与 distributed checkpoint 恢复；
- 使用真实 `parallelize_module` 验证 Colwise/Rowwise placement 和一步更新；
- 固化 4 卡正确性前置门、性能矩阵和 profiler 计划。

本地证据只说明数学语义和状态恢复成立，不说明 CUDA 显存收益、NCCL 通信代价或扩展效率。后者仍需统一租卡实验。

## 本地验证

在仓库根目录运行：

```powershell
.venv\Scripts\python 07_parallelism/benchmarks/estimate_memory.py `
  --output 07_parallelism/results/memory-estimate.json

.venv\Scripts\python 07_parallelism/benchmarks/run_tp_correctness.py `
  --config 07_parallelism/configs/local_correctness.toml `
  --raw-directory 07_parallelism/results/raw/tp-correctness `
  --output 07_parallelism/results/tp-correctness.json

.venv\Scripts\python 07_parallelism/benchmarks/run_fsdp2_capability.py `
  --raw-directory 07_parallelism/results/raw/fsdp2-capability `
  --output 07_parallelism/results/fsdp2-capability.json

.venv\Scripts\python 07_parallelism/benchmarks/run_native_tp_capability.py `
  --raw-directory 07_parallelism/results/raw/native-tp-capability `
  --output 07_parallelism/results/native-tp-capability.json
```

Windows 跳过依赖 torchrun/Gloo 的集成测试；这些命令应在 Linux CPU 环境执行。当前工作区已经留下真实 Linux CPU/Gloo 结果：自定义 TP 的 4 个条件均通过，最大误差不超过 `1.2e-7`；FSDP2 的 SGD 一步更新最大误差约 `1.5e-8`，checkpoint 恢复后下一步误差为 0；原生 TP 最大误差约 `8.9e-8`。

生成本地验收摘要：

```bash
python 07_parallelism/benchmarks/summarize_module07.py \
  --memory 07_parallelism/results/memory-estimate.json \
  --tp-correctness 07_parallelism/results/tp-correctness.json \
  --fsdp2-capability 07_parallelism/results/fsdp2-capability.json \
  --native-tp-capability 07_parallelism/results/native-tp-capability.json \
  --output 07_parallelism/results/module07-acceptance.json
```

摘要在未提供 GPU artifact 时必须是 `passed_local_gates`，不能是 `complete`。

## 统一 4 卡实验

先检查计划，不占用 GPU：

```bash
python 07_parallelism/benchmarks/run_gpu_parallelism.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --output /root/trainscale-results/module07/gpu-plan.json \
  --dry-run
```

正式运行要求至少 4 张 CUDA GPU 和 NCCL。它先执行 2/4 卡 FSDP2 与原生 TP correctness probe；任何前置门失败都会停止性能矩阵。通过后运行 11 个条件，每个 3 个独立进程作业，报告三次中位数和相对极差：

```bash
python 07_parallelism/benchmarks/run_gpu_parallelism.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --raw-directory /root/trainscale-results/module07/gpu-raw \
  --output /root/trainscale-results/module07/gpu-parallelism.json
```

性能结束后再采集 4 卡 DDP、layer-wrap FSDP2、TP trace：

```bash
python 07_parallelism/benchmarks/run_gpu_profiles.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --correctness-artifact /root/trainscale-results/module07/gpu-parallelism.json \
  --raw-directory /root/trainscale-results/module07/profile-raw \
  --output /root/trainscale-results/module07/gpu-profiles.json
```

profiler 时间只用于定位 AllReduce、AllGather、ReduceScatter，不进入正式吞吐结论。所有命令、rank JSON、日志、trace 和 DCP 文件都会记录 SHA-256。

## 策略选择树

1. 模型、训练状态和目标 local batch 都能放入单卡：先用 DDP，建立稳定 baseline。
2. 参数能参与单层计算，但参数、梯度和 optimizer state 总和造成显存瓶颈：使用 FSDP2；检查 AllGather/ReduceScatter 代价，并比较 root-wrap 与 layer-wrap。
3. 单层参数或 activation 已无法放入单卡，或数据并行使 local batch 低到不可接受：使用 TP；优先在高速机内互联上切 attention heads 和 MLP hidden dimension。
4. 同时存在训练状态和单层计算瓶颈，且 GPU 数量足够：才考虑 TP×DP/FSDP 的 2D mesh。它是扩展项，不阻塞 07 v1.0。

单卡 OOM 对照必须由模型配置和显存容量自然触发，不能通过无关进程抢占显存制造。本轮若没有合适模型规模，应如实记录 `unavailable`，保留显存曲线与理论转折点，不强造 OOM。

## 实现边界

07 不实现生产级自动并行搜索、完整 Megatron/TorchTitan、多节点弹性恢复、pipeline parallel 或任意 Transformer 自动改写。框架只提供 primitive；交付重点仍是最小实现、测量、瓶颈解释和一次可复现的策略比较。

逐项状态见 [07 验收清单](../docs/07-issues.md)，实验设计与结果解释见 [实验记录](experiments/README.md)。
