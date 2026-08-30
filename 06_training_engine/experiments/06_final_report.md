# 06 最终实验报告：Bucket、异步通信与真实 Overlap

## 先学什么

训练引擎里的 reducer 要解决两个问题：梯度何时可通信，以及通信何时必须完成。

- **bulk**：backward 全结束后一次同步；简单、collective 少，但没有重叠机会。
- **per-parameter**：每个梯度 ready 就同步；最早启动，但产生大量小 collective。
- **bucket sync**：把参数合并成少量 bucket，降低启动次数，但每次阻塞等待。
- **bucket async**：bucket ready 后异步发起，在 optimizer step 前统一等待；它创造 overlap 的可能。
- **DDP**：成熟 reference，除 bucket 外还处理 hook 顺序、unused 参数、调度和多种后端细节。

关键概念是：`async_op=True` 或“提前 launch”只表示候选重叠。只有 timeline 中 NCCL kernel 与 backward compute kernel 的时间区间相交，才能宣称真实 overlap。

术语解释见 [分布式训练与通信术语表](../../docs/concepts/distributed-systems-glossary.md)。

## 运行前预测

- per-parameter 最早通信，但小 collective 太多，可能比 bulk 更慢；
- bucket async 应比 bucket sync 更有重叠机会；
- bucket 太大时启动过晚，太小时 launch 次数过多，因此不存在跨模型通用的最佳大小；
- AMP 只有在更快的低精度计算足以摊薄 cast 和 scale 开销时才会加速。

## 正确性与状态机

五种策略 × accumulation 1/2 的 10 个 Gloo case 全部通过；最大梯度误差 `1.788e-7`，最大参数更新误差 `1.49e-8`。测试还包含未参与 forward 的参数、异步 handle 在 step 前完成和一致 bucket plan。

AMP overflow 的 2/4 GPU、bucket-async/DDP 四个 case 也全部通过：正常 step 更新；注入 overflow 后 optimizer step 被跳过，scale 从 65536 降到 32768，参数保持不变。它验证的是混合精度状态机，不只是“程序没有报错”。

## Reducer 消融

4 GPU、medium、FP32、accumulation=1：

| 策略 | 吞吐 | 相对观察 |
|---|---:|---|
| bulk | 6,835 samples/s | 本模型的少量大通信很合适 |
| per-parameter | 5,171 samples/s | 启动次数过多 |
| bucket sync | 5,675 samples/s | 分桶但阻塞等待 |
| bucket async | 5,881 samples/s | 比 sync 高 3.64% |
| DDP | 6,752 samples/s | 成熟调度接近 bulk |

这组结果没有证明 bulk 普遍优于 overlap；它说明当前 Tiny Transformer 的 backward 太短，通信粒度和启动开销比隐藏通信更重要。

AMP 在该 tiny workload 中比 FP32 慢约 13.8%–14.1%。Tensor Core 的潜在收益没有摊薄 scale/unscale、cast 和小 kernel 开销。这和一般规律一致：AMP 是能力，不是无条件加速开关；模型规模和算术强度决定是否获益。

## 延伸实验：1 MiB Bucket

默认约 10 MiB bucket 时，5 个 measured steps 每 rank 只有 5 个 NCCL kernel，手写 sync/async 都没有观测到真实 GPU kernel overlap。将 bucket 调到 1 MiB 后：

| 指标 | 10 MiB | 1 MiB |
|---|---:|---:|
| 每 rank NCCL kernels / 5 steps | 5 | 25 |
| bucket-async 中位 overlap | 0% | 2.676% |
| bucket-async 吞吐 | 5,881 | 5,451 samples/s |

1 MiB 的 async trace 中，NCCL 与 attention backward、SGEMM 等 kernel 确实相交；sync 仍为 0%。但吞吐反而下降 7.31%，因为 collective 数量增加 5 倍，launch/synchronization 开销超过隐藏掉的通信时间。

这是本章最重要的结论：**overlap 是实现机制，step time 才是优化目标。** Profiler 证明“发生了什么”，benchmark 判断“值不值得”。

复现延伸实验：

```bash
python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --bucket-cap-mb 1.0 \
  --raw-directory "$RUN_ROOT/module06/extension-1m/profile-raw" \
  --output "$RUN_ROOT/module06/extension-1m/overlap-profile-1m.json"
```

`profile-raw/` 是 Chrome trace 与 rank JSON 的目录，不是单个下载文件。它只在你要用 Perfetto/Chrome trace 深挖时间线时需要；常规学习和结果核验保留汇总 JSON 与压缩归档即可。

机器可读摘要见 [`../results/module06_final_summary.json`](../results/module06_final_summary.json)。
