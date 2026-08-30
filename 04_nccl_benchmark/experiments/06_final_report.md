# 04 最终实验报告：从通信曲线解释 DDP

## 先学什么

这一章不是为了记住一个“4090D 带宽数字”，而是学习三个可迁移的方法：

1. **延迟区与带宽区不同。** 小消息主要支付固定启动延迟；消息足够大后才接近链路平台带宽。
2. **`algbw` 与 `busbw` 回答的问题不同。** 前者是有效载荷速率，后者按 collective 的通信量模型折算链路负载；跨 collective 比较时必须先统一口径。
3. **通信基准必须回到训练负载。** DDP 传输的是梯度 bucket，不是任意大小的字节，因此要把模型真实 payload 映射到曲线，再看 timeline。

一般规律是：通信时间可粗略写成 `T ≈ α × rounds + bytes / bandwidth`。`α` 是每轮固定开销；小消息受 `α` 支配，大消息受带宽支配。这是分析起点，不是对任何机器的硬编码答案。

不熟悉 rank、collective、`busbw` 或 strong/weak scaling 时，先查
[分布式训练与通信术语表](../../docs/concepts/distributed-systems-glossary.md)。

## 运行前预测

- 消息很小时 latency 变化不大，但换算出的 GB/s 很低且容易波动；
- 消息增大后 `busbw` 应逐渐进入平台区；
- 4 GPU 理论通信量更大，但能否得到更高平台值由链路和 NCCL 调度决定；
- 跨 NUMA pair 可能更慢，但在看到重复曲线前不能把拓扑标签写成结论；
- tiny DDP 的 strong scaling 可能因固定同步成本而不升反降。

预测不是答案。实验的任务正是指出哪些成立、哪些需要缩小措辞。

## 我们怎样验证

环境为单机 4×RTX 4090 D，GPU 2–3 为 PHB 路径，GPU 0–2 为跨 NUMA 的 SYS 路径；没有 NVLink，NCCL 实际使用共享内存/PCIe 路径。正式实验包含四种 collective、2/4 GPU、从小消息到 64 MiB 以上的扫描、三次重复、DDP payload 映射和 profiler trace。

03 的 MLP 有 2,623,744 个参数，FP32 梯度载荷为 10,494,976 bytes，约 10.01 MiB。因此 10 MiB 和默认 25 MiB bucket 是本章最重要的两个读点。

## 结果与解释

### 1. 大消息平台区

2 GPU AllReduce 在 64 MiB 以上的平台区：

| GPU pair | 拓扑 | 平台 `busbw` |
|---|---|---:|
| 2–3 | PHB | 7.81 GB/s |
| 0–2 | SYS | 7.85 GB/s |

近端反而低约 0.51%。这不是“SYS 更快”的证据，而是说明在这台无 P2P/NVLink、由 NCCL 选择共享内存路径的机器上，拓扑标签没有形成超出噪声的稳定差异。

### 2. 训练 payload 所在位置

| 消息大小 | 2 GPU `busbw` | 4 GPU `busbw` |
|---:|---:|---:|
| 10,494,976 bytes | 7.61 GB/s | 9.45 GB/s |
| 25 MiB | 7.74 GB/s | 9.55 GB/s |

10 MiB 已靠近大消息平台区。因此该 DDP workload 的问题不是“消息太小，完全没用到带宽”，而是模型计算很短、同步和进程调度固定成本占比太高。

### 3. DDP scaling 补测

五次长窗口实验 correctness 全部通过，但 measurement quality 仍未过 5% 稳定性门：

| 模式 | 1 GPU | 2 GPU | 4 GPU | 应怎样读 |
|---|---:|---:|---:|---|
| strong，中位吞吐 | 143,537 | 135,001 | 135,345 samples/s | 固定 global batch，通信开销大于并行收益 |
| weak，中位吞吐 | 74,572 | 138,117 | 263,883 samples/s | 4 GPU 为 3.54×，定性扩展良好 |
| 相对极差 | 8.61% | 11.76% / 3.00% | 12.55% / 9.38% | 不能包装成高精度效率 |

strong scaling 失败并不否定 NCCL；它照应 Amdahl 定律：可并行计算部分太小，固定同步成本会吞掉收益。weak scaling 每张卡保留相似工作量，更容易摊薄固定成本。

## 延伸：拓扑与 NCCL 策略到底有没有影响

延伸实验做了 36 个可校验作业。PHB/NODE/SYS 的大消息平台分别为 7.84/7.85/7.89 GB/s，最大差异 0.64%，仍无稳定排序。10 MiB 和 25 MiB 处 SYS 比 PHB 低约 2.06% 和 1.66%，说明拓扑影响可能依赖消息大小，但证据不足以推广成通则。

固定 4 GPU 后比较 NCCL 策略：

| 策略 | 大消息平台 | 相对 Auto |
|---|---:|---:|
| Auto | 9.58 GB/s | baseline |
| Ring + Simple | 9.56 GB/s | -0.21% |
| Tree + Simple | 8.29 GB/s | -13.47% |
| Ring + LL | 4.78 GB/s | -50.10% |

LL 在个别 4–32 KiB 点可能更快，但重复波动很大；大消息明显更慢。一般结论不是“永远用 Ring + Simple”，而是：**NCCL 自动调优是可靠默认值，强制算法/协议主要用于诊断和理解机制。**

### 怎样复现这个延伸

下面只改变 GPU pair 或 NCCL policy；其余参数保持一致。先确认 `NCCL_TEST_BIN` 指向
`all_reduce_perf`。三组 pair 必须根据你自己的 `topology.txt` 修改：

```bash
export NCCL_TEST_BIN=/root/nccl-tests-v2.19.7-src/build/all_reduce_perf
export EXT_DIR="$RUN_ROOT/module04/nccl-explainability"
mkdir -p "$EXT_DIR"

# 参考主机上的 PHB/NODE/SYS pair；不能直接复制标签到另一台机器。
for item in phb:2,3 node:1,2 sys:0,2; do
  name=${item%%:*}
  pair=${item#*:}
  for run in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="$pair" "$NCCL_TEST_BIN" \
      -b 8 -e 268435456 -f 2 -g 2 -w 5 -n 20 -d float \
      > "$EXT_DIR/topology-${name}-run${run}.log" 2>&1
    echo $? > "$EXT_DIR/topology-${name}-run${run}.exit"
  done
done

# Auto 基线。
for run in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$NCCL_TEST_BIN" \
    -b 8 -e 268435456 -f 2 -g 4 -w 5 -n 20 -d float \
    > "$EXT_DIR/policy-auto-run${run}.log" 2>&1
  echo $? > "$EXT_DIR/policy-auto-run${run}.exit"
done

# 强制策略仅用于诊断。
for item in ring-simple:Ring:Simple tree-simple:Tree:Simple ring-ll:Ring:LL; do
  name=${item%%:*}; rest=${item#*:}; algo=${rest%%:*}; proto=${rest#*:}
  for run in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_ALGO="$algo" NCCL_PROTO="$proto" \
      "$NCCL_TEST_BIN" -b 8 -e 268435456 -f 2 -g 4 -w 5 -n 20 -d float \
      > "$EXT_DIR/policy-${name}-run${run}.log" 2>&1
    echo $? > "$EXT_DIR/policy-${name}-run${run}.exit"
  done
done

grep -H -E "Out of bounds|Avg bus bandwidth" "$EXT_DIR"/*.log
unset NCCL_ALGO NCCL_PROTO
```

强制 policy 失败或明显变慢也是有效诊断结果，不能只保留最快组合。

## 学完后应能回答

- 为什么小消息延迟不能用 GB/s 排名？
- 为什么 `nvidia-smi topo -m` 的标签本身不能证明性能差异？
- 为什么 NCCL 曲线很好，tiny DDP strong scaling 仍可能变慢？
- 为什么一次最快值不能代替重复实验中位数和波动范围？

机器可读摘要见 [`../results/module04_final_summary.json`](../results/module04_final_summary.json)。原始证据保存在校验归档中，不进入 Git；归档与内部文件哈希均已验证。
