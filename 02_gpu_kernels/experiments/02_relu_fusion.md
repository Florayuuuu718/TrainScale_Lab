# 实验 02：Add + ReLU——融合为什么可能更快

> 状态：PyTorch eager 与 fused Triton 的 forward、backward correctness、性能和 Profiler 对照已完成。

## 为什么做

训练图里常见 `y = relu(x + addend)`。不融合时，add 先生成中间 tensor，ReLU 再读它；融合后，一个 kernel 可以读两份输入并直接写 y。这个实验用最简单的组合解释“少一次 launch”和“少一次中间结果往返”。本轮两份输入形状相同，不测试 broadcast bias。

## 小白名词

- **ReLU**：`max(x, 0)`，负数变 0，正数保留。
- **fusion**：把多个逻辑算子合成一次 GPU kernel。
- **中间 tensor**：前一个算子的输出、后一个算子的输入。
- **forward/backward**：前向算输出；反向算梯度。

## 一般预期

tiny case 可能主要省一次 launch；large case 还可能省掉中间 tensor 的写入和再次读取。但 PyTorch compile 本身也可能完成融合，所以手写 Triton 不保证更快。

## 跟着做：运行 Add + ReLU 融合

阅读 [`triton_ops.py`](../trainscale_kernels/triton_ops.py) 中的
`_relu_add_kernel`、`_relu_add_backward_kernel`、`relu_add` 和
`relu_add_backward`。PyTorch reference 是 `torch.relu(x + bias)`；Triton 版本
在同一个 kernel 中完成加法和 ReLU，不保存单独的 `x + bias` 中间张量。

```bash
TRAINSCALE_RUN_SM120_TRITON=1 PYTHONPATH=02_gpu_kernels \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  02_gpu_kernels/tests/test_triton_ops.py -k relu_add

.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator relu_add --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/02_relu_add.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/02_relu_add.json
```

pytest 先证明 forward 和 backward 与 PyTorch 对齐；benchmark 再比较 forward。
终端应出现两个 shape、四条成功路径和 `all_cases_passed=True`。小 shape 很可能
因为 launch 固定成本没有加速，大 shape 才更容易体现少一次中间张量读写；不要
为了得到“更漂亮”的结果删掉小 shape。正式复现使用 `21/10` samples/warm-up。

## 实际结果

| 元素数 | PyTorch eager | fused Triton | Triton 相对 PyTorch |
|---:|---:|---:|---:|
| 257 | 17.801 µs | 16.984 µs | 1.048× |
| 1,048,576 | 23.295 µs | 19.752 µs | 1.179× |

forward 最大绝对误差为 0；509 元素的独立测试还验证了 `dx` 与第二输入梯度。Profiler 中，eager 的 20 次调用出现 add kernel 约 115.018 µs、ReLU kernel 约 99.985 µs；Triton 是一个 fused kernel，20 次约 219.892 µs。Profiler 行可能嵌套，不能把所有行随意相加成 wall time。

## 理论分析

融合后的理论最小流量是读 x、读 addend、写 y；未融合还要写并重读 add 中间结果，所以理论上少搬约 `2 × N × element_size` 字节，并少一次 kernel launch。large case 的 1.179× 加速与这个机制方向一致；tiny case 只有 1.048×，说明固定开销和不同 kernel 的单次效率会稀释收益。

Profiler 中 Triton 单个 kernel 的 device time 并没有神奇地小于两个 eager kernel 聚合量级。逻辑延迟仍更好，说明融合价值还来自少一次框架调度、少一个中间 tensor 分配和更短的依赖链，而不只是 kernel 内算术。

Backward 在 `x+b=0` 处数学上不可导，实现必须与选定 reference 的约定一致；PyTorch 通常取 0。这也是 correctness test 必须覆盖的边界。

## 结论与收尾

融合 Triton 在本机两个 forward case 均略快，large case 快约 17.9%，且 forward/backward 数值对齐。但这是“同形状两输入 + ReLU”的有限结论，不自动推广到 broadcast、其他 dtype 或复杂算子链。
