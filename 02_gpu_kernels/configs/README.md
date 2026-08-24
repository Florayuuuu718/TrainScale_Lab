# 02 配置与矩阵

配置的作用是冻结输入域和测量方法，不把 shape 列表散落在脚本中。所有文件由 CPU-only `benchmark_contract.py` 严格解析：未知字段、重复 ID、非法 dtype/layout、非正数 shape 和无效 warp 数都会在启动 GPU 前报错。

## 已实现配置

| 文件 | 用途 |
|---|---|
| `correctness.toml` | 每个算子的 shape、dtype、layout、数值分布和容差 |
| `benchmark_smoke.toml` | 快速验证入口、JSON schema 和少量 GPU case |
| `benchmark_full.toml` | 正式 shape sweep、warm-up、重复次数和实现列表 |
| `profiler.toml` | 固定的 memory-bound / compute-bound 代表 case |
| `cuda_comparison.toml` | PyTorch/Triton/CUDA 的 Vector Add 与 Softmax 公平输入域 |
| `layer_norm_training.toml` | hidden、FP32/FP16、eps、forward/backward 扫描 |
| `matmul_autotune_cases.toml` | 512³ 与 ragged 509³ 两个选型 case |
| `matmul_candidates.toml` | 四组 block M/N/K、group M 和 num warps 候选 |

## Shape 分层

| 类别 | 目的 | 示例 |
|---|---|---|
| tiny | 暴露 launch 固定成本 | element count 17/257；小 M/N/K |
| aligned | 验证理想 tile | 128、256、1024 等 |
| ragged | 验证 mask 和尾块 | 127、1000、4097 等 |
| prime | 防止只适配整齐 shape | 257、509 等 |
| realistic | 对应训练中的 hidden/head/sequence | 由实验文档冻结 |
| stress | 显存、索引和实现上限 | 单独运行，允许 OOM/unsupported |

这些类别已经映射到实际 TOML，但每个算子仍按自己的数学维度选择 shape，不能盲目复用同一列表。例如 509 对 Softmax 是 row width，对 MatMul 则同时用于 M/N/K 尾块。

## Dtype 与容差

- FP32 是所有主线算子的必测类型；
- FP16/BF16 仅在硬件和实现支持时启用；
- reduction/matmul/attention 必须明确 accumulation dtype；
- `atol`、`rtol` 按算子和 dtype 固定在配置中；
- 任何容差修改必须在实验报告中解释，不能由测试失败反向调参。

## Case 标识

每个 case 使用稳定 ID，至少编码：

`operator / implementation / shape / dtype / layout / mode`

原始 JSON 必须同时保存展开后的完整字段，不能要求读者从文件名猜配置。

CPU 验证命令：

```bash
.venv/bin/python -m pytest -q \
  02_gpu_kernels/tests/test_benchmark_contract.py
```

## 不改 Python，直接选择练习范围

学习者可以用统一 runner 的 `--operator` 选择一个算子的全部 case，或用可重复的
`--case` 选择精确 case；二者不能混用。例如：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/run_triton_comparison.py \
  --suite full --operator attention --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/attention.json
```

这条命令直接读取上面的 TOML，不需要为了换 shape 修改 benchmark 源码。完整
逐实验命令见[实验索引](../experiments/README.md)。
