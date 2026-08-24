# 02 测试计划

02 将测试分成 CPU 快速测试、GPU correctness 和性能实验三层。性能波动不能污染 correctness CI。

## CPU 快速测试

- reference 数学定义、shape 推导和配置校验；
- benchmark 结果 schema、指标公式和聚合逻辑；
- unsupported case 的分类与序列化；
- 不导入 CUDA extension 也能完成 collection；
- 适合进入普通 PR CI。

## GPU correctness

每个实现至少覆盖：

- aligned、ragged、prime 和 tiny shape；
- 声明支持的 dtype/layout；
- deterministic seed 与相同输入；
- NaN/Inf 策略和大幅值输入；
- 非整 tile 的 mask；
- 不支持输入显式报错；
- 要求 backward 的算子对照 PyTorch gradients；
- benchmark 前自动运行对应 correctness gate。

GPU correctness 可以在本地或自托管 GPU runner 运行，不以性能阈值判断通过。

当前 `test_triton_ops.py` 覆盖六类算子的 ragged shape、预分配输出、Triton Softmax baseline/optimized、forward/backward、四组 MatMul tile、Attention 的四种 head dim 和错误边界。因为历史错误是进程级段错误，SM 12.0 默认保护性 skip；先让环境探针通过，再显式开启：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py
TRAINSCALE_RUN_SM120_TRITON=1 \
  .venv/bin/python -m pytest -q 02_gpu_kernels/tests/test_triton_ops.py
```

本机最终 stable 结果为 `15 passed`。早期 stable/nightly 的 13 项快照仍保存在环境 JSON；新增两项来自 Attention head-dim 参数扩展，不代表旧快照造假。nightly 仍只是故障诊断兜底。

CPU CI 当前运行：

- `test_benchmark_contract.py`：TOML、percentile、结果 schema、MatMul 候选；
- `test_cuda_build_command.py`：真实架构和兼容 flags 不被构建脚本丢失；
- `test_result_artifacts.py`：所有正式 JSON 的 correctness 总门、case 数和 SHA-256 汇总。
- `test_show_results.py`：初学者结果表能显示 case、latency 和 speedup。

## 性能回归

性能数据默认属于实验，不属于普通单元测试。只有具备固定 GPU、功耗/时钟策略、足够重复次数和历史分布后，才允许加入宽松的 regression alarm；不得把作者机器的绝对延迟写成跨硬件 CI 门槛。

## 当前测试文件

| 文件 | 覆盖内容 |
|---|---|
| `test_benchmark_contract.py` | CPU 配置、统计与 schema |
| `test_cuda_build_command.py` | CPU 构建命令契约 |
| `test_result_artifacts.py` | CPU 正式结果归档契约 |
| `test_show_results.py` | CPU 初学者结果阅读器 |
| `test_triton_ops.py` | GPU 六算子 correctness、gradient 与边界 |
