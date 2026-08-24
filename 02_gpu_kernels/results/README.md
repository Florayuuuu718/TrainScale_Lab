# 02 结果契约

结果分为可提交摘要和本地大型原始产物。

## 当前结构

~~~text
results/
├── eager_baselines.json                 # driver 577.05 历史 PyTorch 基线/失败现场
├── profiler_summary.json                # driver 577.05 历史 Profiler 摘要
├── sm120_environment_validation.json    # driver 610.88 环境门与两套 Python 环境验证
├── triton_comparison_sm120_cu129.json   # 14 组 PyTorch/Triton 正确性与计时
├── triton_profiler_sm120_cu129.json     # 8 组代表性 Profiler 聚合行
├── cuda_triton_comparison_sm120_cu129_cuda130.json # 9 case、41 条 kernel-only 记录
├── layer_norm_training_sm120_cu129.json # 4 case × forward/backward × 两种实现
├── matmul_autotune_sm120_cu129.json     # 2 shape × 4 Triton 候选 + PyTorch
├── module02_summary_sm120.json          # 正式结果总门、赢家和源文件 SHA-256
├── module02_acceptance_sm120.json       # Windows/WSL 最终发布验收与已知边界
└── raw/                     # Nsight、trace、逐样本数据、编译缓存
~~~

`raw/` 应进入 Git ignore；小型 JSON/SVG 只有在能追溯到报告、配置和 commit 时才提交。

当前没有单独复制一份 `correctness.json`：每个 case 的状态、容差和最大误差已与性能样本一起保存在 comparison JSON 中，环境测试摘要保存在 validation JSON 中。历史失败与当前成功结果分开命名，失败不能用 0 或空表伪装。

## 当前证据边界

- 默认环境最终 GPU correctness 为 15 passed；早期 stable/nightly 13 项环境快照保留；
- 14 组 forward 性能对照全部通过 correctness gate；
- Vector Add/Softmax 的 9 个 case、41 条 PyTorch/Triton/CUDA 路径全部通过；
- LayerNorm 8 个 phase comparison 与 MatMul 8 个 Triton candidate run 全部通过；
- Profiler 覆盖 Vector Add、融合 ReLU、MatMul 与 Attention；
- 发布验收记录把 Windows 静态/CPU 检查与 WSL 真实 GPU/Toolkit 检查分开保存；
- `raw/` trace 可在本地重建，不提交大文件。

## 每条 benchmark 记录至少包含

- schema version、timestamp、git commit；
- OS、Python、PyTorch、CUDA runtime、Triton、driver；
- GPU 名称、compute capability、显存；
- operator、implementation、variant；
- shape、dtype、layout、mode；
- seed、warm-up、repeats、sample count；
- compile latency、median/p10/p90 latency；
- 适用时的 bytes、FLOPs、GB/s、TFLOPS、peak memory；
- correctness status、tolerances、max absolute/relative error；
- status：success/unsupported/OOM/compile_error/runtime_error；
- error 摘要与原始产物路径。

失败记录只保存必要摘要，避免把包含本机绝对路径的大型日志直接提交。
