# 02 · GPU Kernels 验收清单

以下条目可直接转成 GitHub Issues。状态只有“未开始、进行中、已完成、阻塞”；没有可复现证据时不能标记完成。03 必须等 02-01..12 全部关闭后再开始。

| ID | 标题 | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|
| 02-01 | 冻结 02 范围与环境契约 | 01 完成 | README 明确主线/进阶边界；记录 Linux/WSL、GPU、driver、PyTorch/CUDA/Triton；CUDA C++ 前验证 `nvcc` | 已完成 |
| 02-02 | 建立 correctness matrix 与 benchmark harness | 02-01 | 8 份 TOML 可复用；冷启动/稳态分离；CUDA Event 同步；统一 JSON/schema CPU 测试通过 | 已完成 |
| 02-03 | Vector Add：PyTorch/Triton/CUDA C++ | 02-02 | ragged/prime、FP32/FP16/BF16、Triton/CUDA scalar/packed、latency/带宽均已归档 | 已完成 |
| 02-04 | ReLU 与逐元素融合 | 02-03 | forward/backward 对齐 PyTorch；比较独立 add+ReLU 与 fused kernel；解释 launch 和中间张量流量 | 已完成 |
| 02-05 | Stable Softmax：Triton baseline/optimized | 02-02 | stable max-subtraction；17–4097 ragged 行宽；single-warp baseline/warp-policy 候选和失败反例已归档 | 已完成 |
| 02-06 | Softmax CUDA C++ 深挖 | 02-03, 02-05 | serial baseline 与 shared-memory block reduction 已实现；五行宽 correctness/性能/理论解释完整 | 已完成 |
| 02-07 | LayerNorm forward/backward | 02-05 | y/mean/rstd/dx/dweight/dbias 对齐；hidden/dtype/eps 与 forward/backward 8 组对照已归档 | 已完成 |
| 02-08 | Blocked MatMul 与 autotune | 02-02 | ragged forward/backward、FP32 reference、四候选 tile/warp 排名与两 shape 选择已归档 | 已完成 |
| 02-09 | Fused Attention forward | 02-05, 02-08 | causal/non-causal、head dim 16/32/64/128、sequence>256 与非法 head dim 边界已测试 | 已完成 |
| 02-10 | 跨算子 profiler 与 roofline 分析 | 02-04, 02-06, 02-07, 02-08, 02-09 | Vector Add/ReLU/MatMul/Attention 已保存 PyTorch/Triton profiler 证据，并与字节/FLOPs 理论对照 | 已完成 |
| 02-11 | 测试、CI 分层与结果归档 | 02-03..10 | CPU reference/config/schema/result tests 已进入 CI；15 项 GPU tests 独立 opt-in；正式 JSON/ignored raw 边界明确 | 已完成 |
| 02-12 | 02 发布验收 | 02-01..11 | 新环境 smoke run；ruff/mypy/pytest；六算子报告、SHA-256 汇总、失败记录和复现命令 | 已完成 |

最终证据见 [`module02_acceptance_sm120.json`](../02_gpu_kernels/results/module02_acceptance_sm120.json)：Windows 侧 `ruff`、`mypy`、`26 passed, 1 skipped`，WSL 侧 Toolkit 实编译/运行探针、12-path config smoke、按算子教程命令与 `15 passed` GPU 测试均通过。性能结论只适用于记录的单机环境。

## 进阶项，不阻塞 02 完成

- Attention backward 或完整 FlashAttention 教学实现；
- persistent kernel、warp specialization、Tensor Memory Accelerator；
- 多 GPU kernel 和跨设备算子；
- 更多 CUDA C++ 算子或与 CUTLASS/cuBLASLt 深度集成；
- 自动性能回归报警和多 GPU 型号基线。

进阶项只有在主线结果显示明确问题时再建立 issue，不能反向扩大 02-01..12 的完成定义。
