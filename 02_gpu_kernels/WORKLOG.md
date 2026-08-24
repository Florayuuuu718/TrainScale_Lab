# 02 开发断点记录

> 原断点时间：2026-08-24 02:13（Asia/Shanghai）。本文件保留当时未完成的开发现场，不能当作正式实验结果。

> 续跑状态：2026-08-24 已从本断点恢复。四方/五方 CUDA-Triton 对照、LayerNorm training、MatMul autotune 与汇总 JSON 均已正式完成；本文件以下内容保留为开发过程记录，当前状态以模块 README 和结果汇总为准。

> 封存状态：最终 Windows 与 WSL 回归结果以
> [`module02_acceptance_sm120.json`](results/module02_acceptance_sm120.json) 为准。

## 已经正式验证并归档的基础状态

- Windows driver 610.88；RTX 5060 Laptop，SM 12.0；
- stable `torch 2.12.1+cu129 / Triton 3.7.1` 环境探针全部通过；
- 原 02 Triton GPU tests：`13 passed`；
- 原 14 组 PyTorch/Triton forward comparison 全部通过 correctness gate；
- CUDA 13.0 standalone smoke 在 Ubuntu 26.04 上使用
  `-U_GNU_SOURCE -D_DEFAULT_SOURCE` 编译、运行通过。

以上正式证据仍在 `results/` 中，未被当天的未完成实验覆盖。

## 当天新增、已经单独验证的内容

1. 新增 CPU-only `benchmark_contract.py`，把 case、suite、percentile 和结果 schema
   从脚本硬编码中拆出；
2. 新增 `benchmark_full.toml`、`benchmark_smoke.toml`、`correctness.toml`、
   `profiler.toml` 和 `cuda_comparison.toml`；
3. 原 `run_triton_comparison.py` 已改为读取 TOML；
4. 当时新增 5 项 CPU 测试，独立运行结果为 `5 passed`；
5. 新增正式 CUDA C++ `kernel_bench.cu` 和构建脚本；
6. CUDA 程序成功为 `sm_120` 编译；以下独立 smoke 均通过 correctness：
   - Vector Add baseline FP32，N=4097；
   - Vector Add optimized FP32/FP16/BF16，N=4097；
   - Softmax serial baseline FP32，8×509；
   - Softmax block reduction optimized FP32，8×509。
7. 8×509 Softmax 的 5-sample 快速结果中，serial baseline median 约
   25.17 µs，block reduction median 约 5.76 µs；这只是开发 smoke，不能进入
   正式报告。

## 当时的中断位置

新增 `run_cuda_triton_comparison.py` 后，执行“13 项 Triton tests + 9 组四方
comparison 的 5-sample smoke”时，命令在运行约 26.5 秒后被用户主动中断。

- 中断后检查确认没有残留的 pytest、CUDA benchmark 或 comparison 进程；
- `/tmp/cuda-triton-smoke.json` 即使存在也不得视为完整结果；
- 当时最后一次 Ruff/py_compile 和 `out=` GPU 回归还未执行；
- 此后的正式续跑已经完成这些检查，不能用本段历史状态覆盖最终验收。

## 当时记录的续跑顺序

1. Windows 先运行 Ruff、py_compile、CPU tests；
2. 把最新 `02_gpu_kernels/` 同步到 `/home/...-sm120-validation`；
3. 重新构建 `/tmp/trainscale-kernel-bench`；
4. 单独运行 Triton tests，确认 `out=` 改动没有回归；
5. 先用 3 samples 跑四方 runner；若失败，只修具体 case；
6. 通过后以 10 warm-up、21 samples 运行正式 CUDA/PyTorch/Triton 对照并归档；
7. 继续实现 LayerNorm backward benchmark 与 MatMul tile/autotune 候选记录；
8. 最后更新九份实验报告、验收清单和发布检查，不能提前把 02 标为完成。

上述八步现已全部完成。本文件保留它们，是为了让后来者理解正式结果并非一次
命令自然产生，同时也不应把开发 smoke 和中断现场当成正式性能证据。

## 工作区边界

开发期所有改动均先保存在工作区。旧的历史失败 JSON 必须保留，开发 smoke
数字不得写进正式结果；正式性能必须在 WSL `/home/...` checkout 运行。
