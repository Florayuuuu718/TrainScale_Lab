# 实验 07：CUDA C++、Triton、PyTorch 能公平比较吗

> 状态：PyTorch、Triton baseline/policy、CUDA C++ baseline/optimized 的统一 kernel-only 对照已完成。

## 为什么做

同一个算子可以用成熟 PyTorch 库、Triton DSL 或 CUDA C++ 编写。这个实验不是预设谁更快，而是比较：

- 能控制多少底层细节；
- 需要多少构建和调试工作；
- 首次编译与稳态速度；
- 错误边界和可维护性。

## 小白名词

- **CUDA C++**：NVIDIA 的底层 GPU 编程语言和工具链。
- **Triton**：用 Python 风格 DSL 描述 GPU block 计算，由编译器生成设备代码。
- **nvcc**：把 CUDA C++ 源码编译为 GPU/主机代码的编译器。
- **ptxas**：把 PTX 中间表示汇编成针对具体 GPU 的机器代码。
- **DSL**：为特定领域设计的语言。

## 一般预期

Vector Add 主要受 launch/带宽限制，三者稳态差距可能不大；Softmax 更能体现 reduction 和片上资源控制。Triton 通常减少样板代码，CUDA C++ 提供更细控制，PyTorch 库则拥有成熟的跨 shape 调优。

## 跟着做：编译 CUDA C++，再统一比较

CUDA 源码是 [`kernel_bench.cu`](../cuda/kernel_bench.cu)，构建入口是
[`build_cuda_bench.py`](../cuda/build_cuda_bench.py)，统一 runner 是
[`run_cuda_triton_comparison.py`](../benchmarks/run_cuda_triton_comparison.py)。
先确认已经按[环境教程](../ENVIRONMENT.md#7-cuda-ctoolkit-是单独的能力门)安装
Toolkit，且 `nvcc --list-gpu-code` 包含 `sm_120`。

Ubuntu 24.04 推荐路线：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py --require-nvcc

.venv/bin/python 02_gpu_kernels/cuda/build_cuda_bench.py \
  --output /tmp/trainscale-kernel-bench

.venv/bin/python 02_gpu_kernels/benchmarks/run_cuda_triton_comparison.py \
  --cuda-binary /tmp/trainscale-kernel-bench \
  --cuda-toolkit 13.0.88 --cuda-build-flag=-arch=sm_120 \
  --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/07_cuda_compare.json

.venv/bin/python 02_gpu_kernels/benchmarks/show_results.py \
  02_gpu_kernels/results/raw/tutorial/07_cuda_compare.json
```

若你已经使用 Ubuntu 26.04，本机验证过的完整替代命令是：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py \
  --require-nvcc \
  --nvcc-flag=-U_GNU_SOURCE \
  --nvcc-flag=-D_DEFAULT_SOURCE

.venv/bin/python 02_gpu_kernels/cuda/build_cuda_bench.py \
  --output /tmp/trainscale-kernel-bench \
  --nvcc-flag=-U_GNU_SOURCE \
  --nvcc-flag=-D_DEFAULT_SOURCE

.venv/bin/python 02_gpu_kernels/benchmarks/run_cuda_triton_comparison.py \
  --cuda-binary /tmp/trainscale-kernel-bench \
  --cuda-toolkit 13.0.88 \
  --cuda-build-flag=-arch=sm_120 \
  --cuda-build-flag=-U_GNU_SOURCE \
  --cuda-build-flag=-D_DEFAULT_SOURCE \
  --samples 5 --warmup 2 \
  --output 02_gpu_kernels/results/raw/tutorial/07_cuda_compare.json
```

这些 feature-macro flags 是 Ubuntu 26.04/glibc 2.43 的已验证 workaround，不能
据此宣称该发行版进入 CUDA 13.0 官方支持范围。更多解释见
[`cuda/README.md`](../cuda/README.md)。

runner 应打印 Vector Add 的四种路径和 Softmax 的五种路径，共 41 条
`success`，最后为 `all_cases_passed=True`。二进制执行、正确性、CUDA Event
计时均在 runner 内完成；不能只编译成功就声称实验完成。正式复现使用 `21/10`。

## 实际结果

| 路线 | 实际状态 | 证据 |
|---|---|---|
| PyTorch 2.12.1+cu129 | 通过 | 14 个正式 forward case 与代表性 Profiler 已归档 |
| Triton 3.7.1 | 通过 | driver 610.88 下探针、15 项最终测试、正式性能 case 全通过 |
| CUDA Toolkit 13.0 | 通过 | `sm_120` Vector Add scalar/packed 和 Softmax serial/block 全部正确 |
| 统一对照 | 通过 | 9 case、41 条适用路径，10 warm-up、21 samples、全部 correctness passed |

为保证计时边界一致，全部路径预分配输出并排除 device allocation/host copy。Vector Add 最大 case 中 CUDA `float4` 为 11.841 µs；Softmax 4097 列中 PyTorch 9.759 µs、CUDA block 10.834 µs、Triton single warp 16.896 µs、CUDA serial 180.996 µs。完整矩阵分散在实验 01/03，原始数据在结果 JSON。

## 理论解释

PyTorch CUDA wheel 携带运行训练所需的 CUDA runtime 和库，所以 eager 可运行；编译自定义 `.cu` 文件还需要系统 Toolkit/`nvcc`。Triton 自带 `ptxas`，因此“没有 nvcc”不是 Triton 崩溃的原因。

本机旧 driver 577.05 下的 Triton 崩溃在升级到 610.88 后消失，默认 cu129 环境无需替换。系统 Toolkit 与 PyTorch wheel runtime 是两层：只做 Triton 不要求 `nvcc`，编译 CUDA C++ 才需要 Toolkit。Ubuntu 26.04/glibc 2.43 上，CUDA 13.0 头文件默认会出现 `rsqrt/rsqrtf` 声明冲突；本机以 `-U_GNU_SOURCE -D_DEFAULT_SOURCE` 完成 smoke，但新安装仍推荐官方支持的 Ubuntu 24.04 组合。

## 结论与收尾

三条工具链已经在同一输入公式、shape/dtype 和 kernel-only 计时口径下完成比较。没有一种语言在全部 case 获胜：tiny/large、dtype、reduction 组织和成熟库选型共同决定结果。默认根 `.venv` 继续负责 PyTorch/Triton；Toolkit 只为 standalone CUDA 程序服务，避免无必要的 wheel/extension ABI 混装。
