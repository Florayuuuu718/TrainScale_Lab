# CUDA C++：从工具链 smoke 到正式算子

`smoke_vector_add.cu` 不是最终性能实现。它只回答一个更早的问题：系统 CUDA
Toolkit 能否为当前 GPU 编译机器代码，并由驱动完成加载、launch、同步和结果拷回。

## 官方支持组合：直接编译

新安装推荐 Ubuntu 24.04 LTS；CUDA 13.0 官方支持表没有列出 Ubuntu 26.04。
在 Ubuntu 仓库根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

nvcc --version
nvcc --list-gpu-code | grep -Fx sm_120
nvcc -std=c++17 -O2 -arch=sm_120 \
  02_gpu_kernels/cuda/smoke_vector_add.cu \
  -o /tmp/trainscale-smoke-vector-add
/tmp/trainscale-smoke-vector-add
```

RTX 5060（SM 12.0）的通过输出应包含 `CUDA C++ smoke passed` 和 `SM 12.0`。
只看到 `nvcc --version` 不算通过，因为它没有证明生成的 kernel 能在 GPU 上运行。

也可以让统一环境探针执行同一编译和运行门：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py \
  --require-nvcc
```

## 已有 Ubuntu 26.04 + CUDA 13.0：已验证 workaround

本机 Ubuntu 26.04/glibc 2.43 默认编译会报告 `rsqrt/rsqrtf` exception
specification 不兼容。以下 feature-macro 参数不修改系统 CUDA 头文件，本机已
编译并运行通过：

```bash
nvcc -std=c++17 -O2 -arch=sm_120 \
  -U_GNU_SOURCE -D_DEFAULT_SOURCE \
  02_gpu_kernels/cuda/smoke_vector_add.cu \
  -o /tmp/trainscale-smoke-vector-add
/tmp/trainscale-smoke-vector-add
```

统一探针对应为：

```bash
.venv/bin/python 02_gpu_kernels/benchmarks/check_environment.py \
  --require-nvcc \
  --nvcc-flag=-U_GNU_SOURCE \
  --nvcc-flag=-D_DEFAULT_SOURCE
```

这是对现有机器的 workaround，不是官方支持承诺。希望从零少走弯路，应选择
Ubuntu 24.04 与 Toolkit 13.0 的官方组合。

## 正式 Vector Add 与 Softmax

`kernel_bench.cu` 是正式实验入口，包含：

- Vector Add scalar baseline；
- FP32 `float4`、FP16 `half2`、BF16 `bfloat162` packed 版本；
- 每行单线程串行 reduction 的 Softmax baseline；
- 每行一个 block、shared-memory 并行 max/sum reduction 的 Softmax optimized；
- CPU FP32 reference、ragged tail、correctness gate、cold start、21-sample CUDA Event 计时和 JSON 输出。

构建：

```bash
.venv/bin/python 02_gpu_kernels/cuda/build_cuda_bench.py \
  --output /tmp/trainscale-kernel-bench
```

Ubuntu 26.04 本机命令为：

```bash
.venv/bin/python 02_gpu_kernels/cuda/build_cuda_bench.py \
  --output /tmp/trainscale-kernel-bench \
  --nvcc-flag=-U_GNU_SOURCE \
  --nvcc-flag=-D_DEFAULT_SOURCE
```

统一 runner 会记录二进制 SHA-256、Toolkit、编译 flags，并对 9 个 case 运行全部适用路径。完整结果见 [`cuda_triton_comparison_sm120_cu129_cuda130.json`](../results/cuda_triton_comparison_sm120_cu129_cuda130.json)。

## smoke 与正式实验分别证明什么

这个程序不引用 PyTorch，因此它证明：`nvcc` 为真实架构生成代码、driver 能
加载和 launch、同步/拷回成功、4097 个 ragged 元素数值正确。它不会把“系统
Toolkit 是否工作”和“PyTorch C++ extension 是否与 wheel ABI/runtime 匹配”
混成一个问题。

正式 standalone 算子现已完成，但仍不证明 PyTorch C++ extension ABI：本实验刻意把 CUDA Toolkit 13.0 程序与 PyTorch cu129 wheel 分开，双方通过确定性公式和各自 reference 对齐输入域，不在进程内混合 ABI。
