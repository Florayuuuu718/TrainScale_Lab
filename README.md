# TrainScale Lab

**English** | [简体中文](README_zh-CN.md)

> Build a training system from a verifiable PyTorch training loop, then extend it layer by layer with GPU kernels, distributed training, collective communication, and a miniature training engine.

TrainScale Lab is a hands-on open-source project for learners interested in **ML Systems, AI Infrastructure, and Distributed Training**. It is neither a link collection nor a thin wrapper around existing frameworks. At every stage, you will build the smallest correct implementation, measure it, explain the bottleneck, and complete a reproducible optimization.

## Quick Navigation

[Start here](#start-here) · [Quick start](#01-quick-start) · [Learning path](#learning-path) · [Stage tasks](#stage-by-stage-work) · [Reproducibility](#reproducibility-contract)

| Destination | What it contains |
|---|---|
| [01 · PyTorch Training](01_pytorch_training/README.md) | Complete module 01 tutorial and reproduction order |
| [02 · GPU Kernels](02_gpu_kernels/README.md) | Runnable Triton kernels, acceptance checklist, and nine experiment reports |
| [03 · Distributed Training](03_distributed_training/README.md) | Runnable Gloo/DDP tutorial, scaling protocol, and seven experiment reports |
| [Source code](01_pytorch_training/trainscale_training) | Data, models, engine, checkpoint, benchmarks, and profiler |
| [Configurations](01_pytorch_training/configs/README.md) | TOML experiment recipes |
| [Correctness tests](01_pytorch_training/tests/README.md) | Explanation of all 10 tests |
| [Experiment reports](01_pytorch_training/experiments/README.md) | Successful experiments, theory, and troubleshooting |
| [Tracked results](01_pytorch_training/results/README.md) | Compact JSON summaries and SVG curves |
| [Documentation map](docs/README.md) | Concepts, environment, and repository foundation |
| [Module 01 acceptance checklist](docs/01-issues.md) | Foundation and implementation status for module 01 |

## Start Here

The repository now contains the complete module 01 and its repository foundation: a synthetic MLP baseline, CIFAR-10 CNN training, checkpoint/resume, AMP, gradient accumulation, profiling, and controlled performance experiments. The goal is training-system correctness and experimental method, not leaderboard accuracy.

If you are new to PyTorch, follow the Chinese beginner path in this order:

1. [Documentation map](docs/README.md)
2. [Start module 01](docs/getting-started/README.md)
3. [PyTorch training concepts](docs/concepts/pytorch-training-basics.md)
4. [01 · PyTorch Training reproduction guide](01_pytorch_training/README.md)

On Windows with an NVIDIA GPU, set up [WSL2 with the official Ubuntu distribution](docs/getting-started/wsl2-gpu.md) before running the GPU track. The native Windows commands below are the CPU/basic route; the complete compile, Profiler, Triton, and later NCCL track runs in Ubuntu.

## 01 Quick Start

The frozen baseline is Python 3.11, PyTorch 2.12.1, and CUDA 12.9 for NVIDIA GPU runs. CPU CI uses the matching PyTorch 2.12.1 CPU wheel.

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\ruff check .
.venv\Scripts\pytest
.venv\Scripts\python -m trainscale_training.train --config 01_pytorch_training/configs/synthetic_cpu.toml
```

For the full NVIDIA route, run `uv sync --extra cu129 --extra dev --python 3.11` inside Ubuntu and use `.venv/bin/...`; do not reuse the Windows `.venv`. The CPU and CUDA extras are intentionally mutually exclusive. Large raw checkpoints/traces, datasets, the local environment, and caches are ignored; compact JSON/SVG results are tracked with their analyses.

See the [environment guide](docs/getting-started/environment.md) for virtual-environment isolation, uv cache reuse, CPU/GPU wheel selection, CUDA verification, and when `nvcc` becomes necessary.

Modules 01 and 02 are sealed after local Windows CPU and Ubuntu GPU acceptance. On the RTX 5060 (SM 12.0), module 02's stable cu129/Triton environment passes all 15 final GPU tests. Its archived evidence includes 14 PyTorch/Triton forward cases, 41 PyTorch/Triton/CUDA kernel paths, LayerNorm forward/backward, finite MatMul tuning, and representative profiling. The machine-readable [module 02 acceptance record](02_gpu_kernels/results/module02_acceptance_sm120.json) separates static checks from real GPU execution.

For module 02, keep the stable root `.venv` by default and run the [crash-isolated environment probe](02_gpu_kernels/ENVIRONMENT.md) before any large experiment. If Triton fails, update the Windows NVIDIA driver and restart WSL first; only create the documented external cu130 nightly environment if the same probe still fails. A system CUDA Toolkit is required only for the CUDA C++ chapter, not for PyTorch or Triton.

Module 03's locally executable track is also complete: 2/4-rank CPU/Gloo semantics, sharding, gradient equivalence, checkpoint/resume, scaling, and profiling pass, and the one-GPU NCCL/DDP baseline runs on the same RTX 5060. Because this host exposes only one GPU, 2/4/8-GPU cases are archived as `unavailable` rather than fabricated measurements; the same frozen config is ready for a real multi-GPU host.

## What You Will Learn

After completing the main track, you should be able to:

- build PyTorch training, validation, checkpoint/resume, and profiling workflows;
- implement and compare common GPU operators with PyTorch, CUDA, and Triton;
- explain processes, ranks, data sharding, and gradient synchronization in DDP;
- use NCCL to measure collective latency, algorithm bandwidth, and bus bandwidth;
- implement Naive AllReduce and Ring AllReduce from scratch and analyze their communication complexity;
- build a miniature training engine with AMP, gradient accumulation, gradient bucketing, and communication/computation overlap;
- choose between DDP, FSDP2, and TP using evidence from memory, throughput, scaling efficiency, and profiler traces.

## Learning Path

```text
Training correctness
   ↓
Single-GPU performance and profiling
   ↓
CUDA / Triton kernels
   ↓
Multi-GPU training with DDP
   ↓
NCCL communication analysis
   ↓
Ring AllReduce from scratch
   ↓
Mini distributed training engine
   ↓
FSDP2 / Tensor Parallel
```

| Stage | What you build | Central question | Evidence produced |
|---|---|---|---|
| [01](01_pytorch_training/README.md) | Single-GPU PyTorch trainer | What makes a training run reliable? | Loss/accuracy, throughput, memory, resume consistency |
| [02](02_gpu_kernels/README.md) | GPU Kernel Lab | Why is an operator fast or slow? | Correctness, latency, bandwidth/TFLOPS, profiler evidence |
| [03](03_distributed_training/README.md) | Distributed Training Lab | Why does distributed training not scale linearly? | DDP correctness, CPU scaling, one-GPU NCCL, explicit multi-GPU boundary |
| 04 | NCCL Performance Lab | When is communication latency- or bandwidth-bound? | Message-size–bandwidth curves |
| 05 | TinyCollective | What actually happens inside AllReduce? | Naive/Ring/NCCL comparison |
| 06 | Mini Training Engine | How can communication be hidden and memory controlled? | Ablations, timelines, throughput improvements |
| 07 | FSDP2 / TP extension | How should a model be partitioned when it no longer fits? | Peak memory, correctness, scaling efficiency |

## Repository Layout

All seven module directories now exist. Modules 01 and 02 are sealed, and module 03's locally executable track is complete with an explicit multi-GPU hardware boundary. Each later directory has a README that states its scope and current status.

```text
trainscale-lab/
├── 01_pytorch_training/       # Reproducible single-GPU baseline
├── 02_gpu_kernels/            # PyTorch / CUDA / Triton comparisons
├── 03_distributed_training/   # DDP and scaling benchmarks
├── 04_nccl_benchmark/         # Collective communication experiments
├── 05_tiny_collective/        # Naive and Ring AllReduce
├── 06_training_engine/        # Final miniature distributed engine
├── 07_parallelism/            # FSDP2, TP, and composed parallelism
├── benchmarks/                # Shared benchmark entry points and schemas
├── docs/                      # Cross-module concepts, setup, and experiments
└── README.md
```

`01`–`07` are the only public module identifiers. Experiments may be numbered inside a module, but the repository does not use a second top-level milestone scheme. Each module should include its own README, environment description, minimal command, tests, and experiment report.

## How to Learn Instead of Merely Running Code

Every experiment follows the same loop:

1. **Predict** — write down the expected bottleneck and behavior before running anything.
2. **Baseline** — implement the simplest correct version first.
3. **Measure** — fix the hardware, data, warm-up, and number of iterations.
4. **Explain** — use profiler or communication metrics to identify the cause.
5. **Change one variable** — such as precision, batch size, bucket size, or kernel tile.
6. **Validate** — recheck numerical correctness and convergence, not only speed.
7. **Record** — commit configs, raw data, plots, and conclusions so others can reproduce them.

Recommended experiment table:

| experiment | hardware | precision | batch size | throughput | peak memory | correctness |
|---|---|---|---:|---:|---:|---|
| baseline | pending | FP32 | pending | pending | pending | unverified |
| optimization-A | same | TBD | same | pending | pending | unverified |

## Stage-by-Stage Work

### 01 · PyTorch Training

- Build `Dataset → DataLoader → forward → loss → backward → optimizer.step` yourself.
- Add validation, scheduling, checkpoint/resume, random seeds, and logging.
- Compare FP32, AMP, gradient accumulation, and `torch.compile`.
- Use Profiler to determine whether the bottleneck is data loading, CPU launch overhead, GPU compute, or memory.

Start with CIFAR-10 or an offline synthetic dataset. The goal is not maximum accuracy; it is reproducibility, correct resume behavior, and trustworthy measurement.

### 02 · GPU Kernels

Proceed through `Vector Add → ReLU → Softmax → LayerNorm → MatMul → Attention`. Every operator should include:

- a PyTorch reference;
- a naive implementation;
- an optimized CUDA or Triton implementation;
- numerical validation with `torch.testing.assert_close`;
- benchmarks across multiple shapes and dtypes.

Concepts such as threads, blocks, warps, coalesced access, shared memory, register pressure, occupancy, and roofline analysis should always be connected to observed results.

### 03 · Distributed Training

- First use Gloo + CPU to understand multiprocess semantics, then use NCCL + GPU for performance experiments.
- Implement `init_process_group`, `DistributedSampler`, DDP, and distributed checkpoints.
- Compare throughput on 1/2/4/8 GPUs and calculate speedup and scaling efficiency.
- Verify sample partitioning, loss aggregation, and parameter consistency across ranks.

### 04 · NCCL Benchmark

Use `nccl-tests` to benchmark AllReduce, AllGather, ReduceScatter, and Broadcast. Sweep from small to large messages, identify latency-bound and bandwidth-bound regions, and record the topology plus NCCL, CUDA, and driver versions.

### 05 · TinyCollective

Use `torch.distributed.send/recv` to build educational implementations of:

- Gather + Reduce + Broadcast;
- Ring ReduceScatter + Ring AllGather;
- correctness and performance comparisons with `torch.distributed.all_reduce`;
- communication volume, round count, and theoretical complexity, followed by experimental validation.

### 06 · Mini Training Engine

Combine the first five stages into a small, readable training engine and incrementally add:

- single-GPU and DDP execution;
- mixed precision and gradient accumulation;
- gradient bucketing;
- asynchronous collectives;
- communication/computation overlap;
- training, memory, compute, and communication profiling;
- minimal checkpoint and recovery support.

Every feature should have a flag and an ablation experiment that explains both its benefits and costs.

### 07 · FSDP2 / Tensor Parallel

Introduce FSDP2 only when a model genuinely does not fit on one GPU. Learn TP/PP and DeviceMesh when FSDP2 reaches scaling limits. The focus is why a technique is chosen—not merely how to make its configuration run.

## Start with the Hardware You Have

| Resources | What you can complete |
|---|---|
| CPU only | Training loops, tests, Gloo multiprocessing, collective logic |
| 1 NVIDIA GPU | AMP, Profiler, CUDA/Triton, single-GPU benchmarks |
| 2–4 GPUs | DDP, NCCL, Ring AllReduce, introductory FSDP2 |
| 8 GPUs or multiple nodes | Scaling, topology effects, overlap, composed parallelism |

Validate expensive experiments first with small correctness tests. Run only frozen benchmark configurations in the cloud, and publish the instance type and cost in the report.

## How to Study the Reference Projects

| Project | Role in TrainScale Lab | What to inspect |
|---|---|---|
| [pytorch/examples](https://github.com/pytorch/examples) | Training and distributed baselines | Project structure and official API usage |
| [timm](https://github.com/huggingface/pytorch-image-models) | Mature image training system | Data pipeline, optimizers, training engineering |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Small but complete GPT trainer | Training loop, checkpoints, DDP |
| [Triton](https://github.com/triton-lang/triton) | GPU kernel learning | Official tutorials and programming model |
| [TritonBench](https://github.com/triton-lang/tritonbench) | Benchmark design reference | Operator corpus and methodology |
| [CUDA Samples](https://github.com/NVIDIA/cuda-samples) | CUDA fundamentals and device capabilities | Memory, parallel model, toolchain |
| [nccl-tests](https://github.com/NVIDIA/nccl-tests) | Communication baseline | `all_reduce_perf` and `busbw` |
| [TorchTitan](https://github.com/pytorch/torchtitan) | Final architecture reference | Parallelization, checkpointing, profiling |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Large-scale parallelism reference | Composition of TP/PP/DP |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Systems-design reference | ZeRO, communication, memory optimization |
| [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) | Optimization methodology | How each change affects time and convergence |

Read source code with a concrete question: first expose a bottleneck in TrainScale Lab, then look for a design answer in a mature project. Do not copy the final implementation blindly.

## Reproducibility Contract

Every experiment report must record:

- Git commit, command, and complete configuration;
- GPU/CPU/interconnect topology;
- OS, Python, PyTorch, CUDA, NCCL, and driver versions;
- dataset version, random seed, and preprocessing;
- warm-up, repetitions, mean, and dispersion;
- throughput definition, peak memory, and correctness tolerance;
- failed experiments and known limitations.

Unmeasured results are explicitly labeled `pending`. Absolute performance across different hardware is not ranked directly; relative changes in the same environment are preferred.

## Contributing

The project is at an early stage. Contributions are welcome in the form of:

- smaller and clearer reference implementations;
- benchmarks reproducible across different hardware;
- correctness tests, performance-regression tests, and troubleshooting notes;
- counterexamples or more rigorous explanations of experimental conclusions.

Code contributions should include the environment, reproduction command, and result files. A performance screenshot without traceable evidence is not considered a complete experiment.

## Project Scope

TrainScale Lab is an educational and research implementation. It does not promise production-grade stability, security, or fault tolerance. RDMA verbs, DPDK, the Linux kernel network stack, and a full NCCL source-code study are intentionally not prerequisites; these topics can be introduced when a measured bottleneck calls for them.

## License

The project uses the [Apache-2.0 License](LICENSE). Reference projects are used for study and comparison; this does not imply that their code will be copied into this repository.
