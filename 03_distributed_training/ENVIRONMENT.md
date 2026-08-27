# 03 环境教程：先 Gloo，再 NCCL

## 先说结论

本地 03 不需要新建第二个 Python 环境。继续使用 01/02 的根 `.venv`：CPU 多进程用
Gloo；GPU DDP 用 NCCL。`torchrun` 是启动多个 Python worker 的工具，不是新的
通信后端。系统 CUDA Toolkit/`nvcc` 也不是 DDP 前置条件；PyTorch cu129 wheel
和 Windows 显卡驱动已经提供本模块所需的 CUDA/NCCL 用户态能力。

## 1. 先确认项目位置

正式实验必须在 WSL Linux 文件系统中运行：

```bash
cd ~/projects/TrainScale_Lab
pwd
```

`pwd` 应以 `/home/` 开头。`/mnt/c`、`/mnt/d` 可做只读检查，但进程启动、trace、
checkpoint 和性能实验不要跨 Windows 挂载文件系统。

## 2. 安装和环境探针

```bash
uv sync --extra cu129 --extra dev --python 3.11
uv pip check --python .venv/bin/python

.venv/bin/python 03_distributed_training/benchmarks/check_environment.py
```

CPU 路线至少应得到：

```text
distributed_available: true
gloo_available: true
cpu_distributed_ready=True
```

GPU 路线还需要 `nccl_available=true`、`cuda_available=true` 和至少 1 张 GPU。
查看可见设备：

```bash
nvidia-smi -L
.venv/bin/python -c "import torch; print(torch.cuda.device_count())"
```

## 3. 三个名字不要混淆

| 名称 | 作用 | 本教程用途 |
|---|---|---|
| `torchrun` | 创建和监督多个 Python 进程，并设置 rank 环境变量 | 所有 03 实验的启动器 |
| Gloo | CPU collective 后端 | 无 GPU也能学 rank、sampler、DDP correctness |
| NCCL | NVIDIA GPU collective 后端 | 单/多 GPU DDP 性能路线 |

`torchrun --nproc-per-node=4` 表示本机启动 4 个 worker。CPU/Gloo 可以让 4 个
worker 共享 CPU；GPU/NCCL 通常要求一进程一 GPU，因此只有 1 张可见 GPU 时不能
启动 2 个 GPU worker 来假装双卡 scaling。

## 4. 第一个两 rank 检查

```bash
mkdir -p 03_distributed_training/results/raw/tutorial

.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment semantics --world-size 2 \
  --output 03_distributed_training/results/raw/tutorial/00_semantics.json
```

应看到 `semantics: success` 和 `all_checks_passed=True`。runner 内部实际调用
`python -m torch.distributed.run --standalone --nproc-per-node=2 ...`，每个 rank
写独立 JSON，再由父进程检查数量和值。这样某个 rank 崩溃不会被误记为完整成功。

## 5. 常见失败怎么判断

- `Address already in use`：另一个作业占用 rendezvous 端口；本项目使用
  `--standalone` 自动选端口，先检查是否有残留 torchrun 进程。
- 某些 rank 一直等待：通常是 collective 次数/顺序不一致，或一个 rank 已异常；
  不要简单增加 timeout 掩盖问题。
- `nproc-per-node` 大于 GPU 数：减少 world size，或换到真正的多 GPU 机器。
- NCCL 不可用：确认使用 cu129 环境并在 WSL 中运行，不要改成 Gloo 后仍把结果
  标成 NCCL 性能。
- Windows 原生多进程行为不同：本教程正式路线是 WSL/Linux，CPU CI 也在 Linux。

## 6. 多 GPU 机器怎样运行

```bash
nvidia-smi -L

.venv/bin/python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/gpu_scaling.toml \
  --output 03_distributed_training/results/raw/tutorial/gpu_scaling.json
```

runner 自动比较配置要求和 `torch.cuda.device_count()`：可执行的 world size 会真实
启动；超过设备数的 case 写 `unavailable`。如果机器只有 2 张 GPU，会运行 1/2，
把 4/8 保留为不可用，而不是失败或 0 throughput。

租用云 GPU 前不要只看显存。当前 runner 使用 `--standalone --nnodes=1`，必须租
同一实例中的多张 GPU，而不是多台独立单卡实例。还要保存 `nvidia-smi topo -m`，
因为 PCIe、NVLink 和 NUMA 路径会直接影响 AllReduce。

云端有两条明确路线：

1. **严格复现**：继续使用 Python 3.11、PyTorch 2.12.1+cu129 根环境；适合比较
   软件优化，但首次需下载数 GB CUDA wheel。
2. **硬件 scaling**：验证基础镜像现有 PyTorch/CUDA/NCCL 后，使用同一解释器完成
   所有 world size；相对 speedup 有效，但绝对吞吐不得与根环境当成同软件栈比较。

完整选型、私有仓库 bundle、smoke、三次正式测量、SHA-256 下载校验与关机步骤见
[实验 07：云端四卡](experiments/07_cloud_4gpu.md)。

## 参考资料

- [PyTorch：torchrun](https://docs.pytorch.org/docs/stable/elastic/run)
- [PyTorch：DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch：DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch：multi-GPU DDP tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
