# 实验 07：从租云 GPU 到可信的四卡结果

## 为什么做

本地单卡可以证明 NCCL/DDP 代码能运行，却不能证明 2/4 卡的真实 speedup。租用
云 GPU 的难点也不只是“找到四张卡”：如果租成四台互不相通的单卡实例、在付费
时段才开始调环境、只跑一次就挑最快值，或者结果没有下载便释放实例，最后仍然
得不到可复核的实验。

本实验把一次付费云端测量拆成完整闭环：冻结代码 → 选择单机多卡 → 检查拓扑 →
环境探针 → smoke test → 三次正式实验 → 中位数汇总 → SHA-256 校验 → 关机止费。

## 小白名词

- **单机多卡**：一个 Linux 实例同时看到多张 GPU。本模块的 `--standalone` 启动器
  只支持这种模式，不支持几台独立单卡实例组成多机作业。
- **拓扑**：GPU、PCIe、CPU NUMA 节点和网卡之间的连接关系。相同型号和数量的 GPU
  也可能因为连接路径不同而得到不同通信性能。
- **NUMA**：一台双路服务器的 CPU/内存局部性结构。跨 NUMA 的 `SYS` 路径通常比
  同一 NUMA 域内的路径更远。
- **NVLink**：部分数据中心 GPU 提供的高速 GPU 互连。本次 4090D 主机没有 NVLink。
- **smoke test**：先用小模型、少量 step 验证流程，不用于发布最终性能结论。
- **中位数**：三次测量排序后的中间值。它比“挑最快一次”更不容易被偶发抖动误导。
- **按量计费**：实例处于运行状态便计费，不取决于 GPU 利用率是否为 0。

## 一般预期

四卡实例应让 `torch.cuda.device_count()` 返回 4，且 NCCL 可用。smoke 的 1/2 卡
case 应成功；正式配方的 1/2/4 卡应成功，8 卡因硬件不足应为 `unavailable`。

不要预先假定四卡一定比单卡快。Strong scaling 中每个 rank 的计算量随卡数减少，
但模型梯度大小和 AllReduce 仍然存在；如果模型很小，通信和 launch 开销可能超过
新增计算资源的收益。Weak scaling 更可能得到正 speedup，但仍会受通信和拓扑限制。

## 1. 付费前先冻结代码

在本地完成测试和提交，再租 GPU。不要把宝贵的四卡时间用于编辑代码：

```powershell
cd D:\00A\project\TrainScale_Lab

git status --short
git push origin HEAD
```

`git status --short` 应没有输出。私有仓库不想配置 GitHub Token 时，可创建 Git bundle：

```powershell
git bundle create ..\TrainScale_Lab.bundle --all
git bundle verify ..\TrainScale_Lab.bundle
```

把 `TrainScale_Lab.bundle` 上传到云实例的数据盘，再执行：

```bash
cd /root/autodl-tmp
git clone ./TrainScale_Lab.bundle TrainScale_Lab
cd TrainScale_Lab

git rev-parse --short HEAD
git status --short
```

终端中的 URL 必须是纯 URL，不能把 Markdown 的 `[文字](地址)` 原样粘进 Bash；
目录名中的 `_` 也不需要写成 `\_`。

## 2. 主机和镜像怎样选

最低选择契约如下：

| 项目 | 要求 | 原因 |
|---|---|---|
| 实例 | 一台实例中的 4 张相同 GPU | 当前 runner 是单机 `--standalone --nnodes=1` |
| 显存 | 每卡至少 12GB | 当前教学 MLP 很小，不必为 80GB 显存付费 |
| 系统 | Ubuntu 22.04/24.04 | PyTorch/NCCL 正式路线使用 Linux |
| 驱动 | 能运行选定 PyTorch CUDA wheel | 容器内通常不能自行升级宿主驱动 |
| 磁盘 | 项目和缓存放数据盘 | CUDA wheel 依赖总量可达数 GB |
| 镜像 | 官方基础镜像 | 减少未知依赖和凭据风险 |

本次选择 AutoDL 单机 `4×RTX 4090D 24GB`、64 CPU 核、320GB 内存，页面当时显示
单卡 ¥1.88/小时，即四卡 ¥7.52/小时。价格和库存会变化，教程只把它作为本次实验
记录，不把它当作长期报价。AutoDL 官方说明同一实例中的多卡位于同一物理主机；
按量实例关机后停止计费，但 GPU 不会保留。

镜像自带 Python 3.10 或 3.12 都不是决定因素。真正要先决定的是实验目标：

### 路线 A：严格复现根环境

需要和本地 Python 3.11、PyTorch 2.12.1+cu129 完全一致时使用：

```bash
cd /root/autodl-tmp/TrainScale_Lab
export UV_CACHE_DIR=/root/autodl-tmp/.uv-cache

uv sync --extra cu129 --extra dev --python 3.11
.venv/bin/python 03_distributed_training/benchmarks/check_environment.py
```

优点是软件环境一致；代价是第一次可能下载 torch、cuDNN、cuBLAS、NCCL、Triton
等数 GB 文件。最好在单卡/无卡准备阶段完成缓存，再扩容到四卡。

### 路线 B：只测同机相对 scaling

如果云镜像已经包含可用的 PyTorch/CUDA/NCCL，可先探针：

```bash
python -c "import sys, torch; import torch.distributed as dist; \
print('Python:', sys.version); print('torch:', torch.__version__); \
print('CUDA:', torch.version.cuda); print('GPUs:', torch.cuda.device_count()); \
print('NCCL:', dist.is_nccl_available())"
```

只要 CUDA 可用、GPU 数为 4、NCCL 为 `True`，可用该解释器直接运行 03 的脚本。
所有 world size 都在同一软件环境内，因此相对 speedup 有效；但绝对吞吐不能与根
环境结果做同软件栈比较。本次为节省付费时间采用该路线：Python 3.12.3、PyTorch
2.8.0+cu128、driver 580.76.05。PyTorch 从 2.6 起不再发布官方 Conda binary，
因此不能假设 `conda install pytorch` 能得到项目锁定的 2.12.1+cu129；Conda 可用于
准备 Python，官方 PyTorch CUDA build 仍应使用对应 wheel。

## 3. 先证明四张卡和拓扑是真的

```bash
nvidia-smi -L
nvidia-smi topo -m
```

本次拓扑是：

```text
GPU0 ↔ GPU1: NODE, NUMA 0
GPU2 ↔ GPU3: NODE, NUMA 1
两组之间:     SYS
NVLink:        无
```

2 卡 case 默认使用 GPU0/1，留在一个 NUMA 域；4 卡 case 跨两个 NUMA 域。这个差异
是分析 scaling 拐点时必须保留的证据，不能只写“4×4090D”。

保存环境：

```bash
mkdir -p 03_distributed_training/results/raw/rental

nvidia-smi > 03_distributed_training/results/raw/rental/nvidia-smi.txt
nvidia-smi -L > 03_distributed_training/results/raw/rental/gpu-list.txt
nvidia-smi topo -m > 03_distributed_training/results/raw/rental/gpu-topology.txt

python 03_distributed_training/benchmarks/check_environment.py \
  --output 03_distributed_training/results/raw/rental/environment.json
```

使用路线 A 时，把以上 `python` 换成 `.venv/bin/python`；后文同理。

## 4. 先跑 smoke，失败就不要烧正式预算

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

python 03_distributed_training/benchmarks/run_scaling.py \
  --config 03_distributed_training/configs/gpu_scaling_smoke.toml \
  --output 03_distributed_training/results/raw/rental/gpu_smoke.json

unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS

python 03_distributed_training/benchmarks/show_distributed_results.py \
  03_distributed_training/results/raw/rental/gpu_smoke.json
```

本次 smoke 的实际结果：

| mode | world | throughput | speedup | efficiency |
|---|---:|---:|---:|---:|
| strong | 1 | 31,736.6 | 1.000 | 100.0% |
| strong | 2 | 43,173.4 | 1.360 | 68.0% |
| weak | 1 | 23,007.0 | 1.000 | 100.0% |
| weak | 2 | 43,110.5 | 1.874 | 93.7% |

它证明双卡 NCCL 路径可运行，不是正式吞吐结论，因为只测了 3 step 的小模型。

## 5. 正式实验为什么要跑三次

```bash
for run in 1 2 3
do
  python 03_distributed_training/benchmarks/run_scaling.py \
    --config 03_distributed_training/configs/gpu_scaling.toml \
    --output "03_distributed_training/results/raw/rental/gpu_formal_run${run}.json"
done
```

每次应有 strong/weak 的 1/2/4 卡 `success`，8 卡 `unavailable`。三次运行必须使用
同一 config、commit、环境和干净 worktree；汇总脚本会检查这些条件，不能把不一致
的运行混在一起。

## 6. 下载前打包和校验，之后再关机

```bash
tar -czf /root/autodl-tmp/module03-4gpu-results.tar.gz \
  03_distributed_training/results/raw/rental

sha256sum /root/autodl-tmp/module03-4gpu-results.tar.gz
```

本次服务器压缩包 SHA-256 为：

```text
63b0bb1efc17313cfd9df381afe67281d9daa2eb634a4fe570861ca7f3077e54
```

下载后在 Windows 验证：

```powershell
Get-FileHash `
  -LiteralPath "D:\Downloads\module03-4gpu-results.tar.gz" `
  -Algorithm SHA256
```

哈希一致后再在控制台点击“关机”，并等状态变为“已关机”。关闭浏览器、SSH 或让
GPU 利用率降到 0 都不会停止按量计费；“无卡模式”可用于之后整理/下载，但仍有
低额费用；“释放实例”会清除数据，不应在本地校验前使用。

## 7. 在本地生成可复核的中位数结果

把八个证据文件放到：

```text
03_distributed_training/results/evidence/cloud_4x4090d/
```

运行：

```powershell
.venv\Scripts\python `
  03_distributed_training\benchmarks\aggregate_scaling_runs.py `
  --evidence-directory 03_distributed_training\results\evidence\cloud_4x4090d `
  --archive-sha256 63b0bb1efc17313cfd9df381afe67281d9daa2eb634a4fe570861ca7f3077e54 `
  --output 03_distributed_training\results\scaling_nccl_4x4090d.json
```

脚本验证三次 case、配置、commit、环境、状态和干净 worktree，记录每个源文件的
SHA-256，再对吞吐、最慢 rank 时间和显存取中位数。Speedup 从中位吞吐重新计算，
不是对三次 speedup 随意平均。

## 实际结果

环境：AutoDL 单机 4×RTX 4090D、无 NVLink、两个 NUMA 域、Python 3.12.3、
PyTorch 2.8.0+cu128、driver 580.76.05。源提交为 `d2b2882`，三次 worktree 均干净。

| mode | world | local/global batch | 中位吞吐 | speedup | efficiency | 三次相对极差 |
|---|---:|---:|---:|---:|---:|---:|
| strong | 1 | 256 / 256 | 252,630.9 | 1.000 | 100.0% | 15.2% |
| strong | 2 | 128 / 256 | 151,684.8 | 0.600 | 30.0% | 10.2% |
| strong | 4 | 64 / 256 | 128,908.3 | 0.510 | 12.8% | 1.39% |
| weak | 1 | 128 / 128 | 126,560.8 | 1.000 | 100.0% | 18.2% |
| weak | 2 | 128 / 256 | 145,138.2 | 1.147 | 57.3% | 8.39% |
| weak | 4 | 128 / 512 | 255,353.6 | 2.018 | 50.4% | 0.96% |
| strong/weak | 8 | — | — | — | — | unavailable：只有 4 GPU |

“三次相对极差”是 `(max - min) / median`，用于提醒读者测量抖动，而不是置信区间。

## 理论解释

一次 DDP step 可以粗略写成：

```text
T_step ≈ max_rank(T_forward + T_backward) + T_AllReduce + T_launch/等待
```

Strong scaling 固定 global batch=256。world 从 1 增加到 4 时，每 rank batch 从
256 降到 64，小 MLP 的矩阵乘法越来越短；但参数梯度大小没有按四倍缩小，NCCL
collective、进程调度和同步等待仍在。因此本实验的 strong 2/4 卡不但没有加速，
反而只有单卡吞吐的 60.0%/51.0%。这是“工作粒度太小”的反例，不是 DDP 错误。

Weak scaling 固定每 rank batch=128，world=4 时 global batch=512，所以每张卡仍有
相同计算工作。四卡吞吐达到单卡的 2.018 倍，但离理想 4 倍仍远，效率 50.4%。
原因包括梯度 AllReduce、无 NVLink、GPU0/1 与 GPU2/3 之间跨 `SYS`/NUMA，以及
最慢 rank 决定 step 时间。

还必须看到测量窗口限制：正式配置只有 5 次 warm-up 和 20 次 measured step，单卡
一次正式计时只有约 20ms，导致单卡三次吞吐相对极差达到 15%–18%。因此这些数据
足以证明真实 NCCL 多卡路径和“小工作负载不线性扩展”的机制，不适合当作 4090D
生产训练性能榜单。后续比较优化时应增加 step、扩大模型，并同时使用 Profiler 或
NCCL benchmark 分离计算和通信时间。

## 结论与收尾

本次补齐了真实 1/2/4 GPU NCCL 实验：所有 18 个可执行 repetition-case 成功，
8 GPU 仍诚实标为不可用。最重要的结论不是“四卡有多快”，而是：

1. GPU 数量增加不保证 strong scaling；工作粒度和通信计算比决定结果；
2. Weak scaling 更接近正扩展，但本机拓扑下四卡效率仍只有约 50%；
3. 云端结果必须连同 commit、环境、拓扑、重复值、哈希和硬件边界一起发布；
4. smoke、正式测量、下载校验和关机止费应成为一个固定流程。

机器可读中位数见
[`scaling_nccl_4x4090d.json`](../results/scaling_nccl_4x4090d.json)，原始证据见
[`results/evidence/cloud_4x4090d/`](../results/evidence/cloud_4x4090d/)。

## 参考资料

- [PyTorch：torchrun](https://docs.pytorch.org/docs/stable/elastic/run)
- [PyTorch：DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch：从 2.6 起停止发布官方 Conda binary](https://pytorch.org/blog/pytorch2-6/)
- [AutoDL：实例与单机多卡](https://www.autodl.com/docs/env/)
- [AutoDL：按量计费与关机](https://api.autodl.com/docs/price/)
- [AutoDL：数据保留规则](https://www.autodl.com/docs/instance_data/)
