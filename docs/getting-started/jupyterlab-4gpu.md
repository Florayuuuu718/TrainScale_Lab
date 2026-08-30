# 从零在 JupyterLab 完成 04–07 四卡实验

这是一条面向第一次租 GPU 的完整路线。你不需要先会 Jupyter Notebook；本项目主要使用
JupyterLab 里的 **Terminal** 运行命令，用左侧文件浏览器确认和下载结果。Notebook 不是
分布式实验的必需入口，因为 `torchrun` 会启动多个独立进程，Terminal 更容易看到退出码和日志。

## 0. 先理解你要租什么

租一台包含 **4 张同型号 NVIDIA GPU 的单机实例**，不是四台单卡实例。推荐 Ubuntu 22.04/24.04、
PyTorch CUDA 开发镜像、每卡至少 16 GiB、至少 50 GiB 可写空间。04–07 的必需 world size 最大为
4；8 卡、NVLink 和多节点是扩展，不是 v1.0 完成条件。

复现分两种目标：

- **复现实验结论**：当前项目锁定 Python 3.11、PyTorch 2.12.1；版本变化后绝对吞吐允许不同，
  correctness 和趋势应按教程重新判断。
- **审阅归档数字**：2026-08 的参考归档使用 Ubuntu 22.04、Python 3.12.3、PyTorch
  2.8.0+cu128、CUDA toolkit 12.8、NCCL 2.27.3、4×RTX 4090 D。它是历史冻结环境，
  不等于要求新用户降级当前依赖。

驱动显示的“CUDA Version”是驱动能支持的最高 CUDA 版本，`torch.version.cuda` 是 PyTorch
wheel 使用的 runtime，`nvcc --version` 是本机 toolkit。三者不必完全相同；只要驱动足够新、
PyTorch 能识别 GPU，通常可以运行。编译 `nccl-tests` 时才必须关注 `nvcc` 和头文件/库路径。

参考 4090 D campaign 在代码和 `nccl-tests` 已准备好后，04–07 主任务约需 30–50 分钟；下载、
网络或首次构建可能显著延长。按小时计费时以 2 小时为目标窗口、3 小时为费用余量，但不要
为了赶整点删除 repetitions。
环境门失败就先关机排障，本地修好后再租卡，通常比在云端临时改代码更省钱。

## 1. 打开 JupyterLab 和 Terminal

实例启动后，在云平台控制台打开 JupyterLab。界面通常有三部分：左侧文件浏览器、中央工作区、
顶部菜单。点击 Launcher 中的 **Terminal**；如果没有 Launcher，使用 `File → New → Terminal`。

命令块末尾的反斜杠 `\` 表示“下一行仍属于同一条命令”。必须从代码块复制原始字符，不要复制
网页渲染产生的 `\_`、`&#x20;` 或 Markdown 链接括号。

每条长命令结束后可以运行 `echo $?`：`0` 通常表示成功，非零表示需要先检查刚才的输出。
不要在不知道失败原因时继续后续模块。唯一例外是 04 长窗口 campaign 的退出码 `2`，它表示
correctness 已通过但重复波动超过质量门，教程会在对应位置再次说明。

先设两个变量。实例重启后 shell 变量会消失，需要重新执行：

```bash
export PROJECT_ROOT=/root/TrainScale_Lab
export RUN_ROOT=/root/trainscale-results/rental-0407
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p "$RUN_ROOT"/{environment,module04,module05,module06,module07}
```

如果你的项目目录不同，用下面命令寻找，再修改 `PROJECT_ROOT`：

```bash
find /root -maxdepth 3 -name pyproject.toml -type f
```

## 2. 获取代码并冻结本次版本

首次使用：

```bash
cd /root
git clone https://github.com/Florayuuuu718/TrainScale_Lab.git TrainScale_Lab
cd "$PROJECT_ROOT"
```

已有目录：

```bash
cd "$PROJECT_ROOT"
git status -sb
git pull --ff-only
```

确认代码存在后，后续每次新开 Terminal 都先进入项目：

```bash
cd "$PROJECT_ROOT" || {
  echo "项目目录不存在，请检查 PROJECT_ROOT"
  exit 1
}
```

不要在正式 campaign 中途 `git pull` 或升级 PyTorch。记录 commit：

```bash
git status --short
git rev-parse HEAD | tee "$RUN_ROOT/environment/project-git-commit.txt"
```

`git status --short` 理想情况下无输出。若有改动，先确认它们是不是你要测试的代码；不要用
`git reset --hard` 粗暴清除未知文件。

## 3. 环境预检：每一行都要看懂

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv | tee "$RUN_ROOT/environment/gpu-summary.txt"

nvidia-smi topo -m | tee "$RUN_ROOT/environment/topology.txt"

python - <<'PY' | tee "$RUN_ROOT/environment/torch-nccl.txt"
import torch
import torch.distributed as dist

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())
print("gpu_names=", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("nccl_available=", dist.is_nccl_available())
print("nccl_version=", torch.cuda.nccl.version() if dist.is_nccl_available() else None)
PY

python 04_nccl_benchmark/benchmarks/check_environment.py \
  --require-multi-gpu \
  --output "$RUN_ROOT/environment/module04-environment.json"
```

继续前确认：`gpu_count=4`、四张卡 `memory.used` 接近 0、`nccl_available=True`、
`module04_multi_gpu_ready=True`。`topo -m` 中 `PHB/NODE/SYS` 是路径类别，不是性能结论；
是否有影响要靠 04 的可控实验。

若缺少 pytest，只安装测试工具，不替换云镜像中的 torch：

```bash
python -m pip install "pytest>=8.3,<10"
python -m pytest -q \
  04_nccl_benchmark/tests \
  05_tiny_collective/tests \
  06_training_engine/tests \
  07_parallelism/tests
```

Linux CPU/Gloo 的 FSDP2 能力可能因 PyTorch 版本显示 unavailable；这不能直接判定 CUDA/NCCL
失败。07 的正式 runner 会先执行独立 CUDA preflight，并在失败时停止性能实验。

## 4. 04：先测通信，再解释训练

从未构建过时，先读 [`../../04_nccl_benchmark/ENVIRONMENT.md`](../../04_nccl_benchmark/ENVIRONMENT.md)。
不要把“更换 GPU”当作编译器/头文件冲突的修复；构建问题由兼容的 Ubuntu、CUDA toolkit、
编译器和 NCCL headers/library 组合解决。

先指定包含 `all_reduce_perf`、`all_gather_perf` 等程序的 build 目录：

```bash
export NCCL_TEST_DIR=/root/nccl-tests-v2.19.7-src/build
test -x "$NCCL_TEST_DIR/all_reduce_perf" || {
  echo "nccl-tests binary missing"
  exit 1
}
```

先 smoke。失败就停止，不进入 formal：

```bash
python 04_nccl_benchmark/benchmarks/run_collectives.py \
  --config 04_nccl_benchmark/configs/nccl_smoke.toml \
  --binary-directory "$NCCL_TEST_DIR" \
  --raw-directory "$RUN_ROOT/module04/smoke-raw" \
  --output "$RUN_ROOT/module04/smoke.json"
```

正式配置原样运行三次，每次写入独立目录：

```bash
for run in 1 2 3; do
  python 04_nccl_benchmark/benchmarks/run_collectives.py \
    --config 04_nccl_benchmark/configs/nccl_formal.toml \
    --binary-directory "$NCCL_TEST_DIR" \
    --raw-directory "$RUN_ROOT/module04/formal-run${run}-raw" \
    --output "$RUN_ROOT/module04/formal-run${run}.json" || exit 1
done

python 04_nccl_benchmark/benchmarks/aggregate_runs.py \
  "$RUN_ROOT/module04/formal-run1.json" \
  "$RUN_ROOT/module04/formal-run2.json" \
  "$RUN_ROOT/module04/formal-run3.json" \
  --output "$RUN_ROOT/module04/formal-aggregate.json"
```

把真实梯度 payload 接回 DDP：

```bash
python 04_nccl_benchmark/benchmarks/plan_ddp_bridge.py \
  --config 04_nccl_benchmark/configs/ddp_bridge.toml \
  --output "$RUN_ROOT/module04/ddp-payload-plan.json"

python 04_nccl_benchmark/benchmarks/run_ddp_bridge.py \
  --config 04_nccl_benchmark/configs/ddp_bridge.toml \
  --raw-directory "$RUN_ROOT/module04/ddp-bridge-raw" \
  --output "$RUN_ROOT/module04/ddp-bridge.json" \
  --timeout-seconds 600
```

pair 的拓扑含义必须以本次 `topology.txt` 为准，不能把配置名中的 pair 编号直接解释成
“近/远”。完整实验设计见
[`../../04_nccl_benchmark/experiments/04_multi_gpu_campaign.md`](../../04_nccl_benchmark/experiments/04_multi_gpu_campaign.md)。

若要复现本项目已发现的 scaling 边界：

```bash
python 04_nccl_benchmark/benchmarks/run_ddp_scaling_campaign.py \
  --config 04_nccl_benchmark/configs/ddp_scaling_long.toml \
  --output-directory "$RUN_ROOT/module04/ddp-scaling-long" \
  --summary-output "$RUN_ROOT/module04/ddp-scaling-long.json" \
  --repetitions 5 \
  --stability-threshold 0.05 \
  --warning-threshold 0.10 \
  --timeout-seconds 1800
```

退出码 2 表示波动超过质量门，不是 correctness 失败。保留结果，用中位数和相对极差说明
测量边界；不要反复重跑直到出现一个“好看”的数字。

## 5. 05：亲手实现的 Ring 与 NCCL 对照

```bash
python 05_tiny_collective/benchmarks/run_correctness.py \
  --config 05_tiny_collective/configs/cpu_correctness.toml \
  --output "$RUN_ROOT/module05/cpu-correctness.json"

python 05_tiny_collective/benchmarks/run_gpu_comparison.py \
  --config 05_tiny_collective/configs/gpu_comparison.toml \
  --raw-directory "$RUN_ROOT/module05/gpu-raw" \
  --output "$RUN_ROOT/module05/gpu-comparison.json"
```

先确认 `correctness.status=passed`，再比较大消息；小消息的 GB/s 很容易被固定延迟放大。预期
学习结果是理解 root 热点、ring 的轮数/通信量，以及“教学算法”和“生产实现”的差距。

## 6. 06：Reducer、AMP 和 Overlap

先用单设备和 CPU/Gloo reference 验证训练与 reducer 数学：

```bash
python 06_training_engine/benchmarks/run_single_device_baseline.py \
  --config 06_training_engine/configs/local_baseline.toml \
  --output "$RUN_ROOT/module06/local-baseline.json"

python 06_training_engine/benchmarks/run_reducer_correctness.py \
  --config 06_training_engine/configs/local_correctness.toml \
  --raw-directory "$RUN_ROOT/module06/local-correctness-raw" \
  --output "$RUN_ROOT/module06/local-correctness.json"
```

这一步失败时不要启动 GPU ablation。性能更快不能补救梯度或参数更新错误。

```bash
python 06_training_engine/benchmarks/run_gpu_ablation.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/ablation-raw" \
  --output "$RUN_ROOT/module06/ablation.json"

python 06_training_engine/benchmarks/run_amp_overflow_probe.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/amp-overflow-raw" \
  --output "$RUN_ROOT/module06/amp-overflow.json"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --raw-directory "$RUN_ROOT/module06/profile-raw" \
  --output "$RUN_ROOT/module06/overlap-profile.json"
```

Profiler 启动失败时先关闭 Git/Hugging Face 网络加速代理；代理环境变量可能被 torchrun 子进程
继承。不要覆盖失败目录，换一个 `profile-raw-fixed` 名称保留失败证据。

延伸实验只改变 bucket size：

```bash
python 06_training_engine/benchmarks/run_overlap_profile.py \
  --config 06_training_engine/configs/gpu_ablation.toml \
  --bucket-cap-mb 1.0 \
  --raw-directory "$RUN_ROOT/module06/extension-1m/profile-raw" \
  --output "$RUN_ROOT/module06/extension-1m/overlap-profile-1m.json"
```

在 JupyterLab 左侧找到 `.json`，双击可查看；Chrome trace 通常很大，建议先打包再下载。
“有 overlap”必须和 step throughput 一起判断，不能只凭时间线宣布优化成功。

## 7. 07：DDP、FSDP2、TP 的选择实验

先生成理论显存估算并运行本地语义/capability gates：

```bash
python 07_parallelism/benchmarks/estimate_memory.py \
  --output "$RUN_ROOT/module07/memory-estimate.json"

python 07_parallelism/benchmarks/run_tp_correctness.py \
  --config 07_parallelism/configs/local_correctness.toml \
  --raw-directory "$RUN_ROOT/module07/tp-local-raw" \
  --output "$RUN_ROOT/module07/tp-local.json"

python 07_parallelism/benchmarks/run_fsdp2_capability.py \
  --raw-directory "$RUN_ROOT/module07/fsdp2-local-raw" \
  --output "$RUN_ROOT/module07/fsdp2-local.json"

python 07_parallelism/benchmarks/run_native_tp_capability.py \
  --raw-directory "$RUN_ROOT/module07/native-tp-local-raw" \
  --output "$RUN_ROOT/module07/native-tp-local.json"
```

CPU/Gloo FSDP2 可能按版本记录为 `unavailable`。这时保留 artifact 并继续正式 runner；真正的
CUDA/NCCL FSDP2 preflight 会在下一条命令开头执行，失败则自动停止性能矩阵。

```bash
python 07_parallelism/benchmarks/run_gpu_parallelism.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --raw-directory "$RUN_ROOT/module07/gpu-raw" \
  --output "$RUN_ROOT/module07/gpu-parallelism.json"

python 07_parallelism/benchmarks/run_gpu_profiles.py \
  --config 07_parallelism/configs/gpu_parallelism.toml \
  --correctness-artifact "$RUN_ROOT/module07/gpu-parallelism.json" \
  --raw-directory "$RUN_ROOT/module07/profile-raw" \
  --output "$RUN_ROOT/module07/gpu-profiles.json"
```

第一条命令先跑 correctness preflight，只有通过才进入性能矩阵。看到 FSDP/TP 比 DDP 慢并不
表示实验失败；对小模型而言，collective 代价超过切分收益是重要结论。不要用别的进程抢占显存
来制造 OOM。

## 8. 生成模块验收摘要

```bash
python 06_training_engine/benchmarks/summarize_module06.py \
  --baseline "$RUN_ROOT/module06/local-baseline.json" \
  --correctness "$RUN_ROOT/module06/local-correctness.json" \
  --gpu-ablation "$RUN_ROOT/module06/ablation.json" \
  --overlap-profile "$RUN_ROOT/module06/overlap-profile.json" \
  --amp-overflow "$RUN_ROOT/module06/amp-overflow.json" \
  --output "$RUN_ROOT/module06/acceptance.json"

python 07_parallelism/benchmarks/summarize_module07.py \
  --memory "$RUN_ROOT/module07/memory-estimate.json" \
  --tp-correctness "$RUN_ROOT/module07/tp-local.json" \
  --fsdp2-capability "$RUN_ROOT/module07/fsdp2-local.json" \
  --native-tp-capability "$RUN_ROOT/module07/native-tp-local.json" \
  --gpu-parallelism "$RUN_ROOT/module07/gpu-parallelism.json" \
  --gpu-profiles "$RUN_ROOT/module07/gpu-profiles.json" \
  --output "$RUN_ROOT/module07/acceptance.json"
```

快速查看顶层状态：

```bash
python - "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("module*/**/*.json")):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if "status" in value and path.name in {
        "smoke.json",
        "formal-aggregate.json",
        "ddp-bridge.json",
        "ddp-scaling-long.json",
        "gpu-comparison.json",
        "acceptance.json",
        "gpu-parallelism.json",
        "gpu-profiles.json",
    }:
        print(path.relative_to(root), "status=", value["status"])
PY
```

06/07 的 `acceptance.json` 应为 `complete`。04 long scaling 可以是 artifact `success` 但
measurement quality `failed`；这表示实验完成且波动被诚实记录。

## 9. 检查、打包、下载、校验

先列出顶层结果，不让上一次失败文件冒充本次结果：

```bash
find "$RUN_ROOT" -maxdepth 3 -type f -name '*.json' -print | sort
```

生成内部文件清单和归档。先生成清单，再创建归档，避免清单递归包含自己：

```bash
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"

tar -C /root/trainscale-results \
  -czf /root/trainscale-rental-0407.tar.gz rental-0407

sha256sum /root/trainscale-rental-0407.tar.gz \
  > /root/trainscale-rental-0407.tar.gz.sha256
```

在 JupyterLab 左侧进入 `/root`，右键两个文件并选择 Download：

- `trainscale-rental-0407.tar.gz`
- `trainscale-rental-0407.tar.gz.sha256`

若浏览器不支持下载目录，这是正常的；目录必须先 `tar -czf` 成单个文件。`profile-raw/` 不必
逐文件下载，它已经包含在归档中。

本地 Windows PowerShell 校验：

```powershell
Get-FileHash .\trainscale-rental-0407.tar.gz -Algorithm SHA256
Get-Content .\trainscale-rental-0407.tar.gz.sha256
tar -tzf .\trainscale-rental-0407.tar.gz | Select-Object -First 20
```

两个 SHA-256 必须相同，并且压缩包能列出内容。完成后回到云平台控制台执行“关机/释放实例”；
关闭 JupyterLab 标签页不会停止计费。

## 10. 读结果的固定顺序

每章都按下面五问读，不要先找最大数字：

1. correctness 是否通过？
2. 硬件、版本、配置和 commit 是否固定？
3. 指标的定义是什么，重复和波动怎样？
4. profiler/拓扑/通信量能否解释现象？
5. 结论在哪些模型、消息大小、拓扑和版本上成立？

参考结论分别见 04、05、06、07 的 `experiments/*_final_report.md`。你的绝对数字可以不同，
但必须用同一证据链说明为什么相同或不同。
