# TrainScale Lab 交互式学习层

这里的 Notebook 用来做预测、观察中间量、读取正式 JSON 证据和绘图。训练、通信和多进程逻辑仍由
01–07 模块中的 Python runner 执行；Notebook 不维护第二份算法实现。

## 文件结构

```text
notebooks/
├── 00_start_here.ipynb
├── 01_reliable_training_loop.ipynb
├── 02_gpu_kernel_reasoning.ipynb
├── 03_ddp_fundamentals.ipynb
├── 04_nccl_latency_bandwidth.ipynb
├── 05_collective_algorithms.ipynb
├── 06_reducer_bucket_overlap.ipynb
├── 07_parallelism_strategies.ipynb
├── _support/                 # 路径、artifact、runner 和绘图辅助代码
├── _runs/                    # 本机生成，Git 忽略
└── build_notebooks.py        # 维护者使用的确定性生成脚本
```

## 在自己的电脑安装

推荐使用项目已有的 `uv` 环境。Windows PowerShell：

```powershell
uv venv --python 3.11 .venv
uv sync --extra cpu --extra notebook
uv run python -m ipykernel install --user --name trainscale-lab --display-name "TrainScale Lab (Python 3.11)"
uv run jupyter lab
```

Linux、macOS 或 WSL2 使用相同的四条命令。浏览器打开后进入 `notebooks/00_start_here.ipynb`，
内核选择 `TrainScale Lab (Python 3.11)`。只想安装经典界面的用户可继续使用同一环境：

```powershell
uv run jupyter notebook
```

如果不用 `uv`，可在 Python 3.11 虚拟环境中执行
`python -m pip install -e ".[cpu,notebook]"`，然后运行 `python -m jupyter lab`。

NVIDIA GPU 用户不要在原生 Windows 环境中把 CPU wheel 直接换成 CUDA extra。请先按
[WSL2 GPU 教程](../docs/getting-started/wsl2-gpu.md)准备环境，再执行
`uv sync --extra cu129 --extra notebook`。

## 三种模式

每本 Notebook 第一处可编辑单元都有：

```python
MODE = "reference"  # reference | local | gpu
```

- `reference`：默认。只读取仓库内精简结果，任何普通 CPU 电脑都能从头执行。
- `local`：运行轻量 correctness 或最小实验，输出写入 `_runs/`。
- `gpu`：调用正式 GPU runner；多卡章节仍需 Linux、CUDA/NCCL 和对应数量的 GPU。

环境变量 `TRAINSCALE_NOTEBOOK_MODE=reference` 可以强制使用参考模式，CI 正是这样验收。

## 使用规则

1. 每次从 `Kernel → Restart Kernel and Run All Cells` 开始，确认状态卡里的 mode 和 Python。
2. 先写预测，再运行观察和 correctness 单元，最后才看性能。
3. `_runs/` 是临时结果，不是正式结论；不要把它直接提交到 Git。
4. 失败日志会保留，修复后使用新的 run id，不覆盖失败证据。
5. 多进程和昂贵 GPU 实验优先使用 JupyterLab Terminal，便于看到退出码和完整日志。

## 验证 Notebook

```powershell
uv run --extra notebook python notebooks/build_notebooks.py
uv run --extra notebook --extra dev pytest -q notebooks/tests
```

测试会检查 nbformat、稳定 cell id、相对链接、敏感绝对路径，并在强制 `reference` 模式下执行全部
Notebook。Notebook 内容要调整时，修改 `build_notebooks.py` 后重新生成，不要只手工修改某一个
JSON 文件。
