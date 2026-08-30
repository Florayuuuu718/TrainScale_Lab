# 在 Windows 项目中补齐 Linux/Gloo 集成测试

这篇教程解决一个常见疑问：在 Windows 运行 `pytest` 时看到 `skipped`，是不是项目没有测试
完整？答案取决于跳过原因。TrainScale Lab 的 05–07 模块有 6 项测试需要 Linux 的多进程和
Gloo 路径；它们在原生 Windows 上按设计跳过，但可以在 WSL2 Ubuntu 或原生 Linux 中完整运行。

这不是 GPU 实验，也不需要租卡。它验证的是分布式程序的**正确性门**：多个 CPU 进程能否
启动、交换张量、得到一致更新，并在错误配置出现时正确失败。

## 1. 先理解 `passed`、`skipped` 和 `unavailable`

| 状态 | 含义 | 应怎样处理 |
|---|---|---|
| `passed` | 测试执行了，断言全部成立 | 可以进入下一层验证 |
| `skipped` | 当前平台不适合运行这项测试，测试没有执行 | 到声明支持的平台补跑 |
| `failed` | 测试执行后违反了预期 | 保留日志并排查，不能忽略 |
| capability `unavailable` | 程序正常探测到当前后端或版本不支持某能力 | 如实记录边界；它不等于测试框架失败 |

最后一项尤其容易混淆。例如，某个 PyTorch 版本可能不支持 CPU/Gloo 上的 FSDP2 reduction
语义。capability 测试可以正确识别并记录这个限制，因此 pytest 仍可能通过；这不代表 FSDP2
在该后端已经可用。CUDA/NCCL 能力必须由单独的 GPU gate 判断。

## 2. 这 6 项测试验证什么

| 模块 | 项数 | 验证内容 |
|---|---:|---|
| 05 · Tiny Collective | 1 | 2/3/4 rank centralized 与 ring AllReduce correctness matrix |
| 06 · Training Engine | 2 | 五种 reducer 的全局 batch 对齐；错误 bucket plan 是否快速失败 |
| 07 · Parallelism | 3 | 教学 TP、原生 TP，以及 FSDP2 CPU capability 与 checkpoint 证据 |

这里的一“项”可能在内部启动很多 case。看到 `6 passed` 不表示只比较了 6 个张量，而表示
6 个集成测试入口及其内部矩阵全部满足验收条件。

## 3. 准备 WSL2 CPU 环境

如果尚未安装 WSL2，先完成 [WSL2 Ubuntu 教程](wsl2-gpu.md)。只有 CPU 也可以执行本文，
不需要安装 CUDA Toolkit 或 Linux NVIDIA 显示驱动。

下面所有命令都在 **Ubuntu Terminal** 中运行。推荐把仓库放在 Linux 文件系统中：

```bash
cd ~/projects/TrainScale_Lab
uv sync --extra cpu --extra dev --python 3.11
```

不要复用 Windows 的 `.venv`。虚拟环境包含平台相关的可执行文件和 wheel，Windows 与 Linux
必须分别创建。`uv sync` 会按项目锁定关系安装 CPU 版 PyTorch 和测试工具。

先检查解释器和 Gloo：

```bash
.venv/bin/python - <<'PY'
import sys
import torch
import torch.distributed as dist

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gloo available:", dist.is_gloo_available())
PY
```

CPU 路线中 `cuda available: False` 是正常现象；本文只要求 `gloo available: True`。

## 4. 单独运行原先跳过的测试

仍在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp /tmp/trainscale-gloo-tests \
  05_tiny_collective/tests/test_module05_gloo_integration.py \
  06_training_engine/tests/test_module06_gloo_integration.py \
  07_parallelism/tests/test_module07_gloo_integration.py
```

这些测试会反复启动 CPU 多进程，几分钟没有新输出不一定是卡死。先等待 pytest 给出最终状态，
不要因为终端暂时安静就连续启动第二份相同任务。成功时末尾应类似：

```text
......                                                                   [100%]
6 passed in ...s
```

耗时取决于 CPU、WSL 状态和 PyTorch 版本，不应把参考耗时当作性能基线。

## 5. 再运行完整的 04–07 本地测试

单独的 Linux/Gloo gate 通过后，可在同一个 Ubuntu 环境执行完整回归：

```bash
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  04_nccl_benchmark/tests \
  05_tiny_collective/tests \
  06_training_engine/tests \
  07_parallelism/tests
```

04 的真实 NCCL benchmark 仍然属于 GPU 实验；这里运行的是 04 的本地规划、解析和验收逻辑。
不要因为 CPU pytest 通过，就宣称 NCCL 性能已经验证。

## 6. 怎样阅读失败

- `No module named pytest`：没有安装开发依赖，重新执行 `uv sync --extra cpu --extra dev`；
- `Gloo available: False`：当前 PyTorch build 不含 Gloo，应使用项目配置提供的 CPU wheel；
- 子进程超时：先保留完整 traceback，确认没有另一份相同测试在运行，再换新的
  `--basetemp` 目录重试；
- FSDP2 artifact 为 `unavailable`：查看 artifact 中的 rank、后端和错误证据，再判断它是否是
  已知的 CPU/Gloo 版本限制；不要修改 JSON 把它写成 `success`；
- WSL 测试通过但 GPU gate 失败：分别检查 CUDA/NCCL 环境，Gloo 正确性不能替代 GPU 能力。

## 7. 本项目的已验证参考

2026-08-29，本仓库在 Ubuntu WSL2、Python 3.11.16、PyTorch 2.12.1+cpu、Gloo 可用的环境中，
上述 6 项测试得到：

```text
6 passed in 237.70s
```

这个记录证明测试路径可运行，不承诺所有机器耗时相同。学习者真正需要复现的是“6 项均执行并
符合断言”，而不是精确复刻 237.70 秒。

完成后回到 [文档总导航](../README.md)，继续正式的 04–07 correctness、GPU benchmark 和
Profiler 路线。
