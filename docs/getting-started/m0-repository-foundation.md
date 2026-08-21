# M0 仓库基建：每个文件解决什么问题

M0 不训练模型。它建立可复现、可测试、可协作的项目边界，让后面的实验结果值得相信。

## 核心文件

| 文件 | 作用 | 如果缺少会怎样 |
|---|---|---|
| `.python-version` | 声明 Python 3.11 | 不同机器可能使用不同 Python |
| `pyproject.toml` | 定义包、依赖、命令和工具配置 | 安装过程只能依赖口头说明 |
| `uv.lock` | 锁定依赖解析结果和 wheel 来源 | 同一命令在不同日期可能装出不同环境 |
| `.gitignore` | 排除环境、缓存、checkpoint 和实验原始产物 | 仓库会混入机器相关大文件 |
| `LICENSE` | Apache-2.0 使用与分发规则 | 代码的法律使用边界不明确 |
| `.github/workflows/cpu-ci.yml` | 每次 push/PR 自动 lint 和测试 | 本机通过不代表提交后仍然通过 |
| `01_pytorch_training/trainscale_training/` | M1 可安装 Python 包与源码 | 阶段代码和学习入口会分离 |
| `01_pytorch_training/tests/` | 可重复执行的正确性证据 | 修改代码后无法快速发现回归 |

## 为什么同时锁 CPU 和 CUDA wheel

`pyproject.toml` 定义两个互斥 extra：

- `cpu`：安装 `torch==2.11.0+cpu`，供 CPU CI 和无 GPU 学习者使用；
- `cu128`：安装 `torch==2.11.0+cu128`，供 NVIDIA GPU 环境使用。

两者共享 PyTorch 2.11 API，但二进制运行库不同。互斥规则防止同一环境同时请求两个 torch 变体。`uv.lock` 记录两条可复现解析路径，实际安装时必须明确选择一条。

## 从零验收 M0

```powershell
git status --short --branch
uv venv --python 3.11 .venv
uv sync --extra cpu --extra dev
.venv\Scripts\python --version
.venv\Scripts\ruff check .
.venv\Scripts\pytest -q
```

检查点：

- Git 分支应为 `main`；
- Python 应为 3.11；
- ruff 应无 lint 错误；
- pytest 应全部通过；
- `git status` 不应显示 `.venv`、checkpoint、results 或 uv cache。

## CPU CI 在做什么

GitHub Actions 使用全新的 Ubuntu runner：

1. checkout 当前提交；
2. 安装固定的 uv；
3. 根据 `uv.lock` 安装 Python 3.11、CPU PyTorch 和开发依赖；
4. 执行 `ruff check .`；
5. 执行 `pytest`。

CI 的价值是模拟“没有你本机历史环境的新机器”。如果本机通过而 CI 失败，常见原因是遗漏依赖、大小写路径、平台差异或未提交必要文件。

## M0 当前边界

本地 Git 已初始化；GitHub 仓库和远端 CI 需要首次创建、push 后才能完成外部验收。M0 不负责模型效果、GPU kernel 或分布式训练。
