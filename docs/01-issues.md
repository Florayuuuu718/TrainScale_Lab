# 01 · PyTorch Training 验收清单

以下条目可直接转成 GitHub Issues；02 必须等全部 01 验收项关闭后再开始。

| ID | 标题 | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|
| 01-P01 | 初始化仓库、License 与 Python 3.11 环境 | 无 | `main` 分支、Apache-2.0、`.venv` 与安装命令可复现 | 已完成 |
| 01-P02 | 建立包、lint、pytest 与 CPU CI | 01-P01 | `ruff check .`、`pytest` 通过；PR 触发 CPU CI | 已完成 |
| 01-01 | Synthetic dataset 与单 batch overfit | 01-P02 | 离线数据可复现，测试达到 100% batch accuracy | 已完成 |
| 01-02 | 最小 FP32 train/validation loop | 01-01 | 样本加权 loss/accuracy；train/eval mode 测试通过 | 已完成 |
| 01-03 | 完整 checkpoint/resume | 01-02 | model/optimizer/scheduler/scaler/epoch/step/RNG/config/metrics 可恢复 | 已完成 |
| 01-04 | DataLoader workers 吞吐实验 | 01-01 | 固定变量、重复测量、保存原始 JSON 并写出结论 | 已完成 |
| 01-05 | CIFAR-10 数据、CNN 与配置驱动训练 | 01-02 | 子集训练收敛；配置、日志、曲线可追溯 | 已完成 |
| 01-06 | AMP、累积、scheduler 与消融 | 01-03 | 正确性测试；FP32/AMP/compile 吞吐、显存与正确性落盘 | 已完成 |
| 01-07 | CPU/CUDA Profiler | 01-05 | CPU activity 可解释；CUDA device time 通过验收 | 已完成 |
| 01-08 | v0.1 发布验收 | 01-01..07 | 新环境 smoke run；本地质量门全绿；push 后确认 GitHub CI | 本地完成，远端待 push |

## 暂不进入 v0.1 的工作

CUDA/Triton 自定义 kernel 属于 02，DDP 与 NCCL 分别属于 03 和 04。01 对 `torch.compile` 的调用仅作为训练消融；当前 Windows Triton 缺失已经记录，不提前开展自定义 kernel。
