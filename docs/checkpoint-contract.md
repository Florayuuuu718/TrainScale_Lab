# M1 checkpoint 完整状态契约

checkpoint 使用 `schema_version=1`，并采用同目录临时文件加原子替换，避免进程中断留下半写文件。

| 字段 | 内容 | 恢复目的 |
|---|---|---|
| `schema_version` | checkpoint schema 版本 | 拒绝静默读取不兼容格式 |
| `epoch`, `global_step` | 训练进度 | 恢复日志、调度和保存节奏 |
| `model` | `model.state_dict()` | 参数与 buffer |
| `optimizer` | `optimizer.state_dict()` | momentum、Adam moments 等 |
| `scheduler` | scheduler state 或 `None` | 学习率轨迹 |
| `scaler` | AMP GradScaler state 或 `None` | 动态 loss scale |
| `rng.python` | Python `random` state | Python 随机操作 |
| `rng.numpy` | NumPy RNG state | NumPy 随机操作 |
| `rng.torch_cpu` | PyTorch CPU RNG | CPU 随机算子 |
| `rng.torch_cuda` | 所有可见 GPU RNG 或 `None` | CUDA 随机算子 |
| `rng.data_generator` | 可选 DataLoader generator state | shuffle/采样顺序 |
| `config` | 本次运行配置快照 | 审计与重现实验 |
| `metrics` | 保存点指标 | 选择 best/last 与报告 |
| `metadata` | PyTorch 版本 | 环境兼容性诊断 |

恢复测试同时覆盖 Python/PyTorch/DataLoader RNG，以及“连续训练下一步”和“保存后恢复下一步”的 loss 与参数一致性。M1 不承诺跨 PyTorch 主版本读取 checkpoint。
