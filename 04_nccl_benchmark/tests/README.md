# 04 Tests

```powershell
.venv\Scripts\python -m pytest -q 04_nccl_benchmark/tests
```

CPU CI 覆盖公共 artifact/status/hash、严格 TOML、固定构建版本、命令参数、四种
collective 的 busbw 公式、官方表格解析、错误计数、三次聚合、DDP payload 和
torchrun 命令。真实 NCCL binary、2/4 GPU collective 和 CUDA timeline 是独立的
租卡 gate，不在普通 CPU CI 中伪装执行。

