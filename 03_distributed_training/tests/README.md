# 03 测试说明

测试分三层：

1. Windows/普通 CPU CI：TOML、sampler coverage、scaling 公式、launcher 命令和
   结果阅读器；
2. Linux CPU 集成：真实启动 2 rank Gloo，检查公开 runner 的 AllReduce/Broadcast；
3. WSL/GPU 本地验收：NCCL world=1 与可用多 GPU world size。

```bash
# 普通 CPU 契约测试
.venv/bin/python -m pytest -q 03_distributed_training/tests

# 公开 runner 的真实两 rank语义实验
.venv/bin/python 03_distributed_training/benchmarks/run_correctness.py \
  --experiment semantics --world-size 2 \
  --output 03_distributed_training/results/raw/tutorial/test_semantics.json
```

Windows 会跳过 Linux torchrun 集成测试，但不会把 skip 记成 passed；GitHub Linux
CPU CI 会真实运行。性能数值不设跨机器阈值，correctness 和公式才是普通 CI 门。

| 文件 | 覆盖 |
|---|---|
| `test_contract.py` | 配置、sampler coverage/padding、speedup/efficiency |
| `test_launcher.py` | torchrun standalone 和 nproc 参数 |
| `test_module03_show_results.py` | unavailable 不被打印成伪造吞吐 |
| `test_gloo_integration.py` | Linux 真实两 rank Gloo 公开入口 |
| `test_module03_results.py` | 正式正确性/scaling/profile 与 SHA-256 总门 |
