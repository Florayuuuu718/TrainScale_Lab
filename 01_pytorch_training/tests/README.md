# 01 正确性测试

```powershell
.venv\Scripts\pytest -v
.venv\Scripts\pytest 01_pytorch_training/tests/test_training.py -v
.venv\Scripts\pytest 01_pytorch_training/tests/test_checkpoint.py -v
```

当前 10 个测试覆盖：

| 性质 | 能发现的问题 |
|---|---|
| single batch 达到 100% accuracy | forward/loss/backward/optimizer/标签链路断裂 |
| train/eval mode 与样本计数 | 漏 `model.eval()`、最后小 batch 加权错误 |
| 梯度累积等价于有效 batch | loss 缩放或末组 micro-batch 错误 |
| seed 可复现且不同 seed 会变化 | 随机边界失控 |
| CIFAR CNN shape/mode | 输出类别、BatchNorm 模式错误 |
| checkpoint 恢复全部 RNG | resume 后随机轨迹漂移 |
| resumed 下一步等于 continuous | 漏恢复 optimizer/scheduler/RNG |
| CPU AMP 配置被拒绝 | 静默走错精度路径 |
| 未知 TOML 字段被拒绝 | 拼写错误被忽略 |
| 配置驱动运行产生完整产物 | 模块能跑但整体无法串联 |

测试主要在 CPU 上执行，便于快速运行和进入 GitHub CI；CUDA 实验另行验证 GPU 路径。通过只证明上述性质，不代表模型精度优秀。
