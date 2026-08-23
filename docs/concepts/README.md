# 概念文档

概念文档回答“为什么”，阶段 README 回答“怎么运行”，源码回答“具体怎么实现”。

当前 01 模块概念：

- [PyTorch 训练基础](pytorch-training-basics.md)：synthetic dataset、DataLoader、batch、模型、loss、梯度、optimizer、epoch、validation 和 overfit test。
- [checkpoint 完整状态](../checkpoint-contract.md)：保存与恢复一次训练需要哪些状态。

建议先运行一次 CPU smoke test，再阅读概念文档，然后打开对应源码对照。只读概念而不运行代码，很难形成训练系统的直觉。
