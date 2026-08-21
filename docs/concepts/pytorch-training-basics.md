# PyTorch 训练基础：当前代码到底在学什么

## 1. synthetic dataset 是什么

synthetic 表示数据由程序生成，不依赖下载文件。当前默认生成：

- 特征矩阵 `X`：形状 `[512, 16]`，即 512 个样本，每个样本 16 个浮点特征；
- 隐藏教师矩阵 `W`：形状 `[16, 4]`；
- 4 类分数：`X @ W`，形状 `[512, 4]`；
- 标签 `y`：每行分数最大值所在的类别，即 `argmax(X @ W)`。

可以把每个样本想成一行包含 16 个测量值的表格。隐藏教师根据这些测量值把样本分成 4 类。训练模型看不到教师矩阵，只能通过输入和标签反推出分类规律。

固定 seed 后，`X`、`W` 和标签都固定，因此每次运行使用相同数据，便于复现和排错。

## 2. Dataset、DataLoader 和 batch

- `Dataset` 定义“第 i 个样本是什么”以及“一共有多少样本”。
- `DataLoader` 负责把多个样本组成 batch，并控制 shuffle 和 workers。
- batch 是模型一次同时处理的一组样本。

默认训练集 400 个样本，batch size 64，因此每个 epoch 有 `ceil(400/64)=7` 个训练 step。前 6 个 batch 各 64 个样本，最后一个 batch 有 16 个样本。

## 3. 模型输出是什么

当前 MLP：

```text
[batch, 16]
    -> Linear(16, 32)
    -> ReLU
    -> Linear(32, 4)
    -> [batch, 4] logits
```

logits 是每个类别的原始分数，不是概率。`CrossEntropyLoss` 内部会进行适合分类的归一化和负对数似然计算，因此模型最后一层不需要手动加 Softmax。

## 4. 一次训练 step 发生了什么

```text
batch
  -> model(features)          前向计算 logits
  -> criterion(logits, y)     计算 loss
  -> loss.backward()          计算每个参数的梯度
  -> optimizer.step()         根据梯度更新参数
  -> optimizer.zero_grad()    下一步前清理旧梯度
```

loss 是模型当前错误程度的可优化标量。梯度表示参数朝哪个方向变化会影响 loss。SGD 根据梯度更新参数，重复许多 step 后，模型逐渐接近生成标签的规律。

当前实现是在每一步开始调用 `zero_grad(set_to_none=True)`，效果仍是确保本 step 不意外累加上一步梯度。

## 5. epoch、step 和 global step

- 一个 batch 的参数更新称为一个训练 step；
- 完整遍历一次训练集称为一个 epoch；
- global step 是从训练开始累计的 step 数。

默认每 epoch 7 steps，运行 3 epochs 后 checkpoint 中的 `global_step` 是 21。

## 6. 训练与验证为什么分开

训练阶段：

- `model.train()`；
- 记录梯度；
- 调用 backward 和 optimizer；
- 参数会改变。

验证阶段：

- `model.eval()`；
- `torch.inference_mode()` 禁止构建梯度图；
- 不调用 backward 和 optimizer；
- 参数不改变。

当前模型没有 Dropout 或 BatchNorm，但仍测试 train/eval 切换，因为后续模型会依赖这个状态。如果现在忽略，复杂模型中会产生难以发现的指标错误。

## 7. 如何读 loss 和 accuracy

- `train_loss`：模型在训练数据上的平均错误程度；
- `valid_loss`：模型在未参与参数更新的验证数据上的平均错误程度；
- `valid_accuracy`：验证集中预测类别正确的比例。

smoke run 中希望看到 loss 总体下降、accuracy 总体上升。这只证明训练链路能学习，不代表模型已经完成充分训练，也不能与真实数据集准确率比较。

如果 train loss 下降而 valid loss 长期上升，通常意味着过拟合；但当前 3 epochs 的目的只是快速正确性检查。

## 8. 为什么要做 single-batch overfit test

测试把 16 个样本作为唯一 batch，重复训练 100 次。数据极小、模型容量足够时，模型应该能够完全记住这 16 个标签。

验收条件：

- 最终 loss 小于初始 loss 的 10%；
- 最终 accuracy 为 100%。

如果连一个固定小 batch 都无法过拟合，问题通常不在“模型不够强”，而在训练代码：可能忘记 backward、optimizer 没有 step、梯度被错误清零、标签形状错误，或参数没有进入 optimizer。

single-batch overfit 证明训练链路有基本学习能力，但不证明泛化能力。

## 9. DataLoader workers 实验与训练测试的区别

训练 smoke test关心模型是否学习；workers 实验关心读取 batch 的吞吐。后者给每个样本人为增加 1 ms 延迟，再改变 `num_workers`，并不更新模型参数。

因此 workers 实验的输出是 samples/s，而不是 loss 或 accuracy。它属于系统性能实验，不属于模型效果评估。

## 10. 当前阶段的知识边界

M1 已覆盖 FP32、数据加载、训练/验证、随机种子、完整 checkpoint/resume、AMP、梯度累积、Profiler、CIFAR-10 和 `torch.compile` 消融。当前 Windows 环境的 CIFAR compile 因缺少可用 Triton 失败并已记录；自定义 CUDA kernel 属于 M2，尚未实现。
