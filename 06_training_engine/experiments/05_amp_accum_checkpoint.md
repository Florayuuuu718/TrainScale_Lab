# AMP、Accumulation 与 Checkpoint

- accumulation 只在最后一个 micro-step 同步；loss 按 micro-step 数归一；
- AMP 在 backward 时同步 scaled gradient，所有 handle 完成后再 `unscale_` 和 clip；
- GradScaler 检测 overflow 时跳过 optimizer step，结果 artifact 记录 skip；
- checkpoint 直接复用 01 的 schema、原子写入、optimizer/scaler/RNG 恢复语义。

本地已验证 accumulation 1/2 的 global-batch 等价和 checkpoint 恢复后的下一步一致。FP16 AMP、
overflow/skip 与 accumulation 4 的多 GPU 行为属于正式 GPU gate。独立 overflow runner 在 reducer
完成后、`unscale_` 前注入非有限梯度，覆盖 2/4 卡 bucket-async 与 DDP；它验证控制流，不代表自然
overflow 发生率。
