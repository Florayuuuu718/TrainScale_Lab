# GPU Reducer Ablation

正式矩阵保持一次只改变一个变量：

- 主线：2/4 卡、五种 strategy、medium、FP32、accumulation 1、10.008789 MiB bucket；
- 模型规模：4 卡 bucket async/DDP，small 对 medium；
- bucket：4 卡 medium bucket async，1/10.008789/25 MiB；
- precision/accumulation：4 卡 medium bucket async/DDP，FP32/AMP × 1/4。

去重后 20 个条件，每个独立运行 3 次。报告使用吞吐中位数、相对极差、slowest-rank step p50/p95、
peak allocated memory、collective 数和 payload。每个 job 在计时前先执行 global-batch correctness。
