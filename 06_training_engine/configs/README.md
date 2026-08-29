# Module 06 Configs

- `local_baseline.toml`：small 模型固定 batch overfit，证明模型可训练和参数实际更新；
- `local_correctness.toml`：2-rank Gloo、五种策略、accumulation 1/2、unused parameter；
- `gpu_ablation.toml`：2/4 GPU、small/medium、三种 bucket、FP32/AMP、accumulation 1/4。

GPU runner 从这些维度构造 20 个单变量条件，不执行全笛卡尔积。10.0087890625 MiB 对应
04 实测的 10,494,976-byte DDP gradient payload。
