# 实验 04：一次冻结的多 GPU 采集

推荐租单机 4×同型号、每卡至少 12 GiB 的 NVIDIA GPU。2 GPU 能完成四种 collective
主曲线，但无法完成 pair02/world4 和 4-rank DDP bridge。

顺序固定为：

1. 环境探针、GPU list、topology；
2. 构建固定 commit 的 nccl-tests；
3. smoke；
4. formal run 1/2/3；
5. DDP bridge；
6. 检查每个 artifact 的 status、commit、dirty、config hash；
7. 打包 raw、计算 SHA-256、下载并在本地复核；
8. 关机止费；
9. 本地运行 `aggregate_runs.py` 生成中位数摘要。

任何 smoke failure 都应先停止正式测量。环境不一致的三次运行不能混合聚合，缺失的
4 GPU case 必须保留 `unavailable`。

