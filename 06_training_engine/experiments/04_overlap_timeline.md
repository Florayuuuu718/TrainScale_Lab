# Overlap Timeline

在 4 卡 medium 模型上固定同一 batch、bucket 和 FP32，分别采集 bucket sync、bucket async、DDP
的 CPU+CUDA Chrome trace。正式判定必须检查：

1. backward CUDA kernel 的时间区间；
2. NCCL kernel 的时间区间；
3. 二者交集时长，而非只比较 API launch 时间；
4. optimizer 是否严格晚于最后一个 collective completion。

若 4090D 节点 P2P 受限，应联合 04 的 SHM/transport 证据解释 overlap 和吞吐，不能外推到
NVLink 或多节点环境。
