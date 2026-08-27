# AutoDL 4×RTX 4090D 原始证据

这些文件来自 2026-08-26 的单机四卡租用实验。实验在干净提交
`d2b2882270a7e5ad8de6f666572497d2b3921703` 上运行，服务器打包文件和 Windows
下载文件均通过以下 SHA-256：

```text
63b0bb1efc17313cfd9df381afe67281d9daa2eb634a4fe570861ca7f3077e54
```

| 文件 | 作用 |
|---|---|
| `environment.json` | Python、PyTorch、CUDA、driver、GPU 数与 backend 能力 |
| `gpu-list.txt` | 四张可见 RTX 4090D 的 `nvidia-smi -L` 输出 |
| `gpu-topology.txt` | GPU、NUMA 和 NIC 拓扑 |
| `nvidia-smi.txt` | 驱动、显存、功耗上限、空闲状态快照 |
| `gpu_smoke.json` | 1/2 GPU 小模型流程门 |
| `gpu_formal_run1.json` | 第一次正式 1/2/4 GPU strong/weak 运行 |
| `gpu_formal_run2.json` | 第二次正式运行 |
| `gpu_formal_run3.json` | 第三次正式运行 |

不要手工编辑这些源文件。正式中位数
[`scaling_nccl_4x4090d.json`](../../scaling_nccl_4x4090d.json) 由
[`aggregate_scaling_runs.py`](../../../benchmarks/aggregate_scaling_runs.py) 验证并生成；
汇总 JSON 还记录了这里每个文件的独立 SHA-256。

本证据只证明该短时 synthetic MLP 在这台单机四卡实例上的行为。GPU0/1 与 GPU2/3
分属两个 NUMA 域，组间为 `SYS`，且没有 NVLink；不能把绝对吞吐外推到 A100/H100
NVLink 主机或真实生产模型。
