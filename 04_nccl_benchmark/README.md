# 04 · NCCL Performance Lab

> 状态：本地可实现的配置、契约、解析器、runner、DDP bridge 和 CPU 测试已完成；
> 真实 collective 曲线与 timeline 等待一次冻结的多 GPU 实验。本目录是 collective
> 性能测量与 DDP 通信解释的唯一入口。

04 不把 `nccl-tests` 当作孤立跑分工具。它从 03 已归档的真实 1/2/4 GPU DDP
结果出发，在同一主机、同一软件环境中测量 collective 曲线，并用 GPU timeline
解释为什么小模型 strong scaling 变慢、weak scaling 也没有线性增长。

## 本阶段要回答的问题

- 小消息为什么受启动延迟支配，大消息为什么逐渐受链路带宽支配？
- `algbw`、`busbw` 和端到端 step time 分别表示什么？
- GPU/NUMA/PCIe/NVLink 拓扑怎样改变同一种 collective？
- 03 的 DDP 梯度载荷和 bucket 落在通信曲线的哪个区域？
- 单独的通信上限为什么不能直接等同于训练吞吐？

## 目录地图

```text
04_nccl_benchmark/
├── ENVIRONMENT.md                 # Linux/NCCL、固定源码版本与租卡门
├── trainscale_nccl/
│   ├── contract.py                # TOML、命令、busbw 与 DDP payload 契约
│   ├── parser.py                  # nccl-tests stdout 严格解析
│   ├── environment.py             # GPU/NCCL/topology 能力探针
│   └── ddp_bridge_worker.py       # 03 MLP 的 NCCL/DDP Profiler worker
├── benchmarks/
│   ├── check_environment.py       # 本机能力与拓扑 JSON
│   ├── build_nccl_tests.py        # 固定 v2.19.7/commit 的构建助手
│   ├── run_collectives.py         # success/failed/unavailable runner
│   ├── aggregate_runs.py          # 三次一致性校验与中位数聚合
│   ├── plan_ddp_bridge.py         # 本地推导 03 梯度 payload
│   ├── run_ddp_bridge.py          # 2/4 GPU timeline 启动器
│   └── show_results.py            # 初学者结果表
├── configs/                       # smoke、formal 与 DDP bridge 配方
├── tests/                         # 全部可在 CPU CI 执行的契约测试
├── experiments/                   # 预测、命令、解释与租卡闭环
└── results/                       # 正式摘要；大型原始数据进入 ignored raw/
```

## 当前本地入口

以下命令不需要多 GPU，并且不会伪造性能数据：

```powershell
.venv\Scripts\python 04_nccl_benchmark\benchmarks\check_environment.py

.venv\Scripts\python 04_nccl_benchmark\benchmarks\build_nccl_tests.py `
  --source-directory /tmp/nccl-tests

.venv\Scripts\python 04_nccl_benchmark\benchmarks\plan_ddp_bridge.py `
  --config 04_nccl_benchmark\configs\ddp_bridge.toml
```

真实 Linux 多 GPU 命令见 [环境与租卡门](ENVIRONMENT.md) 和
[实验导航](experiments/README.md)。

## 开发顺序

1. 冻结 Linux、GPU、driver、CUDA、NCCL、拓扑和 `nccl-tests` commit；
2. 建立 CPU 可测的配置、解析、公式和结果 schema 测试；
3. 运行 AllReduce、AllGather、ReduceScatter、Broadcast smoke；
4. 扫描小消息到大消息，重点覆盖 1/4/10/25/64 MiB；
5. 比较可控的 GPU pair、world size 和拓扑路径；
6. 在同一环境复跑 03 的代表性 DDP workload，并采集 GPU timeline；
7. 把实际梯度/bucket 大小映射到 collective 曲线，完成单变量解释；
8. 三次正式运行、聚合、归档原始 stdout/CSV/JSON 和 acceptance。

10 MiB 附近不是随意选择：03 当前 MLP 约有 262 万参数，FP32 梯度载荷约为
10 MiB。正式实现必须从模型参数和 dtype 自动计算该值，不能把这个估算硬编码成结论。

## 正确性先于性能

- 命令实际使用预期的 GPU 数和 collective；
- rank 数、消息字节数、dtype 和迭代数进入结构化结果；
- `algbw`/`busbw` 公式用小型单元测试校验；
- stdout 解析失败必须报错，不能静默生成零值；
- 硬件不足记录 `unavailable`，不能伪造多 GPU 数据；
- 不同主机或软件栈的绝对带宽不能混成同一性能排名。

## 最终证据

- message size–latency/algbw/busbw 曲线；
- 至少 2 GPU 的正式证据，4 GPU 为推荐扩展；
- 同 NUMA/跨 NUMA或其他可控拓扑对照；
- 03 workload 的 GPU timeline 和实际 collective 大小；
- collective 曲线与 DDP scaling 现象的对应解释；
- 环境、配置、原始数据哈希、三次重复值和已知限制。

## 范围边界

04 不实现新的 collective 算法，不通读 NCCL 全部源码，也不把 8 GPU、NVLink 或
多节点作为完成门槛。它提供测量事实；算法内部机制留给 05，训练中的 bucket 与
overlap 留给 06。

进入本模块前，请先完成 [03 · Distributed Training](../03_distributed_training/README.md)。
逐项开发与验收见 [04 验收清单](../docs/04-issues.md)。
