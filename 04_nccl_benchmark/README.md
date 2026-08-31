# 04 · NCCL Performance Lab

> 交互式入口：[04 · NCCL Latency and Bandwidth](../notebooks/04_nccl_latency_bandwidth.ipynb)

> 状态：已完成。4×RTX 4090 D 的 collective、拓扑、DDP bridge、长窗口 scaling 和
> NCCL 策略延伸实验均已校验；scaling correctness 通过，但测量稳定性限制如实保留。

04 不把 `nccl-tests` 当作孤立跑分工具。它从 03 已归档的真实 1/2/4 GPU DDP
结果出发，在同一主机、同一软件环境中测量 collective 曲线，并用 GPU timeline
解释为什么小模型 strong scaling 变慢，而 weak scaling 能接近但达不到理想线性增长。

## 开始前先建立联系

03 已经告诉我们“DDP 没有线性加速”，但只看端到端吞吐还不知道时间花在哪里。04 把训练
拆成消息大小、collective 类型和拓扑路径，建立通信上限，再把真实梯度 payload 放回曲线。
如果 `rank`、AllReduce、strong/weak scaling 仍不熟悉，先查
[术语表](../docs/concepts/distributed-systems-glossary.md) 或复习
[03](../03_distributed_training/README.md)。

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

## 建议学习顺序

1. 先回答上面的五个问题，并把预测写下来；
2. 按 [实验导航](experiments/README.md) 从协议、曲线、拓扑走到 DDP bridge；
3. 本地运行解析器与 artifact tests；需要真实曲线时再走
   [JupyterLab 四卡教程](../docs/getting-started/jupyterlab-4gpu.md)；
4. 实验后再读 [最终报告](experiments/06_final_report.md)，用
   [机器可读摘要](results/module04_final_summary.json) 对照结果，不要求绝对数字相同。

## 不租卡也能先完成什么

以下命令不需要多 GPU，并且不会伪造性能数据：

```powershell
.venv\Scripts\python 04_nccl_benchmark\benchmarks\check_environment.py

.venv\Scripts\python 04_nccl_benchmark\benchmarks\plan_ddp_bridge.py `
  --config 04_nccl_benchmark\configs\ddp_bridge.toml

.venv\Scripts\python -m pytest -q 04_nccl_benchmark\tests
```

这些命令验证环境探针、配置、公式、parser 和 10 MiB payload 推导，不产生多 GPU 性能值。
`nccl-tests` 的真实构建与执行要求 Linux/WSL；不要在 Windows PowerShell 中照抄 `/tmp` 路径。

真实 Linux 多 GPU 命令见 [环境与租卡门](ENVIRONMENT.md)、
[实验导航](experiments/README.md) 和 [JupyterLab 一站式教程](../docs/getting-started/jupyterlab-4gpu.md)。

## 实验实施顺序

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

## 已获得的最终证据

- 2/4 GPU 四类 collective 的 message size–latency/algbw/busbw 曲线；
- PHB/NODE/SYS pair 对照，但差异不足以宣称稳定拓扑排序；
- 10,494,976-byte 梯度 payload 与 25 MiB bucket 的曲线映射；
- DDP correctness/timeline，以及五次长窗口 strong/weak scaling；
- Auto/Ring/Tree 与 Simple/LL 的诊断性延伸对照；
- 999 个主归档文件、36 个延伸作业和外层归档 SHA-256 校验。

最重要的边界是：long campaign 的 measurement quality 未过稳定性门。中位数可用于
定性解释，不能包装成高精度扩展效率。这是可信实验的一部分，不是未完成。

## 范围边界

04 不实现新的 collective 算法，不通读 NCCL 全部源码，也不把 8 GPU、NVLink 或
多节点作为完成门槛。它提供测量事实；算法内部机制留给 05，训练中的 bucket 与
overlap 留给 06。

进入本模块前，请先完成 [03 · Distributed Training](../03_distributed_training/README.md)。
逐项开发与验收见 [04 验收清单](../docs/04-issues.md)。

学完本章后进入 [05 · TinyCollective](../05_tiny_collective/README.md)：04 告诉你 AllReduce
表现怎样，05 会亲手拆开它的数据流和通信轮次。
