# TrainScale Lab 交互式 Notebook 学习层设计

## 1. 定位

Notebook 是现有教程的**交互式学习层**，不是第二套项目实现，也不取代模块 README、正式
runner、测试或实验报告。

四类材料的职责固定如下：

| 层次 | 主要职责 | 是否是事实来源 |
|---|---|---|
| Markdown 教程 | 建立知识体系、解释实验设计和结论边界 | 是 |
| Notebook | 做预测、观察中间量、调用实验、绘图和练习 | 否，引用教程与 artifact |
| Python runner/tests | 执行训练、通信和正确性验证 | 是 |
| JSON/trace/report | 保存证据并形成最终解释 | 是 |

Notebook 不应把现有 `.py` 代码复制进单元格。核心逻辑发生变化时，只修改正式实现；Notebook
通过导入函数或启动独立子进程继续使用它。

## 2. 面向谁、解决什么问题

目标学习者是第一次接触 ML Systems、AI Infrastructure 或分布式训练的人。Notebook 主要
降低以下门槛：

- 不知道一章应该先观察什么；
- 能运行命令，但不会解释 JSON、Profiler 或性能曲线；
- 暂时没有 GPU，仍希望完成概念学习和结果分析；
- 有 GPU，但容易把 kernel 状态、正式 runner 和临时尝试混在一起；
- 看懂局部代码后，不知道它与前后章节有什么关系。

Notebook 不负责隐藏所有复杂性。rank、collective、bucket、sharding 等关键概念仍会明确展示，
但每次只引入一个主要变量。

## 3. 三种运行模式

每本 Notebook 的第一个可编辑配置单元统一提供：

```python
MODE = "reference"  # reference | local | gpu
```

| 模式 | 硬件要求 | 会做什么 | 适合谁 |
|---|---|---|---|
| `reference` | 普通 CPU | 读取仓库内正式摘要、绘图、完成练习 | 第一次阅读或没有 GPU |
| `local` | CPU，可选单 GPU | 运行轻量 correctness 和最小示例 | 本地开发与概念验证 |
| `gpu` | 对应章节所需 GPU | 调用正式 runner，生成新的临时 artifact | 租卡或具备实验环境 |

“Restart Kernel and Run All” 默认使用 `reference`，因此所有 Notebook 必须能在没有 GPU、
没有原始大文件的环境中从头执行。需要昂贵资源的单元格必须显式检查模式和环境，不能在导入
Notebook 时自动启动。

## 4. 计划中的文件架构

```text
notebooks/
├── README.md
├── 00_start_here.ipynb
├── 01_reliable_training_loop.ipynb
├── 02_gpu_kernel_reasoning.ipynb
├── 03_ddp_fundamentals.ipynb
├── 04_nccl_latency_bandwidth.ipynb
├── 05_collective_algorithms.ipynb
├── 06_reducer_bucket_overlap.ipynb
├── 07_parallelism_strategies.ipynb
├── _support/
│   ├── __init__.py
│   ├── context.py
│   ├── runner.py
│   ├── artifacts.py
│   └── plots.py
└── _runs/                       # 本地生成，Git 忽略
    └── <module>/<run-id>/
```

设计原则：

- 一章一本，文件编号与 01–07 主线一致；
- `00_start_here.ipynb` 只教使用方式，不重复 01 的训练内容；
- `_support/` 只放路径、展示、绘图和安全启动子进程的辅助代码，不放训练算法；
- `_runs/` 只存 Notebook 临时输出，不成为正式结果来源；
- 正式参考结果继续保存在各模块 `results/`，正式解释继续保存在 `experiments/`；
- 第一版不增加 solutions 目录，答案使用折叠提示或链接到正式报告，避免维护两套答案。

实现阶段需要在 `.gitignore` 中加入 `notebooks/_runs/`、Notebook checkpoint 和临时导出文件；
不应忽略 `.ipynb` 本身。

## 5. Notebook 与原文档怎样交互

每本 Notebook 与现有材料形成一个闭环：

```text
模块 README / 概念文档
          ↓ 前置阅读
Notebook：预测 → 小观察 → correctness → 正式 runner
          ↓ 读取 JSON / trace 摘要
Notebook：绘图 → 解释 → 练习
          ↓ 对照
最终实验报告 → 下一模块 README
```

具体约束：

1. Notebook 开头链接模块 README、术语表和必要前置章节；
2. 正式报告链接放在“完成预测和实验以后”，避免一开始泄露结论；
3. 模块 README 增加“交互式学习”入口；
4. Notebook 用相对链接返回原文档和下一章；
5. Notebook 读取模块正式摘要，不把参考数字抄成第二份常量；
6. Notebook 启动的实验仍调用模块现有 runner，并将输出写入 `_runs/`；
7. 学习者确认结果完整后，再按模块发布规范形成正式 artifact，而不是直接提交 `_runs/`。

## 6. 每本 Notebook 的固定形式

页面结构保持一致，让初学者把注意力放在知识本身：

1. **本节问题**：用一句话说明要解释的系统现象；
2. **完成后能回答什么**：列出 3–5 个可检查的学习目标；
3. **前置阅读**：链接原教程和术语表，不大段复制；
4. **运行状态卡**：显示项目根目录、模式、Python、PyTorch、GPU 数量和输出目录；
5. **运行前预测**：先选择或填写预期趋势及理由；
6. **最小观察**：用很小的张量或模型暴露关键中间量；
7. **正确性门**：运行测试或读取 correctness artifact；
8. **正式实验**：由独立子进程调用 runner；昂贵实验仅在 `gpu` 模式运行；
9. **结果可视化**：直接解析 JSON，展示曲线、表格或 timeline 摘要；
10. **与预测对照**：要求学习者指出预测成立或被推翻的地方；
11. **一般规律与边界**：链接正式报告后再总结，区分本机观察与普遍结论；
12. **检查题与下一步**：提供可回答的问题和下一章入口。

单元格标签统一使用：

| 标签 | 用途 |
|---|---|
| `setup` | 环境、路径和模式初始化 |
| `prediction` | 学习者先填写的预测 |
| `reference` | 无硬件也可执行的结果分析 |
| `local-run` | CPU 或单 GPU 轻量实验 |
| `gpu-run` | GPU 或多 GPU 正式实验 |
| `analysis` | 绘图和指标解释 |
| `exercise` | 检查题或开放练习 |

第一版不依赖 JupyterLab 扩展或复杂 widgets。模式、参数和预测使用普通 Python 变量与 Markdown，
确保经典 Notebook、JupyterLab、VS Code Notebook 都能打开。

## 7. 八本 Notebook 的内容大纲

### 00 · Start Here

**核心问题**：怎样使用 Notebook，而不破坏项目的可复现性？

- 找到仓库根目录并验证关键文件；
- 认识 `reference/local/gpu` 三种模式；
- 区分 Notebook kernel、Terminal 子进程和分布式 worker；
- 读取一份示例 JSON artifact；
- 理解 `passed/skipped/failed/unavailable`；
- 演示安全运行命令、查看日志和停止后保留失败证据；
- 链接 Windows、WSL2、Linux/Gloo 与四卡 JupyterLab 教程。

产出：一次环境自检和一张“我可以完成哪些实验”的能力表，不做性能结论。

### 01 · Reliable Training Loop

**核心问题**：一次训练为什么值得相信？

- 用极小 batch 展示 forward、loss、backward、optimizer step；
- 检查梯度何时产生、清零和更新；
- single-batch overfit 作为正确性测试；
- 对比 train/eval 与 no-grad；
- 观察 checkpoint 中模型、优化器、step 和 RNG 状态；
- 读取吞吐摘要，解释 warm-up 与中位数；
- 区分“loss 下降”“可恢复”“性能可信”三类证据。

产出：训练 step 状态表、恢复一致性检查和 CPU/GPU 吞吐对照图。

### 02 · GPU Kernel Reasoning

**核心问题**：算子正确以后，为什么仍可能很慢？

- 从 PyTorch reference 到自定义 kernel 的正确性对照；
- 解释绝对/相对误差和 dtype 容差；
- 展示 shape、stride、连续性和访存量；
- 用小规模 sweep 区分 launch-bound 与 bandwidth-bound；
- 对比 PyTorch、CUDA/Triton 的参考结果；
- 读取 Profiler 摘要，而不是在 Notebook 中编译第二份 kernel；
- 解释优化为什么必须由测量验证。

产出：问题尺寸—延迟曲线、有效带宽图和一次优化前后对照。

### 03 · DDP Fundamentals

**核心问题**：多个进程怎样共同训练一个模型而不改变数学语义？

- 可视化 rank、local rank、world size 和 process group；
- 展示 DistributedSampler 如何切分数据；
- 对照单进程 global batch 与多 rank 梯度；
- 解释 AllReduce 后为什么参数保持一致；
- strong/weak scaling 的预测和计算；
- 读取 DDP timeline，定位计算、同步和等待；
- 讨论 checkpoint 由谁写、如何恢复。

产出：数据覆盖表、梯度/参数误差检查、1/2/4 rank scaling 图。

### 04 · NCCL Latency and Bandwidth

**核心问题**：DDP 通信成本怎样随消息大小、GPU 数量和拓扑变化？

- 用 `T(n) ≈ α + n/β` 建立延迟—带宽直觉；
- 读取 `nvidia-smi topo -m` 并标注近/远 GPU pair；
- 区分 algorithm bandwidth 与 bus bandwidth；
- 绘制 nccl-tests 消息大小曲线，分别分析小消息和大消息平台区；
- 将 Module 03 模型梯度 payload 映射到曲线位置；
- 对照近/远双卡，不用所有消息平均值夸大拓扑影响；
- 分析 DDP scaling 波动和 measurement-quality gate；
- 扩展：NCCL algorithm/protocol 对比。

产出：latency/busbw 双图、payload 标记、拓扑结论与测量质量说明。

### 05 · Collective Algorithms

**核心问题**：centralized、ring 和 NCCL 为什么表现不同？

- 用小数组逐轮展示 centralized reduce+broadcast；
- 用 chunk 动画或表格展示 ring reduce-scatter+all-gather；
- 对非整除长度进行 padding、切片和 correctness 检查；
- 推导每个 rank 的通信量与轮数；
- 先预测 2/4 rank、小/大消息的相对表现；
- 读取 GPU comparison artifact，对照教学实现与 `torch.distributed`；
- 解释教学实现为何能帮助理解、但不能替代 NCCL。

产出：逐轮状态表、通信量推导、算法—消息大小性能对照图。

### 06 · Reducer, Bucket and Overlap

**核心问题**：collective 放回 backward 后，怎样减少等待而不改变梯度？

- 依次观察 bulk、per-parameter、bucket-sync、bucket-async 和 DDP；
- 可视化参数 readiness 与 bucket plan；
- 用 global-batch reference 检查 reducer 数学；
- 展示 accumulation 与 `no_sync` 的语义；
- 读取 bucket-size 消融并区分调度开销与大 bucket 等待；
- 从 trace 摘要判断“异步发起”是否形成真实 overlap；
- 解释 AMP loss scale、overflow 和跨 rank 一致决策；
- 将吞吐结论与 overlap 机制证据分开。

产出：reducer 对照表、bucket timeline、吞吐消融图和 AMP 状态表。

### 07 · Parallelism Strategies

**核心问题**：DDP、FSDP2 和 TP 分别切分什么，代价是什么？

- 从参数、梯度、优化器状态推导内存构成；
- 图示 DDP replication、FSDP2 sharding 和 TP layer partition；
- 查看 DeviceMesh、DTensor placement 与局部 shape；
- 对照一步参数更新和 checkpoint resume correctness；
- 绘制 2/4 GPU 吞吐与峰值显存；
- 读取 collective profile，解释省显存为何可能增加通信；
- 区分 CPU/Gloo capability 与 CUDA/NCCL gate；
- 用模型大小、互联和目标指标完成一次策略选择练习。

产出：内存预算图、策略权衡表、collective 组成图和选择理由。

## 8. `_support` 的边界

辅助模块只提供 Notebook 通用体验：

### `context.py`

- 从当前目录向上查找 `pyproject.toml`，定位仓库根；
- 读取 `TRAINSCALE_NOTEBOOK_MODE`，允许 CI 强制 reference 模式；
- 创建唯一 run-id 和 `_runs/` 输出目录；
- 输出 Python、PyTorch、平台、GPU 和 Git commit 状态卡。

### `runner.py`

- 使用 `subprocess.run()` 参数列表，而不是拼接 shell 字符串；
- 设置超时、工作目录和日志文件；
- 显示命令、返回码与 stdout/stderr 尾部；
- 失败时保留输出并停止依赖该结果的分析；
- 多进程实验始终在 Notebook kernel 外执行。

### `artifacts.py`

- 以 UTF-8 读取 JSON；
- 检查 `schema_version`、`artifact_type`、`status` 和必要字段；
- 明确区分正式参考 artifact 与本次临时运行；
- 为表格和绘图提供小型、只读的数据选择函数。

### `plots.py`

- 统一颜色、单位、坐标轴和误差/波动标记；
- 小消息延迟使用合适的对数坐标；
- 不截断坐标轴制造夸张差异；
- 图题包含硬件/模式或清楚注明“仓库参考结果”。

辅助模块不能包含模型、collective、reducer、FSDP 或 TP 的第二份实现。

## 9. 输出与证据管理

Notebook 页面中的结果分三类：

| 类型 | 保存位置 | 是否提交 Git |
|---|---|---|
| 正式参考摘要 | `<module>/results/*_summary.json` | 是 |
| 本次临时运行 | `notebooks/_runs/<module>/<run-id>/` | 否 |
| Notebook 展示输出 | `.ipynb` cell output | 默认清除，仅保留少量静态示例 |

规则：

- token、主机公网地址、GPU UUID、用户名和绝对租卡路径不得进入 Notebook；
- checkpoint、trace、rank 日志和大图不嵌入 `.ipynb`；
- 参考模式的图由已提交摘要即时生成，不提交重复 PNG；
- 新实验进入正式报告前，仍需经过模块 correctness、哈希和 acceptance 流程；
- Notebook 不修改已提交的正式摘要；所有新输出先进入 `_runs/`；
- 失败结果不得被下一次运行静默覆盖。

## 10. 初学者界面与文字规范

- 每个新术语第一次出现时用一句自然语言解释，并链接术语表；
- 每个代码单元前说明“为什么运行”，单元后说明“应该看到什么”；
- 不要求学习者从长 stdout 中自行寻找关键行；
- 图表同时给单位、样本数、中位数和波动，不只展示最好值；
- 参考结果与本机结果使用不同样式，禁止混在同一列却不标来源；
- 使用“观察到”“支持”“不能说明”等准确措辞，不把相关性写成因果；
- 每章至少有一个会推翻直觉的对照，并解释原因；
- 每章结尾回到系统层问题，而不是停在“代码运行成功”。

## 11. 可复现性与自动验收

Notebook 实现后必须同时满足：

1. 所有 `.ipynb` 是合法 nbformat，包含稳定 cell id；
2. 在 CPU reference 模式下均能 Restart Kernel and Run All；
3. GPU 单元有显式 guard，在 reference/local 模式不会误启动；
4. 不包含本机绝对路径、凭据、GPU UUID 或大段原始日志；
5. 所有 Markdown 相对链接有效；
6. 所有 runner 使用已有配置和 Python 入口，不复制命令实现；
7. 读取 artifact 前检查状态与 schema，不把缺失字段当成 0；
8. 图表脚本接受参考 artifact 与新 artifact，不能只适配一份手写数据；
9. 提交前清除临时输出，保留经过选择的小型教学输出；
10. CI 使用 `nbformat` 做结构检查，并用 `nbclient` 执行 reference smoke test；
11. 04–07 GPU 模式至少在正式租卡环境完整 Run All 一次；
12. Notebook 中引用的结论与对应最终报告一致。

建议实现时在 `pyproject.toml` 增加独立的 `notebook` optional dependency，只包含 JupyterLab、
ipykernel、matplotlib、nbformat 和 nbclient 等展示/验收工具，不混入新的训练框架。

## 12. 实施顺序

### Phase 1：纵向样板

先实现 `_support`、`00_start_here.ipynb` 和 `04_nccl_latency_bandwidth.ipynb`。04 同时覆盖
参考数据、GPU runner、曲线、拓扑和结论边界，最适合检验整体架构是否成立。

### Phase 2：完成 04–07 交互层

依次实现 05、06、07，并在无 GPU reference 模式执行全部 Notebook。之后集中租用 4 GPU，
验证 GPU mode、输出目录、错误提示和 Restart/Run All。

### Phase 3：补齐 01–03

沿用已经稳定的页面模板补齐训练、kernel 和 DDP。02 的编译/GPU 单元必须继续调用正式实现，
03 的多进程程序必须继续在 kernel 外启动。

### Phase 4：发布验收

- 给模块 README 和文档导航加入正式入口；
- 执行 Notebook 结构、reference smoke、链接和敏感信息检查；
- 在四卡 JupyterLab 教程中加入“打开 Notebook 学习”和“Terminal 正式运行”的分工说明；
- 记录首个可复现版本，不承诺跨硬件完全相同的性能数字。

第一版不加入 08/综合大作业、实时 dashboard、复杂 widgets 或自动调参。先保证 00–07 内容
准确、可从头运行、与正式文档一致，再决定是否扩展。

## 13. 设计完成的判断标准

这层 Notebook 成功，不是因为页面看起来丰富，而是学习者能够：

- 没有 GPU 时读取参考证据并完成核心推理；
- 有 GPU 时复用同一套 runner 得到自己的结果；
- 明确知道哪些内容来自本机，哪些来自仓库参考；
- 从曲线和中间量解释系统现象，而不是只复制最终结论；
- 随时回到原教程、源码、artifact 和下一章节；
- 重启 kernel 后仍能按照同样顺序完成实验。

满足这些条件后，Notebook 才真正增强 TrainScale Lab 的学习性，而不是成为另一组难以维护的
演示文件。
