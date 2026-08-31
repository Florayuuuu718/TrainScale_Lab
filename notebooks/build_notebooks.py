"""Generate the checked-in notebooks with stable cell ids.

Run this file after editing chapter metadata.  It uses only the standard library so
maintainers do not need Jupyter installed merely to rebuild the JSON documents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def cell_id(chapter: str, name: str) -> str:
    return hashlib.sha1(f"{chapter}:{name}".encode()).hexdigest()[:12]


def markdown(chapter: str, name: str, source: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id(chapter, name),
        "metadata": {"tags": tags or []},
        "source": source.splitlines(keepends=True),
    }


def code(chapter: str, name: str, source: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(chapter, name),
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CHAPTERS: list[dict[str, Any]] = [
    {
        "number": "00",
        "slug": "start_here",
        "title": "Start Here：可靠地使用交互式学习层",
        "question": "怎样使用 Notebook，同时不破坏实验的可复现性？",
        "goals": ["区分 reference/local/gpu 三种模式", "读懂环境状态卡", "知道临时结果和正式证据的边界"],
        "module": "docs",
        "readme": "../docs/README.md",
        "report": "../docs/notebook-learning-layer-design.md",
        "artifact": "04_nccl_benchmark/results/module04_final_summary.json",
        "required": ["schema_version", "artifact_type", "status"],
        "observation": "capabilities = {\n    'reference': '任何能运行 Python 的电脑',\n    'local': 'CPU 或单 GPU correctness',\n    'gpu': '按章节要求的 GPU 正式 runner',\n}\ncapabilities",
        "analysis": "print('示例 artifact：', artifact['artifact_type'])\nprint('状态：', artifact['status'])\nprint('规则：先 correctness，再看性能；unavailable 不等于 0。')",
        "chart": "labels = ['reference', 'local', 'gpu']\nvalues = [1, 2, 3]\nbar_chart(labels, values, title='模式所需资源等级', ylabel='相对资源等级')",
        "local": None,
        "gpu": None,
        "next": "01_reliable_training_loop.ipynb",
    },
    {
        "number": "01",
        "slug": "reliable_training_loop",
        "title": "Reliable Training Loop：为什么这次训练值得相信",
        "question": "loss 下降、可恢复和性能可信，为什么是三类不同证据？",
        "goals": ["识别一次训练 step 的状态变化", "检查 checkpoint 与续训边界", "用中位吞吐而非最快值解释性能"],
        "module": "01_pytorch_training",
        "readme": "../01_pytorch_training/README.md",
        "report": "../01_pytorch_training/experiments/01_synthetic_baseline.md",
        "artifact": "01_pytorch_training/results/summary.json",
        "required": ["training_runs"],
        "observation": "state = {'gradient': None, 'parameter': 1.0}\nfor event in ['forward', 'loss', 'backward', 'step', 'zero_grad']:\n    if event == 'backward': state['gradient'] = 0.25\n    elif event == 'step': state['parameter'] -= 0.1 * state['gradient']\n    elif event == 'zero_grad': state['gradient'] = None\n    print(event, dict(state))",
        "analysis": "runs = artifact['training_runs']\nfor name in ('synthetic_cpu', 'synthetic_cuda'):\n    run = runs[name]\n    print(name, 'loss=', round(run['final_train_loss'], 4), 'samples/s=', round(run['final_train_samples_per_second'], 1))",
        "chart": "runs = artifact['training_runs']\nlabels = ['CPU', 'CUDA']\nvalues = [runs['synthetic_cpu']['final_train_samples_per_second'], runs['synthetic_cuda']['final_train_samples_per_second']]\nbar_chart(labels, values, title='小型合成任务吞吐', ylabel='samples/s')",
        "local": "[sys.executable, '-m', 'trainscale_training.train', '--config', '01_pytorch_training/configs/synthetic_cpu.toml', '--device', 'cpu', '--output-dir', str(ctx.output_dir / 'train')]",
        "gpu": "[sys.executable, '-m', 'trainscale_training.train', '--config', '01_pytorch_training/configs/synthetic_cuda.toml', '--device', 'cuda', '--output-dir', str(ctx.output_dir / 'train')]",
        "next": "02_gpu_kernel_reasoning.ipynb",
    },
    {
        "number": "02",
        "slug": "gpu_kernel_reasoning",
        "title": "GPU Kernel Reasoning：正确以后为什么仍可能很慢",
        "question": "shape、访存与 launch 开销怎样共同决定 kernel 延迟？",
        "goals": ["区分绝对误差与相对误差", "观察 stride 和连续性", "用测量而不是直觉判断优化"],
        "module": "02_gpu_kernels",
        "readme": "../02_gpu_kernels/README.md",
        "report": "../02_gpu_kernels/experiments/08_profiler_roofline.md",
        "artifact": "02_gpu_kernels/results/module02_summary_sm120.json",
        "required": ["schema_version", "required_passes", "cuda_four_way_winners"],
        "observation": "shape = (3, 4)\ncontiguous_stride = (4, 1)\ntransposed_stride = (1, 4)\n{'shape': shape, 'contiguous_stride': contiguous_stride, 'transposed_stride': transposed_stride}",
        "analysis": "winners = artifact['cuda_four_way_winners']\nfor row in winners[:4]:\n    print(row['case_id'], '→', row['fastest_implementation'], round(row['median_us'], 3), 'µs')",
        "chart": "winners = artifact['cuda_four_way_winners'][:4]\nlabels = [row['case_id'].replace('vector_add_', '') for row in winners]\nvalues = [row['median_us'] for row in winners]\nbar_chart(labels, values, title='代表性最快实现延迟', ylabel='median µs')",
        "local": "[sys.executable, '-m', 'pytest', '-q', '02_gpu_kernels/tests/test_references.py']",
        "gpu": "[sys.executable, '02_gpu_kernels/benchmarks/run_triton_comparison.py', '--suite', 'smoke', '--output', str(ctx.output_dir / 'triton-smoke.json')]",
        "next": "03_ddp_fundamentals.ipynb",
    },
    {
        "number": "03",
        "slug": "ddp_fundamentals",
        "title": "DDP Fundamentals：多个进程怎样保持同一数学语义",
        "question": "数据切分与梯度 AllReduce 怎样让各 rank 得到一致参数？",
        "goals": ["解释 rank/world size/process group", "检查 DistributedSampler 覆盖", "区分 strong 与 weak scaling"],
        "module": "03_distributed_training",
        "readme": "../03_distributed_training/README.md",
        "report": "../03_distributed_training/experiments/05_nccl_scaling.md",
        "artifact": "03_distributed_training/results/module03_summary.json",
        "required": ["schema_version", "gates", "all_executable_gates_passed"],
        "observation": "samples = list(range(10))\nworld_size = 3\nshards = {rank: samples[rank::world_size] for rank in range(world_size)}\nshards",
        "analysis": "gates = artifact['gates']\nprint('通过 gate：', sum(gates.values()), '/', len(gates))\nprint('全部可执行 gate 通过：', artifact['all_executable_gates_passed'])",
        "chart": "gates = artifact['gates']\nbar_chart(['passed', 'not passed'], [sum(gates.values()), len(gates)-sum(gates.values())], title='正确性与实验 gate', ylabel='gate 数量')",
        "local": "[sys.executable, '03_distributed_training/benchmarks/run_correctness.py', '--experiment', 'gradient', '--world-size', '2', '--output', str(ctx.output_dir / 'gradient.json')]",
        "gpu": "[sys.executable, '03_distributed_training/benchmarks/run_scaling.py', '--config', '03_distributed_training/configs/gpu_scaling_smoke.toml', '--output', str(ctx.output_dir / 'scaling.json')]",
        "next": "04_nccl_latency_bandwidth.ipynb",
    },
    {
        "number": "04",
        "slug": "nccl_latency_bandwidth",
        "title": "NCCL Latency and Bandwidth：通信成本从哪里来",
        "question": "消息大小、GPU 数量和拓扑怎样改变 collective 成本？",
        "goals": ["建立 latency-bandwidth 直觉", "区分 algorithm bandwidth 与 bus bandwidth", "识别测量质量边界"],
        "module": "04_nccl_benchmark",
        "readme": "../04_nccl_benchmark/README.md",
        "report": "../04_nccl_benchmark/experiments/06_final_report.md",
        "artifact": "04_nccl_benchmark/results/module04_final_summary.json",
        "required": ["schema_version", "artifact_type", "status", "collective_results"],
        "observation": "alpha_us, bandwidth_gbps = 8.0, 8.0\nfor size in (1024, 1024**2, 64*1024**2):\n    transfer_us = size * 8 / (bandwidth_gbps * 1e9) * 1e6\n    print(f'{size:>9} bytes: model={alpha_us + transfer_us:.2f} µs')",
        "analysis": "result = artifact['collective_results']\nprint('DDP payload bytes:', result['ddp_payload_bytes'])\nprint('large-message plateau:', result['allreduce_large_message_plateau_gbps'])\nprint('measurement quality:', artifact['ddp_scaling_followup']['measurement_quality'])",
        "chart": "strong = artifact['ddp_scaling_followup']['strong']\nlabels = ['1 GPU', '2 GPU', '4 GPU']\nvalues = [strong[f'world_{n}']['median_samples_per_second'] for n in (1,2,4)]\nbar_chart(labels, values, title='Strong scaling 中位吞吐', ylabel='samples/s')",
        "local": "[sys.executable, '04_nccl_benchmark/benchmarks/check_environment.py', '--output', str(ctx.output_dir / 'environment.json')]",
        "gpu": "None  # 需要 NCCL_TEST_DIR；请按本章 README/四卡教程从 Terminal 启动正式 campaign",
        "next": "05_collective_algorithms.ipynb",
    },
    {
        "number": "05",
        "slug": "collective_algorithms",
        "title": "Collective Algorithms：Centralized、Ring 与 NCCL",
        "question": "同样是 AllReduce，不同数据移动方式为什么表现不同？",
        "goals": ["逐轮解释 ring", "推导每 rank 的通信量", "理解教学实现与生产库的差距"],
        "module": "05_tiny_collective",
        "readme": "../05_tiny_collective/README.md",
        "report": "../05_tiny_collective/experiments/04_final_report.md",
        "artifact": "05_tiny_collective/results/module05_final_summary.json",
        "required": ["schema_version", "artifact_type", "status", "correctness"],
        "observation": "world_size = 4\nrounds = [('reduce-scatter', i+1) for i in range(world_size-1)] + [('all-gather', i+1) for i in range(world_size-1)]\nrounds",
        "analysis": "rows = artifact['representative_bus_bandwidth_gbps']['world_4']['67108864_bytes']\nfor name, value in rows.items(): print(name, value, 'GB/s')\nprint('边界：', artifact['boundary'])",
        "chart": "values_by_name = artifact['representative_bus_bandwidth_gbps']['world_4']['67108864_bytes']\nbar_chart(list(values_by_name), list(values_by_name.values()), title='4 GPU / 64 MiB AllReduce', ylabel='bus bandwidth (GB/s)')",
        "local": "[sys.executable, '05_tiny_collective/benchmarks/run_correctness.py', '--config', '05_tiny_collective/configs/cpu_correctness.toml', '--output', str(ctx.output_dir / 'correctness.json')]",
        "gpu": "[sys.executable, '05_tiny_collective/benchmarks/run_gpu_comparison.py', '--config', '05_tiny_collective/configs/gpu_comparison.toml', '--raw-directory', str(ctx.output_dir / 'raw'), '--output', str(ctx.output_dir / 'comparison.json')]",
        "next": "06_reducer_bucket_overlap.ipynb",
    },
    {
        "number": "06",
        "slug": "reducer_bucket_overlap",
        "title": "Reducer, Bucket and Overlap：更早通信是否一定更快",
        "question": "怎样在不改变梯度的前提下减少 backward 后的等待？",
        "goals": ["理解参数 readiness 与 bucket", "检查 reducer 数学正确性", "区分真实 overlap 与吞吐提升"],
        "module": "06_training_engine",
        "readme": "../06_training_engine/README.md",
        "report": "../06_training_engine/experiments/06_final_report.md",
        "artifact": "06_training_engine/results/module06_final_summary.json",
        "required": ["schema_version", "artifact_type", "status", "correctness"],
        "observation": "parameters = [('head', 2), ('block2', 6), ('block1', 5)]  # backward readiness order\ncap = 8\nbuckets, current = [], []\nfor item in parameters:\n    if sum(x[1] for x in current) + item[1] > cap:\n        buckets.append(current)\n        current = []\n    current.append(item)\nbuckets.append(current)\nbuckets",
        "analysis": "throughput = artifact['gpu_medium_fp32_world_4_samples_per_second']\nfor name, value in throughput.items(): print(name, value, 'samples/s')\nprint(artifact['overlap_extension']['conclusion'])",
        "chart": "throughput = artifact['gpu_medium_fp32_world_4_samples_per_second']\nbar_chart(list(throughput), list(throughput.values()), title='4 GPU reducer 消融', ylabel='samples/s')",
        "local": "[sys.executable, '06_training_engine/benchmarks/run_reducer_correctness.py', '--config', '06_training_engine/configs/local_correctness.toml', '--raw-directory', str(ctx.output_dir / 'raw'), '--output', str(ctx.output_dir / 'correctness.json')]",
        "gpu": "[sys.executable, '06_training_engine/benchmarks/run_gpu_ablation.py', '--config', '06_training_engine/configs/gpu_ablation.toml', '--raw-directory', str(ctx.output_dir / 'raw'), '--output', str(ctx.output_dir / 'ablation.json')]",
        "next": "07_parallelism_strategies.ipynb",
    },
    {
        "number": "07",
        "slug": "parallelism_strategies",
        "title": "Parallelism Strategies：DDP、FSDP2 与 TP 怎样选择",
        "question": "复制、状态分片和层内切分分别解决什么容量问题？",
        "goals": ["拆解参数/梯度/优化器显存", "比较 DDP/FSDP2/TP 的 collective", "用瓶颈而非先进程度选择策略"],
        "module": "07_parallelism",
        "readme": "../07_parallelism/README.md",
        "report": "../07_parallelism/experiments/README.md",
        "artifact": "07_parallelism/results/module07_final_summary.json",
        "required": ["schema_version", "artifact_type", "status", "correctness"],
        "observation": "model_mib = 1000\nworld_size = 4\nestimate = {\n    'DDP local model state': model_mib,\n    'sharded local model state': model_mib/world_size,\n}\nestimate",
        "analysis": "medium = artifact['world_4_medium']\nprint('吞吐:', medium['samples_per_second'])\nprint('峰值显存 MiB:', medium['peak_memory_mib'])\nprint('结论:', artifact['conclusion'])",
        "chart": "memory = artifact['world_4_medium']['peak_memory_mib']\nbar_chart(list(memory), list(memory.values()), title='4 GPU 峰值显存', ylabel='MiB')",
        "local": "[sys.executable, '07_parallelism/benchmarks/run_tp_correctness.py', '--config', '07_parallelism/configs/local_correctness.toml', '--raw-directory', str(ctx.output_dir / 'raw'), '--output', str(ctx.output_dir / 'tp.json')]",
        "gpu": "[sys.executable, '07_parallelism/benchmarks/run_gpu_parallelism.py', '--config', '07_parallelism/configs/gpu_parallelism.toml', '--raw-directory', str(ctx.output_dir / 'raw'), '--output', str(ctx.output_dir / 'parallelism.json')]",
        "next": "../README_zh-CN.md",
    },
]


def build_chapter(chapter: dict[str, Any]) -> dict[str, Any]:
    key = chapter["number"]
    goals = "\n".join(f"- {goal}" for goal in chapter["goals"])
    intro = f"""# {chapter['number']} · {chapter['title']}

**本节问题：** {chapter['question']}

完成后你应该能够：

{goals}

前置阅读：[模块 README]({chapter['readme']}) · [术语表](../docs/concepts/distributed-systems-glossary.md)
"""
    setup = f"""# 第一处可编辑配置：reference | local | gpu
MODE = "reference"

import sys
from pathlib import Path

notebook_dir = Path("notebooks") if Path("notebooks/_support").is_dir() else Path.cwd()
if str(notebook_dir) not in sys.path:
    sys.path.insert(0, str(notebook_dir))

from _support.artifacts import load_artifact
from _support.context import create_context
from _support.plots import bar_chart
from _support.runner import run_command

ctx = create_context("{chapter['number']}_{chapter['slug']}", MODE)
ctx.card()
"""
    prediction = """## 运行前预测

先写下你的预测。不要担心猜错；后面需要指出证据支持或推翻了哪一部分。
"""
    prediction_code = "PREDICTION = \"我预计……，因为……\"\nPREDICTION"
    reference = f"""artifact_path = ctx.repo_root / {chapter['artifact']!r}
artifact = load_artifact(artifact_path, required={chapter['required']!r})
print("证据来源：仓库参考结果", artifact_path.relative_to(ctx.repo_root))
print("顶层字段：", sorted(artifact))
"""
    command = f"""commands = {{
    "local": {chapter['local']},
    "gpu": {chapter['gpu']},
}}
command = commands.get(ctx.mode)
if ctx.mode == "reference":
    print("reference 模式：只读已提交证据，不启动实验。")
elif command is None:
    print("本节需要额外环境准备；请使用上方链接中的 Terminal 流程。")
else:
    result = run_command(
        command,
        cwd=ctx.repo_root,
        output_dir=ctx.output_dir,
        label=f"{{ctx.mode}}-run",
        timeout_seconds=1200,
    )
    print({{"passed": result.passed, "seconds": round(result.elapsed_seconds, 2), "log": result.log_path.name}})
"""
    ending = f"""## 与预测对照、一般规律与边界

请回答：你的预测哪一部分被支持，哪一部分被推翻？当前证据只适用于哪些硬件、shape、消息大小或软件版本？

完成实验后再阅读[正式报告]({chapter['report']})。正式报告是结论来源，Notebook 只是交互式观察层。

### 检查题

1. correctness 是否先于性能成立？
2. 当前指标的单位、重复方式和来源是什么？
3. `unavailable`、`failed` 与数值 0 为什么不能混为一谈？

下一步：[打开下一章]({chapter['next']})。
"""
    cells = [
        markdown(key, "intro", intro),
        markdown(key, "setup-note", "## 运行状态卡\n\n默认 `reference` 可在无 GPU 电脑上 Run All。改为 `local` 或 `gpu` 才会启动 runner。"),
        code(key, "setup", setup, ["setup"]),
        markdown(key, "prediction", prediction, ["prediction"]),
        code(key, "prediction-value", prediction_code, ["prediction"]),
        markdown(key, "observation-note", "## 最小观察\n\n这个单元只暴露关键中间状态；正式算法仍来自项目源码。"),
        code(key, "observation", chapter["observation"], ["reference"]),
        markdown(key, "artifact-note", "## 正确性门与参考证据\n\n读取已提交 JSON；字段缺失时立即停止，不把缺失值解释成 0。"),
        code(key, "artifact", reference, ["reference"]),
        code(key, "analysis", chapter["analysis"], ["analysis"]),
        code(key, "chart", chapter["chart"], ["analysis"]),
        markdown(key, "run-note", "## 本地/正式实验\n\n命令使用参数列表在独立子进程中执行，日志和产物只写入 `_runs/`。"),
        code(key, "run", command, ["local-run", "gpu-run"]),
        markdown(key, "ending", ending, ["exercise"]),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "TrainScale Lab (Python 3.11)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for chapter in CHAPTERS:
        path = HERE / f"{chapter['number']}_{chapter['slug']}.ipynb"
        path.write_text(json.dumps(build_chapter(chapter), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
