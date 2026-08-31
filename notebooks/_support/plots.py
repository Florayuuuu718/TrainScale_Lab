"""Consistent, honest plotting defaults for reference and local evidence."""

from collections.abc import Sequence

REFERENCE_COLOR = "#4C78A8"
LOCAL_COLOR = "#F58518"


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    source: str = "仓库参考结果",
):
    """Draw a zero-based bar chart and return its Matplotlib figure."""
    if len(labels) != len(values) or not labels:
        raise ValueError("labels 与 values 必须非空且长度一致。")
    import matplotlib.pyplot as plt

    color = REFERENCE_COLOR if source == "仓库参考结果" else LOCAL_COLOR
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=color)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}（{source}）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
