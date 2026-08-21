"""Render dependency-free SVG loss and accuracy curves from a training summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _points(values: list[float], x0: int, y0: int, width: int, height: int) -> str:
    low, high = min(values), max(values)
    span = high - low or 1.0
    x_step = width / max(len(values) - 1, 1)
    return " ".join(
        f"{x0 + index * x_step:.1f},{y0 + height - (value - low) / span * height:.1f}"
        for index, value in enumerate(values)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    history = value["history"]
    if not history:
        raise SystemExit("summary contains no epochs")
    train_loss = [row["train"]["loss"] for row in history]
    valid_loss = [row["valid"]["loss"] for row in history]
    valid_accuracy = [row["valid"]["accuracy"] for row in history]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420">
<rect width="100%" height="100%" fill="white"/>
<text x="40" y="28" font-family="sans-serif" font-size="20">{value['experiment_name']}</text>
<text x="40" y="55" font-family="sans-serif" font-size="14">loss (blue=train, orange=valid)</text>
<rect x="40" y="70" width="380" height="280" fill="none" stroke="#aaa"/>
<polyline points="{_points(train_loss, 40, 70, 380, 280)}"
 fill="none" stroke="#2563eb" stroke-width="3"/>
<polyline points="{_points(valid_loss, 40, 70, 380, 280)}"
 fill="none" stroke="#ea580c" stroke-width="3"/>
<text x="480" y="55" font-family="sans-serif" font-size="14">validation accuracy</text>
<rect x="480" y="70" width="380" height="280" fill="none" stroke="#aaa"/>
<polyline points="{_points(valid_accuracy, 480, 70, 380, 280)}"
 fill="none" stroke="#16a34a" stroke-width="3"/>
<text x="40" y="390" font-family="sans-serif" font-size="12">epoch 1 .. {len(history)}</text>
</svg>"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
