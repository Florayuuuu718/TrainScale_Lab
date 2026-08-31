"""Small, read-only helpers for committed and locally generated JSON artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """Raised when an artifact cannot support the requested analysis."""


def load_artifact(
    path: Path,
    *,
    required: Iterable[str] = (),
    allowed_statuses: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Read UTF-8 JSON and reject incomplete evidence instead of inventing zeros."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"Artifact 不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Artifact 不是合法 JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Artifact 顶层必须是对象：{path}")
    missing = [field for field in required if field not in value]
    if missing:
        raise ArtifactError(f"Artifact 缺少字段 {missing}：{path}")
    if allowed_statuses is not None:
        allowed = set(allowed_statuses)
        status = value.get("status")
        if status not in allowed:
            raise ArtifactError(f"Artifact 状态 {status!r} 不允许继续分析：{path}")
    return value


def get_path(value: dict[str, Any], dotted_path: str) -> Any:
    """Read a dotted path and fail loudly when the schema differs."""
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ArtifactError(f"Artifact 缺少路径：{dotted_path}")
        current = current[part]
    return current


def rows(value: dict[str, Any], fields: dict[str, str]) -> list[dict[str, Any]]:
    """Select a compact table from an artifact without mutating it."""
    return [{"指标": label, "值": get_path(value, path)} for label, path in fields.items()]
