"""工作区路径边界辅助函数。

【中文名称】工作区路径策略

这组函数负责统一回答一个基础问题：
“某个路径是否还在允许访问的工作区边界内？”

它们属于应用层安全护栏，主要目的是让不同工具在做路径判断时口径一致。
但要注意，它们不是操作系统级沙箱的替代品。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

WORKSPACE_BOUNDARY_NOTE = (
    "（这是一个明确的策略边界，不是临时性故障；"
    "不要通过 shell 小技巧或其它工具绕过，"
    "如果资源确实必须访问，应当向用户确认后再继续）"
)


class WorkspaceBoundaryError(PermissionError):
    """当请求路径越过允许访问的工作区边界时抛出。"""


def resolve_path(path: str | Path, workspace: str | Path | None = None, *, strict: bool = False) -> Path:
    """解析路径；如果给的是相对路径，则按工作区目录补全。"""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and workspace is not None:
        candidate = Path(workspace).expanduser() / candidate
    return candidate.resolve(strict=strict)


def is_path_within(path: str | Path, root: str | Path) -> bool:
    """判断路径是否位于指定根目录内部。"""
    try:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        resolved_root = Path(root).expanduser().resolve(strict=False)
        resolved_path.relative_to(resolved_root)
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def is_path_allowed(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """判断路径是否落在任意一个允许根目录里。"""
    return any(is_path_within(path, root) for root in roots)


def require_path_within(
    path: str | Path,
    root: str | Path,
    *,
    message: str | None = None,
) -> Path:
    """解析路径，并强制要求它位于给定根目录内部。"""
    resolved = Path(path).expanduser().resolve(strict=False)
    if not is_path_within(resolved, root):
        raise WorkspaceBoundaryError(
            message
            or f"Path {path} is outside allowed directory {Path(root).expanduser()}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    return resolved


def resolve_allowed_path(
    path: str | Path,
    *,
    workspace: str | Path | None = None,
    allowed_root: str | Path | None = None,
    extra_allowed_roots: Iterable[str | Path] | None = None,
    strict: bool = False,
) -> Path:
    """解析路径，并在配置了允许根目录时执行边界校验。"""
    resolved = resolve_path(path, workspace, strict=False)
    if allowed_root is None:
        return resolve_path(path, workspace, strict=strict) if strict else resolved

    roots = [allowed_root, *(extra_allowed_roots or [])]
    if not is_path_allowed(resolved, roots):
        raise WorkspaceBoundaryError(
            f"Path {path} is outside allowed directory {Path(allowed_root).expanduser()}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    if strict:
        return resolve_path(path, workspace, strict=True)
    return resolved
