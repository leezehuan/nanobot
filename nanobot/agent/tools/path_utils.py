"""工作区范围工具共享的路径辅助函数。"""

from pathlib import Path

from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_policy import (
    is_path_within,
    resolve_allowed_path,
)


def is_under(path: Path, directory: Path) -> bool:
    """判断一个路径解析后是否位于指定目录之下。"""
    return is_path_within(path, directory)


def resolve_workspace_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """基于 workspace 解析路径，并强制校验是否仍在允许目录范围内。"""
    extra_roots = [get_media_dir(), *(extra_allowed_dirs or [])] if allowed_dir else None
    return resolve_allowed_path(
        path,
        workspace=workspace,
        allowed_root=allowed_dir,
        extra_allowed_roots=extra_roots,
    )
