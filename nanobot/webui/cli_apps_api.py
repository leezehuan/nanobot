"""WebUI 侧的 CLI Apps 辅助函数。

【中文名称】WebUI 命令行应用接口辅助层

这个模块站在 WebUI 后端接口层，负责做两件事：

1. 把前端传来的 CLI App 结构化信息做一次清洗和规范化；
2. 把前端动作（install/update/uninstall/test）转发给 `CliAppManager`。

因此它属于“薄适配层”，自己不实现安装逻辑。
"""

from __future__ import annotations

import re
from typing import Any

from nanobot.apps.cli import CliAppError, CliAppManager, CliAppsRuntimeConfig
from nanobot.config.loader import load_config

QueryParams = dict[str, list[str]]

_CLI_APP_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$", re.IGNORECASE)
_CLI_APP_ATTACHMENT_KEYS = (
    "name",
    "display_name",
    "category",
    "entry_point",
    "logo_url",
    "brand_color",
)


def _clip_ws_string(value: Any, limit: int = 240) -> str | None:
    """裁剪并清洗来自 WebSocket / WebUI 的字符串字段。"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text[:limit]


def normalize_cli_app_mentions(raw: Any) -> list[dict[str, str]]:
    """清洗前端传来的结构化 CLI App 引用列表。

    这里会做：

    - 类型过滤
    - 名称格式校验
    - 去重
    - 字段长度裁剪
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        name = _clip_ws_string(item.get("name"), 64)
        if not name or _CLI_APP_NAME_RE.match(name) is None:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, str] = {"name": key}
        for field in _CLI_APP_ATTACHMENT_KEYS[1:]:
            value = _clip_ws_string(item.get(field), 512 if field == "logo_url" else 160)
            if value:
                row[field] = value
        out.append(row)
    return out


def _query_first(query: QueryParams, key: str) -> str | None:
    """从 query 参数里取出某个字段的第一个值。"""
    values = query.get(key)
    return values[0] if values else None


def _manager() -> CliAppManager:
    """按当前配置创建一个 `CliAppManager`。"""
    config = load_config()
    cli_cfg = config.tools.cli_apps
    return CliAppManager(
        workspace=config.workspace_path,
        runtime=CliAppsRuntimeConfig(
            install_timeout=cli_cfg.install_timeout,
            run_timeout=cli_cfg.run_timeout,
            catalog_ttl_seconds=cli_cfg.catalog_ttl_seconds,
        ),
    )


def cli_apps_payload() -> dict[str, Any]:
    """返回 WebUI 所需的 CLI Apps 总览载荷。"""
    return _manager().payload()


def cli_apps_action(action: str, query: QueryParams) -> dict[str, Any]:
    """执行一次 WebUI 发起的 CLI App 动作。"""
    name = (_query_first(query, "name") or "").strip()
    if not name:
        raise CliAppError("missing CLI app name")
    manager = _manager()
    if action == "install":
        return manager.install(name)
    if action == "update":
        return manager.update(name)
    if action == "uninstall":
        return manager.uninstall(name)
    if action == "test":
        return manager.test(name)
    raise CliAppError(f"unknown CLI app action '{action}'", status=404)
