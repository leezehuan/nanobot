"""供 WebUI 使用的技能摘要与详情接口辅助函数。

【中文名称】技能列表接口层

WebUI 不适合直接拿到本地技能文件的完整路径等内部细节，
所以这里对 `SkillsLoader` 做了一层“安全投影”：

- 提供适合前端展示的技能列表
- 提供单个技能详情
- 避免把不必要的本地文件系统信息泄露到前端
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.skills import SkillsLoader


def webui_skills_payload(
    workspace_path: Path,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any]:
    """返回适合 WebUI 展示的技能列表载荷。"""
    loader = SkillsLoader(workspace_path, disabled_skills=disabled_skills)
    entries = sorted(
        loader.list_skills(filter_unavailable=False),
        key=lambda entry: (entry.get("source") != "workspace", entry["name"]),
    )
    return {"skills": [_skill_payload(loader, entry) for entry in entries]}


def webui_skill_detail_payload(
    workspace_path: Path,
    name: str,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any] | None:
    """返回单个技能的安全详情载荷；不存在时返回 `None`。"""
    loader = SkillsLoader(workspace_path, disabled_skills=disabled_skills)
    entries = loader.list_skills(filter_unavailable=False)
    entry = next((item for item in entries if item["name"] == name), None)
    if entry is None:
        return None
    return {
        **_skill_payload(loader, entry),
        "requirements": loader.get_skill_requirements(name),
        "raw_markdown": loader.load_skill(name) or "",
    }


def _skill_payload(loader: SkillsLoader, entry: dict[str, str]) -> dict[str, Any]:
    """把一个技能条目压缩成前端真正需要的字段。"""
    name = entry["name"]
    metadata = loader.get_skill_metadata(name)
    available, unavailable_reason = loader.get_skill_availability(name)
    return {
        "name": name,
        "description": _description(metadata, name),
        "source": entry.get("source", "unknown"),
        "available": available,
        "unavailable_reason": unavailable_reason,
    }


def _description(metadata: dict[str, Any] | None, fallback: str) -> str:
    """优先使用技能元数据里的描述；没有则回退到名称。"""
    if metadata is None:
        return fallback
    value = metadata.get("description")
    return value.strip() if isinstance(value, str) and value.strip() else fallback
