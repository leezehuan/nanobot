"""CLI Apps 在 AgentLoop 与设置页之间共享的小工具。

【中文名称】CLI 应用辅助函数

这类函数本身不负责安装或执行 CLI App，而是解决“上下文传递”问题：

- 当前会话里附加了哪些 CLI App？
- 当前这一轮消息，应该把哪些 CLI App 信息暴露给模型？

你可以把它理解成“应用运行时提示信息的拼装层”。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """提取需要持久化到会话中的 CLI App 附加信息。

    如果本轮消息元数据里带了 `cli_apps`，这里就把它整理成 session
    可以保存的额外字段，供后续轮次继续使用。
    """
    cli_apps = metadata.get("cli_apps") if isinstance(metadata, Mapping) else None
    return {"cli_apps": cli_apps} if isinstance(cli_apps, list) and cli_apps else {}


def runtime_lines(message: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """生成“模型可见”的 CLI App 运行时提示文本。

    这些文本会被放进模型上下文，让模型知道：

    - 用户提到了哪个 CLI App；
    - 对应技能文件在哪里；
    - 正确调用方式应该是 `run_cli_app`，而不是偷偷绕过工具去跑 shell。
    """
    if skip:
        return []
    text = message.content if isinstance(getattr(message, "content", None), str) else ""
    metadata = message.metadata if isinstance(getattr(message, "metadata", None), Mapping) else None
    return _cli_app_runtime_lines(text, metadata, workspace)


def _cli_app_runtime_lines(
    text: str,
    metadata: Mapping[str, Any] | None,
    workspace: Path,
) -> list[str]:
    """根据文本与结构化元数据，组装 CLI App 提示行。

    这里有两条路径：

    1. 优先读取结构化 `metadata["cli_apps"]`
       这说明前端/上游已经明确告诉我们当前附加了哪些 app。
    2. 如果没有结构化信息，再从文本里扫描 `@app-name`
       这是兼容用户手工在消息里提及应用的兜底方案。
    """
    structured = metadata.get("cli_apps") if isinstance(metadata, Mapping) else None
    if isinstance(structured, list):
        mentions = [
            item for item in structured
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ]
        if mentions:
            return [
                "CLI App Attachment: "
                f"@{str(item['name']).strip().lower()} "
                f"(installed; tool=run_cli_app; "
                f"entry_point={str(item.get('entry_point') or 'unknown')}; "
                f"skill=skills/cli-app-{str(item['name']).strip().lower()}/SKILL.md). "
                "Read the skill when useful, then run this app with `run_cli_app`; do not bypass it with shell."
                for item in mentions
                if str(item.get("name") or "").strip()
            ]
    if "@" not in text:
        return []
    try:
        from nanobot.apps.cli import CliAppManager

        mentions = CliAppManager(workspace=workspace).mentioned_installed_apps(text)
    except Exception:
        return []
    return [
        "CLI App Mention: "
        f"@{item['name']} "
        f"(installed; tool={item['tool']}; "
        f"entry_point={item['entry_point'] or 'unknown'}; "
        f"skill={item['skill']}). "
        "Read the skill when useful, then run this app with `run_cli_app`; do not bypass it with shell."
        for item in mentions
    ]
