"""子 Agent 展示清洗工具：把内部 subagent 注入文本裁剪成人类可读版本。

持久化到磁盘的 subagent 结果通常包含完整任务说明和给模型看的额外提示，
但这些内容不适合直接展示到外部渠道，因此这里负责做“面向用户的裁剪版”。
"""

from __future__ import annotations

from typing import Any

# 限制展示给渠道的 Result 长度；完整文本仍然保留在磁盘/内部上下文中。
_SUBAGENT_CHANNEL_RESULT_MAX_CHARS = 800


def scrub_subagent_announce_body(content: str) -> str:
    """把完整 subagent announce 文本裁剪成适合渠道展示的版本。"""
    stripped = content.replace("\r\n", "\n").strip()
    lines = stripped.splitlines()
    header = ""
    if lines and lines[0].startswith("[Subagent"):
        header = lines[0].strip()

    lower = stripped.lower()
    key = "\nresult:\n"
    ri = lower.find(key)
    if ri == -1:
        key = "\nresult:"
        ri = lower.find(key)
    if ri == -1:
        return header if header else stripped

    after = stripped[ri + len(key) :].lstrip()
    summ_marker = "summarize this naturally"
    si = after.lower().find(summ_marker)
    if si != -1:
        after = after[:si].rstrip()

    body = after.strip()
    limit = _SUBAGENT_CHANNEL_RESULT_MAX_CHARS
    if limit and len(body) > limit:
        body = body[: limit - 1].rstrip() + "…"
    if header and body:
        return f"{header}\n\n{body}"
    return header or body or stripped


def scrub_subagent_messages_for_channel(messages: list[dict[str, Any]]) -> None:
    """原地修改消息列表中携带 ``subagent_result`` 注入的消息内容。"""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("injected_event") != "subagent_result":
            continue
        raw = msg.get("content")
        if not isinstance(raw, str) or not raw.strip():
            continue
        msg["content"] = scrub_subagent_announce_body(raw)
