"""结构化进度事件辅助函数：供 Agent 运行时统一构造进度回调 payload。

这个模块解决的是“同一个 on_progress 回调，可能支持不同参数能力”的兼容问题：
- 有些前端只接受纯文本进度
- 有些前端还支持结构化工具事件
- 有些前端支持文件编辑事件
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.agent.hook import AgentHookContext


def on_progress_accepts_tool_events(cb: Callable[..., Any]) -> bool:
    """判断某个 ``on_progress`` 回调是否显式支持 ``tool_events`` 参数。"""
    return _on_progress_accepts(cb, "tool_events")


def on_progress_accepts_file_edit_events(cb: Callable[..., Any]) -> bool:
    """判断某个 ``on_progress`` 回调是否显式支持 ``file_edit_events`` 参数。"""
    return _on_progress_accepts(cb, "file_edit_events")


def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
    """通过函数签名检查回调是否接受指定命名参数。"""
    try:
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


async def invoke_on_progress(
    on_progress: Callable[..., Awaitable[None]],
    content: str,
    *,
    tool_hint: bool = False,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    """调用 ``on_progress``，并在支持时附带结构化工具事件。"""
    if tool_events and on_progress_accepts_tool_events(on_progress):
        await on_progress(content, tool_hint=tool_hint, tool_events=tool_events)
        return
    await on_progress(content, tool_hint=tool_hint)


async def invoke_file_edit_progress(
    on_progress: Callable[..., Awaitable[None]],
    file_edit_events: list[dict[str, Any]],
) -> None:
    """调用 ``on_progress`` 发送结构化文件编辑事件。"""
    if not file_edit_events or not on_progress_accepts_file_edit_events(on_progress):
        return
    await on_progress("", file_edit_events=file_edit_events)


def _tool_event_arguments(tool_call: Any) -> dict[str, Any]:
    """从 tool_call 中取出 arguments，并保证结果始终是 dict。"""
    arguments = getattr(tool_call, "arguments", {}) or {}
    return arguments if isinstance(arguments, dict) else {}


def build_tool_event_start_payload(tool_call: Any) -> dict[str, Any]:
    """构造“工具开始执行”事件 payload。"""
    return {
        "version": 1,
        "phase": "start",
        "call_id": str(getattr(tool_call, "id", "") or ""),
        "name": getattr(tool_call, "name", ""),
        "arguments": _tool_event_arguments(tool_call),
        "result": None,
        "error": None,
        "files": [],
        "embeds": [],
    }


def tool_event_result_extras(result: Any) -> tuple[list[Any], list[Any]]:
    """从工具结果里提取 files / embeds 这类附加结构化数据。"""
    if not isinstance(result, dict):
        return [], []
    files = result.get("files") if isinstance(result.get("files"), list) else []
    embeds = result.get("embeds") if isinstance(result.get("embeds"), list) else []
    return files, embeds


def build_tool_event_finish_payloads(context: AgentHookContext) -> list[dict[str, Any]]:
    """根据 hook context 生成一批“工具结束/失败”事件 payload。"""
    payloads: list[dict[str, Any]] = []
    count = min(len(context.tool_calls), len(context.tool_results), len(context.tool_events))
    for idx in range(count):
        tool_call = context.tool_calls[idx]
        result = context.tool_results[idx]
        event = context.tool_events[idx] if isinstance(context.tool_events[idx], dict) else {}
        # hook 层把工具执行状态记录在 tool_events 里，这里再折叠成统一 phase。
        status = event.get("status")
        phase = "end" if status == "ok" else "error"
        files, embeds = tool_event_result_extras(result)
        payload = {
            "version": 1,
            "phase": phase,
            "call_id": str(getattr(tool_call, "id", "") or ""),
            "name": getattr(tool_call, "name", ""),
            "arguments": _tool_event_arguments(tool_call),
            "result": result if phase == "end" else None,
            "error": None,
            "files": files,
            "embeds": embeds,
        }
        if phase == "error":
            if isinstance(result, str) and result.strip():
                payload["error"] = result.strip()
            else:
                payload["error"] = str(event.get("detail") or "Tool execution failed")
        payloads.append(payload)
    return payloads
