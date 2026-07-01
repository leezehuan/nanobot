"""Responses API 转换层：把 Chat Completions 风格输入转换成 Responses 格式。

nanobot 的历史抽象和很多 provider 适配层仍以 Chat Completions 风格组织消息。
当底层改走 OpenAI Responses API 时，就需要这一层做结构翻译。
"""

from __future__ import annotations

import json
from typing import Any

from nanobot.providers.base import tool_arguments_json_for_replay


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """把 Chat Completions 风格消息转换成 Responses API ``input`` 条目。

    返回 ``(system_prompt, input_items)``：

    - ``system_prompt``：从 ``system`` 消息中单独抽出
    - ``input_items``：Responses API 需要的 ``input`` 数组

    实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。
    """
    system_prompt = ""
    input_items: list[dict[str, Any]] = []
    used_item_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            # Responses API 常把 system 指令放在顶层 ``instructions`` 字段里，
            # 而不是和普通 input item 混在一起。
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            # 用户消息需要进一步拆成 input_text / input_image 等多模态片段。
            input_items.append(convert_user_message(content))
            continue

        if role == "assistant":
            # assistant 的普通文本回复要转成 completed message item。
            if isinstance(content, str) and content:
                message_id = _unique_item_id(f"msg_{idx}", used_item_ids)
                input_items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                    "status": "completed", "id": message_id,
                })
            for tool_call in msg.get("tool_calls", []) or []:
                # assistant 发出的工具调用，要改写成 Responses API 的 function_call item。
                fn = tool_call.get("function") or {}
                call_id, item_id = split_tool_call_id(tool_call.get("id"))
                response_item_id = _unique_item_id(item_id or f"fc_{idx}", used_item_ids)
                input_items.append({
                    "type": "function_call",
                    "id": response_item_id,
                    "call_id": call_id or f"call_{idx}",
                    "name": fn.get("name"),
                    "arguments": tool_arguments_json_for_replay(fn.get("arguments")),
                })
            continue

        if role == "tool":
            # 工具执行结果对应 Responses API 里的 function_call_output item。
            call_id, _ = split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append({"type": "function_call_output", "call_id": call_id, "output": output_text})

    return system_prompt, input_items


def convert_user_message(content: Any) -> dict[str, Any]:
    """把用户消息内容转换成 Responses API 结构。

    支持的输入：

    - 纯字符串
    - ``text`` 块 -> ``input_text``
    - ``image_url`` 块 -> ``input_image``

    实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。
    """
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把工具 schema 从 Chat Completions 风格拍平成 Responses API 风格。

    Chat Completions 常见结构：
    ``{"type": "function", "function": {...}}``

    Responses API 更偏扁平：
    ``{"type": "function", "name": "...", "parameters": {...}}``

    实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _unique_item_id(item_id: str, used: set[str]) -> str:
    """确保同一个 Responses 请求里的 item id 唯一。

    因为 assistant 文本、tool call、tool output 都会拆成多个 item，
    一旦 ID 冲突，上游就可能无法正确关联调用链。

    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
    """
    if item_id not in used:
        used.add(item_id)
        return item_id

    suffix = 2
    while f"{item_id}_{suffix}" in used:
        suffix += 1
    unique = f"{item_id}_{suffix}"
    used.add(unique)
    return unique


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """拆分形如 ``call_id|item_id`` 的复合工具调用 ID。

    返回 ``(call_id, item_id)``，其中 ``item_id`` 可以为空。

    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
    """
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None
