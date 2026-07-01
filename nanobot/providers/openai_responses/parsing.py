"""Responses API 解析层：把原始流/对象转换成 nanobot 统一结果。

nanobot 内部很多地方仍然偏向使用“类似 Chat Completions”的统一抽象，
而 OpenAI Responses API 的事件流、tool call、reasoning 表示方式并不一样。

这个模块的职责就是做这层翻译。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator

import httpx
from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest, parse_tool_arguments

FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def map_finish_reason(status: str | None) -> str:
    """把 Responses API 状态映射成 Chat Completions 风格的 finish_reason。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return FINISH_REASON_MAP.get(status or "completed", "stop")


def _usage_from_response_obj(response: Any) -> dict[str, int]:
    """从 Responses 响应对象中抽取统一的 usage 统计。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    usage_raw = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if not usage_raw:
        return {}
    if not isinstance(usage_raw, dict):
        dump = getattr(usage_raw, "model_dump", None)
        usage_raw = dump() if callable(dump) else vars(usage_raw)
    prompt_tokens = int(usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0)
    completion_tokens = int(
        usage_raw.get("output_tokens") or usage_raw.get("completion_tokens") or 0
    )
    total_tokens = int(usage_raw.get("total_tokens") or prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _parse_tool_call_arguments(args_raw: Any, name: str | None) -> Any:
    """解析工具参数；若解析失败则保留原值并记告警。
    
    实现方法：按响应或文本结构逐层读取字段，把缺失和异常格式归一化为 nanobot 内部对象。"""
    parsed = parse_tool_arguments(args_raw)
    if parsed == args_raw and isinstance(args_raw, str) and args_raw.strip():
        logger.warning(
            "Failed to parse tool call arguments for '{}': {}",
            name,
            args_raw[:200],
        )
    return parsed


def _tool_arguments_source(*values: Any) -> Any:
    """从多个候选值里挑出第一个真正有内容的参数来源。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return "{}"


async def iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """逐条解析并产出 Responses API SSE 流中的 JSON 事件。

    SSE 本质上是“按空行分隔的一段段文本事件”，这里负责把它刷成 JSON 对象。

    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
    """
    buffer: list[str] = []

    def _flush() -> dict[str, Any] | None:
        """把当前缓冲区中的一条 SSE 事件刷成 JSON。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        data_lines = [line[5:].strip() for line in buffer if line.startswith("data:")]
        buffer.clear()
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            return json.loads(data)
        except Exception:
            logger.warning("Failed to parse SSE event JSON: {}", data[:200])
            return None

    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                event = _flush()
                if event is not None:
                    yield event
            continue
        buffer.append(line)

    # 流结束时把缓冲区里最后还没刷出的事件补刷出来。
    if buffer:
        event = _flush()
        if event is not None:
            yield event


async def consume_sse(
    response: httpx.Response,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str]:
    """消费 SSE 流，并提取正文、工具调用和 finish_reason。

    这是不关心 reasoning 的轻量包装版本。

    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
    """
    content, tool_calls, finish_reason, _, _ = await consume_sse_with_reasoning(
        response,
        on_content_delta=on_content_delta,
        on_tool_call_delta=on_tool_call_delta,
    )
    return content, tool_calls, finish_reason


async def consume_sse_with_reasoning(
    response: httpx.Response,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, dict[str, int], str | None]:
    """消费 SSE 流，并额外收集 reasoning 摘要。

    这是 Responses API 流式解析的核心入口。它会把零散事件聚合成：
    - 完整正文
    - 工具调用列表
    - finish_reason
    - usage
    - reasoning_content

    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
    """
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    tool_call_args_emitted: set[str] = set()
    finish_reason = "stop"
    usage: dict[str, int] = {}
    reasoning_content: str | None = None
    streamed_reasoning = False

    async for event in iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                # 工具调用会先创建一个 function_call item，
                # 真正的 arguments 往往在后续 delta 事件里逐步补全。
                call_id = item.get("call_id")
                if not call_id:
                    continue
                arguments = item.get("arguments")
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": "" if arguments is None else arguments,
                }
                if on_tool_call_delta:
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(item.get("name") or ""),
                        "arguments_delta": "",
                    })
        elif event_type == "response.output_text.delta":
            # 普通文本增量：一边累积到最终内容，一边转发给上层流式回调。
            delta_text = event.get("delta") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.reasoning_summary_text.delta":
            # reasoning 摘要也可能按增量流式输出。
            delta_text = event.get("delta") or ""
            if delta_text:
                reasoning_content = (reasoning_content or "") + delta_text
                streamed_reasoning = True
                if on_reasoning_delta:
                    await on_reasoning_delta(delta_text)
        elif event_type == "response.reasoning_summary_text.done":
            text = event.get("text") or ""
            if text and not streamed_reasoning and not reasoning_content:
                reasoning_content = text
                if on_reasoning_delta:
                    await on_reasoning_delta(text)
        elif event_type == "response.reasoning_summary_part.done":
            part = event.get("part") or {}
            text = part.get("text") if part.get("type") == "summary_text" else None
            if text and not streamed_reasoning and not reasoning_content:
                reasoning_content = text
                if on_reasoning_delta:
                    await on_reasoning_delta(text)
        elif event_type == "response.function_call_arguments.delta":
            # 工具参数是增量式的，所以要按 call_id 做缓冲拼接。
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                delta = event.get("delta") or ""
                current = tool_call_buffers[call_id].get("arguments")
                if not isinstance(current, str):
                    current = ""
                tool_call_buffers[call_id]["arguments"] = current + delta
                if on_tool_call_delta and delta:
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(tool_call_buffers[call_id].get("name") or ""),
                        "arguments_delta": str(delta),
                    })
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                arguments = event.get("arguments")
                tool_call_buffers[call_id]["arguments"] = arguments
                if on_tool_call_delta:
                    tool_call_args_emitted.add(str(call_id))
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(tool_call_buffers[call_id].get("name") or ""),
                        "arguments": "" if arguments is None else str(arguments),
                    })
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                # 当 function_call item 完结时，才真正组装成 ToolCallRequest。
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = _tool_arguments_source(buf.get("arguments"), item.get("arguments"))
                if on_tool_call_delta and str(call_id) not in tool_call_args_emitted:
                    tool_call_args_emitted.add(str(call_id))
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(buf.get("name") or item.get("name") or ""),
                        "arguments": str(args_raw),
                    })
                args = _parse_tool_call_arguments(
                    args_raw,
                    buf.get("name") or item.get("name"),
                )
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=buf.get("name") or item.get("name") or "",
                        arguments=args,
                    )
                )
            elif item.get("type") == "reasoning" and not reasoning_content:
                summary = _extract_reasoning_summary_from_output([item])
                if summary:
                    reasoning_content = summary
                    if on_reasoning_delta:
                        await on_reasoning_delta(summary)
        elif event_type == "response.completed":
            # 整体响应结束事件，通常带最终 status / usage / output。
            response_obj = event.get("response") or {}
            status = response_obj.get("status")
            finish_reason = map_finish_reason(status)
            usage = _usage_from_response_obj(response_obj) or usage
            if not reasoning_content:
                summary = _extract_reasoning_summary_from_output(response_obj.get("output") or [])
                if summary:
                    reasoning_content = summary
                    if on_reasoning_delta:
                        await on_reasoning_delta(summary)
        elif event_type in {"error", "response.failed"}:
            detail = event.get("error") or event.get("message") or event
            raise RuntimeError(f"Response failed: {str(detail)[:500]}")

    return content, tool_calls, finish_reason, usage, reasoning_content


def _extract_reasoning_summary_from_output(output: Any) -> str | None:
    """从 Responses ``output`` 结构中提取 reasoning summary 文本。
    
    实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
    parts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            dump = getattr(item, "model_dump", None)
            item = dump() if callable(dump) else vars(item)
        if item.get("type") != "reasoning":
            continue
        for summary in item.get("summary") or []:
            if not isinstance(summary, dict):
                dump = getattr(summary, "model_dump", None)
                summary = dump() if callable(dump) else vars(summary)
            if summary.get("type") == "summary_text" and summary.get("text"):
                parts.append(summary["text"])
    return "".join(parts) or None


def parse_response_output(response: Any) -> LLMResponse:
    """把非流式 SDK ``Response`` 对象解析成统一 ``LLMResponse``。
    
    实现方法：按响应或文本结构逐层读取字段，把缺失和异常格式归一化为 nanobot 内部对象。"""
    if not isinstance(response, dict):
        dump = getattr(response, "model_dump", None)
        response = dump() if callable(dump) else vars(response)

    output = response.get("output") or []
    content_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    reasoning_content: str | None = None

    for item in output:
        if not isinstance(item, dict):
            dump = getattr(item, "model_dump", None)
            item = dump() if callable(dump) else vars(item)

        item_type = item.get("type")
        if item_type == "message":
            # assistant 文本通常以 message -> content -> output_text 的层级出现。
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    dump = getattr(block, "model_dump", None)
                    block = dump() if callable(dump) else vars(block)
                if block.get("type") == "output_text":
                    content_parts.append(block.get("text") or "")
        elif item_type == "reasoning":
            # reasoning 的 summary 文本会被拼成一个展示摘要。
            for s in item.get("summary") or []:
                if not isinstance(s, dict):
                    dump = getattr(s, "model_dump", None)
                    s = dump() if callable(dump) else vars(s)
                if s.get("type") == "summary_text" and s.get("text"):
                    reasoning_content = (reasoning_content or "") + s["text"]
        elif item_type == "function_call":
            # 非流式模式下，工具调用直接作为 output item 给出。
            call_id = item.get("call_id") or ""
            item_id = item.get("id") or "fc_0"
            args_raw = _tool_arguments_source(item.get("arguments"))
            args = _parse_tool_call_arguments(args_raw, item.get("name"))
            tool_calls.append(ToolCallRequest(
                id=f"{call_id}|{item_id}",
                name=item.get("name") or "",
                arguments=args,
            ))

    usage = _usage_from_response_obj(response)

    status = response.get("status")
    finish_reason = map_finish_reason(status)

    return LLMResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
    )


async def consume_sdk_stream(
    stream: Any,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, dict[str, int], str | None]:
    """消费 OpenAI SDK 的异步 Responses 流。

    和 ``consume_sse_with_reasoning`` 类似，但这里面对的是 SDK 已解码好的事件对象，
    而不是原始 HTTP SSE 文本流。

    实现方法：逐块读取上游事件，把文本增量、工具调用增量和完成信号分别转发给调用方。
    """
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    tool_call_args_emitted: set[str] = set()
    finish_reason = "stop"
    usage: dict[str, int] = {}
    reasoning_content: str | None = None

    async for event in stream:
        # SDK 不同版本的事件对象形态可能略有差异，所以这里用防御式属性读取。
        event_type = getattr(event, "type", None)
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                call_id = getattr(item, "call_id", None)
                if not call_id:
                    continue
                arguments = getattr(item, "arguments", None)
                tool_call_buffers[call_id] = {
                    "id": getattr(item, "id", None) or "fc_0",
                    "name": getattr(item, "name", None),
                    "arguments": "" if arguments is None else arguments,
                }
                if on_tool_call_delta:
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(getattr(item, "name", None) or ""),
                        "arguments_delta": "",
                    })
        elif event_type == "response.output_text.delta":
            delta_text = getattr(event, "delta", "") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.function_call_arguments.delta":
            call_id = getattr(event, "call_id", None)
            if call_id and call_id in tool_call_buffers:
                delta = getattr(event, "delta", "") or ""
                current = tool_call_buffers[call_id].get("arguments")
                if not isinstance(current, str):
                    current = ""
                tool_call_buffers[call_id]["arguments"] = current + delta
                if on_tool_call_delta and delta:
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(tool_call_buffers[call_id].get("name") or ""),
                        "arguments_delta": str(delta),
                    })
        elif event_type == "response.function_call_arguments.done":
            call_id = getattr(event, "call_id", None)
            if call_id and call_id in tool_call_buffers:
                arguments = getattr(event, "arguments", None)
                tool_call_buffers[call_id]["arguments"] = arguments
                if on_tool_call_delta:
                    tool_call_args_emitted.add(str(call_id))
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(tool_call_buffers[call_id].get("name") or ""),
                        "arguments": "" if arguments is None else str(arguments),
                    })
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                call_id = getattr(item, "call_id", None)
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = _tool_arguments_source(
                    buf.get("arguments"),
                    getattr(item, "arguments", None),
                )
                if on_tool_call_delta and str(call_id) not in tool_call_args_emitted:
                    tool_call_args_emitted.add(str(call_id))
                    await on_tool_call_delta({
                        "call_id": str(call_id),
                        "name": str(buf.get("name") or getattr(item, "name", None) or ""),
                        "arguments": str(args_raw),
                    })
                args = _parse_tool_call_arguments(
                    args_raw,
                    buf.get("name") or getattr(item, "name", None),
                )
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or getattr(item, 'id', None) or 'fc_0'}",
                        name=buf.get("name") or getattr(item, "name", None) or "",
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            resp = getattr(event, "response", None)
            status = getattr(resp, "status", None) if resp else None
            finish_reason = map_finish_reason(status)
            if resp:
                usage_obj = getattr(resp, "usage", None)
                if usage_obj:
                    usage = {
                        "prompt_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                    }
                for out_item in getattr(resp, "output", None) or []:
                    if getattr(out_item, "type", None) == "reasoning":
                        for s in getattr(out_item, "summary", None) or []:
                            if getattr(s, "type", None) == "summary_text":
                                text = getattr(s, "text", None)
                                if text:
                                    reasoning_content = (reasoning_content or "") + text
        elif event_type in {"error", "response.failed"}:
            detail = getattr(event, "error", None) or getattr(event, "message", None) or event
            raise RuntimeError(f"Response failed: {str(detail)[:500]}")

    return content, tool_calls, finish_reason, usage, reasoning_content
