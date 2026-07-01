"""Anthropic Provider：直接对接 Claude 原生 SDK。

和 OpenAI-compatible Provider 不同，这里不是走“兼容层”，
而是直接按 Anthropic Messages API 的规则来组消息、发请求、收流式结果。

这个文件很适合学习“不同大模型厂商的消息协议到底哪里不一样”。
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import string
from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    tool_arguments_object_for_replay,
)

_ALNUM = string.ascii_letters + string.digits


def _gen_tool_id() -> str:
    """gen tool id。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return "toolu_" + "".join(secrets.choice(_ALNUM) for _ in range(22))


class AnthropicProvider(LLMProvider):
    """LLM provider using the native Anthropic SDK for Claude models.

    【中文名称】Anthropic Provider（Claude 原生适配器）

    【功能说明】
    这是与 OpenAI-compatible Provider 并列的另一条 Provider 通路。
    不走 OpenAI 兼容层，而是直接按 Anthropic Messages API 的规则来：
    - 消息格式转换（OpenAI chat 格式 → Anthropic Messages 格式）
    - prompt 缓存控制（cache_control marker）
    - 扩展思考（extended thinking）
    - 工具调用与流式响应

    【核心流程】
    1. 构造函数：初始化 AsyncAnthropic 客户端，处理 base_url 规范化
    2. _convert_messages：把 nanobot 内部消息转换成 Anthropic 格式
    3. _merge_consecutive：规范化消息序列（合并连续同 role、去尾部 assistant 等）
    4. _build_kwargs：组装 API 调用参数（含 thinking 预算、缓存标记）
    5. chat / chat_stream：对外公开的同步/流式调用入口
    6. _parse_response：把 Anthropic 响应解析成统一的 LLMResponse

    Handles message format conversion (OpenAI → Anthropic Messages API),
    prompt caching, extended thinking, tool calls, and streaming.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        extra_headers: dict[str, str] | None = None,
    ):
        """init。
        
        初始化 AnthropicProvider 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        from anthropic import AsyncAnthropic

        client_kw: dict[str, Any] = {}
        if api_key:
            client_kw["api_key"] = api_key
        if api_base:
            client_kw["base_url"] = self._normalize_base_url(api_base)
        if extra_headers:
            client_kw["default_headers"] = extra_headers
        # Keep retries centralized in LLMProvider._run_with_retry to avoid retry amplification.
        client_kw["max_retries"] = 0
        self._client = AsyncAnthropic(**client_kw)

    @staticmethod
    def _normalize_base_url(api_base: str) -> str:
        """Anthropic SDK appends /v1 to request paths internally.
        
        实现方法：把输入统一成内部约定格式，处理大小写、空值、别名或 provider 差异。"""
        normalized = api_base.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[: -len("/v1")]
        return normalized

    @classmethod
    def _handle_error(cls, e: Exception) -> LLMResponse:
        """将 Anthropic SDK 异常转换为统一的 LLMResponse 错误格式。

        【中文名称】错误处理

        【功能说明】
        这是异常 → 标准 LLMResponse 的转换枢纽。从 Anropic SDK 异常中提取：

        1. HTTP 响应体 / headers → 错误消息和状态码
        2. x-should-retry 头 → 是否应该重试
        3. retry-after 头 → 重试等待秒数
        4. 异常类名 → 错误类型（timeout / connection 等）
        5. 错误载荷中的 type 和 code 字段

        这样上层 AgentRunner 就能用统一逻辑处理所有 Provider 的异常，
        而不需要关心具体是 Anthropic 还是 OpenAI。

        【参数说明】
        - e: Exception → Anthropic SDK 抛出的异常对象

        【返回值】
        - LLMResponse → 包含错误信息的 LLM 响应，finish_reason="error"

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        payload = (
            getattr(e, "body", None)
            or getattr(e, "doc", None)
            or getattr(response, "text", None)
        )
        if payload is None and response is not None:
            response_json = getattr(response, "json", None)
            if callable(response_json):
                try:
                    payload = response_json()
                except Exception:
                    payload = None
        payload_text = payload if isinstance(payload, str) else str(payload) if payload is not None else ""
        msg = f"Error: {payload_text.strip()[:500]}" if payload_text.strip() else f"Error calling LLM: {e}"
        retry_after = cls._extract_retry_after_from_headers(headers)
        if retry_after is None:
            retry_after = LLMProvider._extract_retry_after(msg)

        status_code = getattr(e, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        should_retry: bool | None = None
        if headers is not None:
            raw = headers.get("x-should-retry")
            if isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered == "true":
                    should_retry = True
                elif lowered == "false":
                    should_retry = False

        error_kind: str | None = None
        error_name = e.__class__.__name__.lower()
        if "timeout" in error_name:
            error_kind = "timeout"
        elif "connection" in error_name:
            error_kind = "connection"
        error_type, error_code = LLMProvider._extract_error_type_code(payload)

        return LLMResponse(
            content=msg,
            finish_reason="error",
            retry_after=retry_after,
            error_status_code=int(status_code) if status_code is not None else None,
            error_kind=error_kind,
            error_type=error_type,
            error_code=error_code,
            error_retry_after_s=retry_after,
            error_should_retry=should_retry,
        )

    @staticmethod
    def _strip_prefix(model: str) -> str:
        """strip prefix。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if model.startswith("anthropic/"):
            return model[len("anthropic/"):]
        return model

    # ------------------------------------------------------------------
    # 消息格式转换：OpenAI chat 格式 -> Anthropic Messages API
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        """把 nanobot 内部消息转换成 Anthropic 所需的 ``(system, messages)``。

        【中文名称】消息格式转换

        【功能说明】
        这是 Anthropic Provider 最核心的转换逻辑。OpenAI 和 Anthropic 的消息
        格式差异很大，此函数承担了"格式翻译"的角色：

        1. system 消息：从消息列表中抽出来，单独作为 system 参数（Anthropic 的
           system 是独立的顶层参数，不混在 messages 数组里）
        2. tool 消息：转换成 ``tool_result`` block，附加到前一条 user 消息上
           （Anthropic 要求 tool_result 必须在 user 消息内）
        3. assistant 消息：拆成 text + tool_use + thinking 等多个 content block
        4. user 消息：转换图片 URL 块（OpenAI image_url → Anthropic image block）
        5. 最后调用 _merge_consecutive 做序列规范化

        【参数说明】
        - messages: list[dict] → nanobot 内部消息列表（OpenAI chat 格式）

        【返回值】
        - (system, messages) 元组，可直接传入 Anthropic Messages API
        - system 可能是 str 或 list（启用缓存控制时包装成 list）
        - messages 是规范化后的 Anthropic 格式消息列表

        实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。
        """
        system: str | list[dict[str, Any]] = ""
        raw: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                system = content if isinstance(content, (str, list)) else str(content or "")
                continue

            if role == "tool":
                block = self._tool_result_block(msg)
                if raw and raw[-1]["role"] == "user":
                    prev_c = raw[-1]["content"]
                    if isinstance(prev_c, list):
                        prev_c.append(block)
                    else:
                        raw[-1]["content"] = [
                            {"type": "text", "text": prev_c or ""}, block,
                        ]
                else:
                    raw.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                raw.append({"role": "assistant", "content": self._assistant_blocks(msg)})
                continue

            if role == "user":
                raw.append({
                    "role": "user",
                    "content": self._convert_user_content(content),
                })
                continue

        return system, self._merge_consecutive(raw)

    @staticmethod
    def _tool_result_block(msg: dict[str, Any]) -> dict[str, Any]:
        """tool result block。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        content = msg.get("content")
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id", ""),
        }
        if isinstance(content, list):
            block["content"] = AnthropicProvider._convert_user_content(content)
        elif isinstance(content, str):
            block["content"] = content
        else:
            block["content"] = str(content) if content else ""
        return block

    @staticmethod
    def _assistant_blocks(msg: dict[str, Any]) -> list[dict[str, Any]]:
        """assistant blocks。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        blocks: list[dict[str, Any]] = []
        content = msg.get("content")

        for tb in msg.get("thinking_blocks") or []:
            if isinstance(tb, dict) and tb.get("type") == "thinking":
                blocks.append({
                    "type": "thinking",
                    "thinking": tb.get("thinking", ""),
                    "signature": tb.get("signature", ""),
                })

        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                blocks.append(item if isinstance(item, dict) else {"type": "text", "text": str(item)})

        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or _gen_tool_id(),
                "name": func.get("name", ""),
                "input": tool_arguments_object_for_replay(args),
            })

        return blocks or [{"type": "text", "text": ""}]

    @staticmethod
    def _convert_user_content(content: Any) -> Any:
        """转换用户消息内容，并把 ``image_url`` 块改写成 Anthropic 图片块。
        
        实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。"""
        if isinstance(content, str) or content is None:
            return content or "(empty)"
        if not isinstance(content, list):
            return str(content)

        result: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                result.append({"type": "text", "text": str(item)})
                continue
            if item.get("type") == "image_url":
                converted = AnthropicProvider._convert_image_block(item)
                if converted:
                    result.append(converted)
                continue
            if not item.get("type"):
                # Anthropic 要求每个 content block 都显式带 ``type``。
                # 如果某个工具返回了裸 dict，这里宁可把它包成 text block，
                # 也不要把非法结构直接发给 API。
                result.append({"type": "text", "text": str(item)})
                continue
            result.append(item)
        return result or "(empty)"

    @staticmethod
    def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
        """把 OpenAI 风格 ``image_url`` block 转成 Anthropic 风格图片块。
        
        实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。"""
        url = (block.get("image_url") or {}).get("url", "")
        if not url:
            return None
        m = re.match(r"data:(image/\w+);base64,(.+)", url, re.DOTALL)
        if m:
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)},
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }

    @staticmethod
    def _has_tool_use(msg: dict[str, Any]) -> bool:
        """判断消息里是否包含 ``tool_use`` block。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )

    @staticmethod
    def _merge_consecutive(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把消息序列规范化成 Anthropic ``/messages`` 可接受的形式。

        【中文名称】消息序列规范化

        【功能说明】
        Anthropic 对消息序列的要求比 OpenAI 更严格，此函数执行三条规则：

        规则 1 —— 合并连续同 role 消息：
          如果相邻两条消息的 role 相同（比如连续两条 user），就把它们的
          content 合并成一条。Anthropic 要求 messages 数组严格交替 role。

        规则 2 —— 去掉尾部 assistant 预填充：
          对话不能以 assistant 结尾。如果尾部是 assistant 且不含 tool_use，
          就把它转成 user 消息；如果只剩这一条 assistant 了，就转成 user 保留内容。

        规则 3 —— 禁止 assistant 开头：
          如果第一条是 assistant（且不含 tool_use），在前面补一条合成 user opener。

        规则 2/3 的特殊处理：如果 assistant 消息里包含 tool_use block，说明
        这是工具调用中间状态，不能随意转 role，否则会破坏 tool_use/tool_result
        的语义绑定关系。

        【参数说明】
        - msgs: list[dict] → 已做格式转换但尚未规范化的消息列表

        【返回值】
        - list[dict] → 符合 Anthropic Messages API 角色交替规则的消息列表

        Anthropic 对消息序列的要求比 OpenAI 更严格：

        1. 连续相同 role 的消息要合并
        2. 对话不能以 assistant 结尾
        3. 对话不能以 assistant 开头

        此外，Anthropic 的 ``tool_use`` block 是放在 ``content`` 里的，
        不能像 OpenAI 那样简单地只看 ``tool_calls`` 字段。

        实现方法：按顺序合并相邻或同类数据，并在冲突时保留更明确的新值。
        """
        merged: list[dict[str, Any]] = []
        for msg in msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_c = merged[-1]["content"]
                cur_c = msg["content"]
                if isinstance(prev_c, str):
                    prev_c = [{"type": "text", "text": prev_c}]
                if isinstance(cur_c, str):
                    cur_c = [{"type": "text", "text": cur_c}]
                if isinstance(cur_c, list):
                    prev_c.extend(cur_c)
                merged[-1]["content"] = prev_c
            else:
                merged.append(msg)

        # 规则 2：去掉尾部 assistant 预填充消息，Anthropic 不接受这种结尾。
        last_popped: dict[str, Any] | None = None
        while merged and merged[-1].get("role") == "assistant":
            last_popped = merged.pop()

        # 修复策略：如果删完后整段历史空了，就尽量把最后一条 assistant
        # 转成 user，避免直接变成空 messages 数组。
        if (
            not merged
            and last_popped is not None
            and not AnthropicProvider._has_tool_use(last_popped)
        ):
            merged.append({"role": "user", "content": last_popped.get("content")})

        # 规则 3：如果第一条是 assistant，就补一条合成 user opener。
        # 但如果 assistant 里带 tool_use，就别乱补，避免把 tool_use/tool_result
        # 关系打乱，造成更难定位的问题。
        if (
            merged
            and merged[0].get("role") == "assistant"
            and not AnthropicProvider._has_tool_use(merged[0])
        ):
            merged.insert(0, {"role": "user", "content": "(conversation continued)"})

        return merged

    # ------------------------------------------------------------------
    # 工具定义转换
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """convert tools。
        
        实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。"""
        if not tools:
            return None
        result = []
        for tool in tools:
            func = tool.get("function", tool)
            entry: dict[str, Any] = {
                "name": func.get("name", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            desc = func.get("description")
            if desc:
                entry["description"] = desc
            if "cache_control" in tool:
                entry["cache_control"] = tool["cache_control"]
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
        thinking_enabled: bool = False,
    ) -> dict[str, Any] | None:
        """convert tool choice。
        
        实现方法：按目标协议逐项转换 role、content、tool call 等字段，并跳过或降级无法表达的结构。"""
        if thinking_enabled:
            return {"type": "auto"}
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    # ------------------------------------------------------------------
    # Prompt caching
    # ------------------------------------------------------------------

    @classmethod
    def _apply_cache_control(
        cls,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
        """apply cache control。
        
        实现方法：在输入对象的副本或当前运行时状态上逐项写入变更，再把相关缓存、子组件或事件同步更新。"""
        marker = {"type": "ephemeral"}

        if isinstance(system, str) and system:
            system = [{"type": "text", "text": system, "cache_control": marker}]
        elif isinstance(system, list) and system:
            system = list(system)
            system[-1] = {**system[-1], "cache_control": marker}

        new_msgs = list(messages)
        if len(new_msgs) >= 3:
            m = new_msgs[-2]
            c = m.get("content")
            if isinstance(c, str):
                new_msgs[-2] = {**m, "content": [{"type": "text", "text": c, "cache_control": marker}]}
            elif isinstance(c, list) and c:
                nc = list(c)
                nc[-1] = {**nc[-1], "cache_control": marker}
                new_msgs[-2] = {**m, "content": nc}

        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": marker}

        return system, new_msgs, new_tools

    # ------------------------------------------------------------------
    # 构建 API 请求参数
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        supports_caching: bool = True,
    ) -> dict[str, Any]:
        """组装一次完整的 Anthropic Messages API 调用参数。

        【中文名称】构建 API 请求参数

        【功能说明】
        这是 ``chat()`` 和 ``chat_stream()`` 共享的参数组装逻辑，负责：

        1. 模型名规范化：去掉 ``anthropic/`` 前缀，提取裸模型名
        2. 消息格式转换：调用 _convert_messages 做 OpenAI→Anthropic 格式转换
        3. 工具定义转换：把 OpenAI 风格的 function 定义转成 Anthropic input_schema
        4. 缓存控制注入：在 system、倒数第 2 条消息、工具的尾部打上 cache_control marker
        5. 扩展思考处理：
           - "adaptive"：让模型自己决定思考量
           - "low"/"medium"/"high"：映射为 1024/4096/8192+ token 的 thinking budget
        6. 特殊模型兼容：claude-opus-4-7 不支持 temperature，必须省略

        【参数说明】
        - messages: 内部消息列表（OpenAI 格式）
        - tools: 工具定义列表
        - model: 模型名（可能带 anthropic/ 前缀）
        - max_tokens: 最大输出 token
        - temperature: 采样温度
        - reasoning_effort: 推理力度（"low"/"medium"/"high"/"adaptive"/None）
        - tool_choice: 工具选择策略
        - supports_caching: 是否启用 prompt 缓存（默认 True）

        【返回值】
        - dict → 可直接解包传给 ``self._client.messages.create(**kwargs)`` 的参数字典

        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。
        """
        model_name = self._strip_prefix(model or self.default_model)
        system, anthropic_msgs = self._convert_messages(self._sanitize_empty_content(messages))
        anthropic_tools = self._convert_tools(tools)

        if supports_caching:
            system, anthropic_msgs, anthropic_tools = self._apply_cache_control(
                system, anthropic_msgs, anthropic_tools,
            )

        max_tokens = max(1, max_tokens)
        thinking_enabled = bool(reasoning_effort) and reasoning_effort.lower() != "none"

        # claude-opus-4-7 已经完全弃用 temperature；
        # 只要带上就会直接 400。
        omit_temperature = "opus-4-7" in model_name

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens,
        }

        if system:
            kwargs["system"] = system

        if reasoning_effort == "adaptive":
            # Adaptive thinking：由模型自己决定是否思考、思考多少。
            # 某些 Claude 新模型支持这种模式，并会自动在工具调用之间交错思考。
            kwargs["thinking"] = {"type": "adaptive"}
            if not omit_temperature:
                kwargs["temperature"] = 1.0
        elif thinking_enabled:
            budget_map = {"low": 1024, "medium": 4096, "high": max(8192, max_tokens)}
            budget = budget_map.get(reasoning_effort.lower(), 4096)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(max_tokens, budget + 4096)
            if not omit_temperature:
                kwargs["temperature"] = 1.0
        elif not omit_temperature:
            kwargs["temperature"] = temperature

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            tc = self._convert_tool_choice(tool_choice, thinking_enabled)
            if tc:
                kwargs["tool_choice"] = tc

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        return kwargs

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """把 Anthropic SDK 的原始响应解析成统一的 LLMResponse。

        【中文名称】响应解析

        【功能说明】
        遍历 Anthropic 响应的 content blocks，把不同类型的 block
        分别提取为 text、tool_call、thinking_block：

        - text block → content_parts 列表
        - tool_use block → ToolCallRequest（含 id/name/arguments）
        - thinking block → thinking_blocks 列表（保留 signature）

        同时从 Anthropic 的 stop_reason 映射到统一的 finish_reason：
        - "tool_use" → "tool_calls"
        - "end_turn" → "stop"
        - "max_tokens" → "length"

        最后计算 token 用量统计，包含 prompt caching 的 cache_read 和
        cache_creation 信息。

        【参数说明】
        - response: Anthropic SDK 返回的 Message 对象

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体，包含文本、工具调用、思考块和用量信息

        实现方法：按响应或文本结构逐层读取字段，把缺失和异常格式归一化为 nanobot 内部对象。
        """
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        thinking_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
            elif block.type == "thinking":
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": getattr(block, "signature", ""),
                })

        stop_map = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}
        finish_reason = stop_map.get(response.stop_reason or "", response.stop_reason or "stop")

        usage: dict[str, int] = {}
        if response.usage:
            input_tokens = response.usage.input_tokens
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            total_prompt_tokens = input_tokens + cache_creation + cache_read
            usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": total_prompt_tokens + response.usage.output_tokens,
            }
            for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                val = getattr(response.usage, attr, 0)
                if val:
                    usage[attr] = val
            # Normalize to cached_tokens for downstream consistency.
            if cache_read:
                usage["cached_tokens"] = cache_read

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thinking_blocks=thinking_blocks or None,
        )

    # ------------------------------------------------------------------
    # 对外公开 API
    # ------------------------------------------------------------------

    @staticmethod
    def _is_streaming_required_error(e: Exception) -> bool:
        """判断异常是否代表 Anthropic 要求本次调用必须改走流式模式。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        return isinstance(e, ValueError) and "streaming is required" in str(e).lower()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """发送非流式对话请求到 Anthropic Messages API。

        【中文名称】非流式对话

        【功能说明】
        这是 Anthropic Provider 的同步调用入口。一次完整的调用流程：

        1. 调用 _build_kwargs 组装参数（消息转换、缓存注入、thinking 预算）
        2. 通过 AsyncAnthropic SDK 发起 messages.create 调用
        3. 如果遇到 "streaming is required" 异常（max_tokens + thinking 预算
           超出服务端超时上限），自动降级走 chat_stream 重试
        4. 调用 _parse_response 将原始响应解析为统一的 LLMResponse

        【参数说明】
        - messages: 消息历史列表（OpenAI chat 格式）
        - tools: 工具定义列表（可选）
        - model: 模型名（可选，默认使用 default_model）
        - max_tokens: 最大输出 token 数（默认 4096）
        - temperature: 采样温度（默认 0.7）
        - reasoning_effort: 推理力度（None/"low"/"medium"/"high"/"adaptive"）
        - tool_choice: 工具选择策略

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体

        实现方法：把统一请求参数转换为当前 provider 的 API 调用，并把返回值解析成 LLMResponse。
        """
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        try:
            response = await self._client.messages.create(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            if self._is_streaming_required_error(e):
                # 当 max_tokens 加上 thinking 预算后可能让请求超过服务端超时上限时，
                # Anthropic SDK 会拒绝非流式调用。这里自动改走 streaming 重试，
                # 上层就不需要感知这条厂商特定限制。
                return await self.chat_stream(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            return self._handle_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """发送流式对话请求到 Anthropic Messages API。

        【中文名称】流式对话

        【功能说明】
        这是 Anthropic Provider 的流式调用入口。与 chat() 不同，它会通过
        SSE 事件流实时推送增量内容给上层。

        完整的流式处理流程：
        1. 调用 _build_kwargs 组装参数（同 chat()）
        2. 通过 AsyncAnthropic SDK 发起 messages.stream 调用
        3. 逐帧解析 SSE 事件流：
           - content_block_start：检测 tool_use block 的开始，记录 call_id 和 name
           - thinking_delta：发射增量 thinking 文本
           - text_delta：发射增量回复文本
           - input_json_delta：发射增量工具参数 JSON
        4. 空闲超时保护：NANOBOT_STREAM_IDLE_TIMEOUT_S 环境变量控制（默认 90s）
           关键：超时以"任何 SSE 事件"为活着的证据，而非仅文本 token。
           这样模型长时间 thinking 时不会误判超时。
        5. 流结束后调用 get_final_message 获取完整响应，再走 _parse_response 解析

        【参数说明】
        - messages: 消息历史列表（OpenAI chat 格式）
        - tools: 工具定义列表（可选）
        - model: 模型名（可选）
        - max_tokens: 最大输出 token 数（默认 4096）
        - temperature: 采样温度（默认 0.7）
        - reasoning_effort: 推理力度
        - tool_choice: 工具选择策略
        - on_content_delta: 文本增量回调 → 每收到一段新文本就调用一次
        - on_thinking_delta: 思考增量回调 → 每收到一段新 thinking 就调用一次
        - on_tool_call_delta: 工具调用增量回调 → 收到工具参数 JSON 片段时调用

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体

        实现方法：把普通聊天参数转换为流式请求，逐块消费增量文本、思考内容和工具调用，最后汇总成统一响应。
        """
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        idle_timeout_s = int(os.environ.get("NANOBOT_STREAM_IDLE_TIMEOUT_S", "90"))
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                if on_content_delta or on_thinking_delta or on_tool_call_delta:
                    # 空闲超时必须跟踪“任何 SSE 事件”，而不只是文本 token。
                    # 否则模型长时间 thinking 时，明明连接还活着，也会被误判超时。
                    tool_blocks: dict[int, dict[str, str]] = {}
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.__anext__(),
                                timeout=idle_timeout_s,
                            )
                        except StopAsyncIteration:
                            break
                        if chunk.type == "content_block_start":
                            block = getattr(chunk, "content_block", None)
                            if getattr(block, "type", None) == "tool_use":
                                index = int(getattr(chunk, "index", 0) or 0)
                                state = {
                                    "call_id": str(getattr(block, "id", "") or ""),
                                    "name": str(getattr(block, "name", "") or ""),
                                }
                                tool_blocks[index] = state
                                if on_tool_call_delta:
                                    await on_tool_call_delta({
                                        "index": index,
                                        **state,
                                        "arguments_delta": "",
                                    })
                        elif (
                            chunk.type == "content_block_delta"
                            and getattr(chunk.delta, "type", None) == "thinking_delta"
                        ):
                            piece = getattr(chunk.delta, "thinking", None) or ""
                            if piece and on_thinking_delta:
                                await on_thinking_delta(piece)
                        elif (
                            chunk.type == "content_block_delta"
                            and getattr(chunk.delta, "type", None) == "text_delta"
                        ):
                            text = getattr(chunk.delta, "text", None) or ""
                            if text and on_content_delta:
                                await on_content_delta(text)
                        elif (
                            chunk.type == "content_block_delta"
                            and getattr(chunk.delta, "type", None) == "input_json_delta"
                        ):
                            partial = getattr(chunk.delta, "partial_json", None) or ""
                            if partial and on_tool_call_delta:
                                index = int(getattr(chunk, "index", 0) or 0)
                                state = tool_blocks.get(index, {})
                                await on_tool_call_delta({
                                    "index": index,
                                    "call_id": state.get("call_id", ""),
                                    "name": state.get("name", ""),
                                    "arguments_delta": partial,
                                })
                response = await asyncio.wait_for(
                    stream.get_final_message(),
                    timeout=idle_timeout_s,
                )
            return self._parse_response(response)
        except asyncio.TimeoutError:
            return LLMResponse(
                content=(
                    f"Error calling LLM: stream stalled for more than "
                    f"{idle_timeout_s} seconds"
                ),
                finish_reason="error",
                error_kind="timeout",
            )
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        """get default model。
        
        实现方法：优先从显式参数或实例状态读取目标值，缺失时回退到默认配置，并把结果整理成调用方期望的类型。"""
        return self.default_model
