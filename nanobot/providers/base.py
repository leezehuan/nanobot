"""LLM Provider 抽象基类与通用数据结构。

这个文件是 Provider 子系统的核心骨架。所有具体服务商实现最终都要遵守这里的抽象：

- ``ToolCallRequest``：模型要求调用工具时的统一表示
- ``LLMResponse``：一次模型响应的统一表示
- ``LLMProvider``：所有 Provider 的通用接口与重试逻辑
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import json_repair
from loguru import logger

from nanobot.utils.helpers import image_placeholder_text


@dataclass
class ToolCallRequest:
    """模型发出的工具调用请求。

    不同 Provider 对工具调用的原始格式各不相同，但进入 AgentRunner 后，
    都会先被标准化成这个结构。
    """
    id: str
    name: str
    arguments: Any
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """序列化成 OpenAI 风格的 ``tool_call`` 字典。

        这样后续历史回放和跨 Provider 兼容会更简单，因为可以统一使用一种结构。

        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。
        """
        arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False)
        )
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments,
            },
        }
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


def parse_tool_arguments(arguments: Any) -> Any:
    """解析 Provider 返回的工具参数，但不做过度猜测。

    设计原则很重要：
    - 合法 JSON 对象字符串可以转成 dict
    - 空串可以视为无参调用
    - 但畸形 JSON 或数组/标量，不在这里强行修复

    真正“能不能执行”要交给 ToolRegistry 校验，避免 Provider 层替执行层瞎猜。

    实现方法：按响应或文本结构逐层读取字段，把缺失和异常格式归一化为 nanobot 内部对象。
    """
    if arguments is None:
        return {}
    if not isinstance(arguments, str):
        return arguments

    stripped = arguments.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except Exception:
        return arguments
    return arguments if parsed is None else parsed


def tool_arguments_object_for_replay(arguments: Any) -> dict[str, Any]:
    """仅用于“历史回放”场景，把参数整理成对象形态。

    注意这里和 ``parse_tool_arguments`` 不同：它允许对畸形 JSON 做兼容修复，
    因为它面对的是“旧历史重放协议兼容”，不是“即将执行的新工具调用”。

    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}

    stripped = arguments.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except Exception:
        try:
            parsed = json_repair.loads(stripped)
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_arguments_json_for_replay(arguments: Any) -> str:
    """仅用于历史回放：把参数转成 JSON 对象字符串。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return json.dumps(tool_arguments_object_for_replay(arguments), ensure_ascii=False)


@dataclass
class LLMResponse:
    """一次 LLM 调用的统一响应结构。"""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    retry_after: float | None = None  # Provider 给出的重试等待秒数。
    reasoning_content: str | None = None  # Kimi、DeepSeek-R1、MiMo 等模型的 reasoning 文本。
    thinking_blocks: list[dict] | None = None  # Anthropic 风格的扩展 thinking 块。
    # 当 finish_reason == "error" 时，重试策略会参考下面这些结构化错误元数据。
    error_status_code: int | None = None
    error_kind: str | None = None  # 例如 "timeout"、"connection"。
    error_type: str | None = None  # Provider 语义类型，例如 insufficient_quota。
    error_code: str | None = None  # Provider 语义错误码，例如 rate_limit_exceeded。
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def has_tool_calls(self) -> bool:
        """判断响应中是否包含工具调用。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        return len(self.tool_calls) > 0

    @property
    def should_execute_tools(self) -> bool:
        """判断当前响应是否应该进入“执行工具”阶段。

        不只是“有 tool_calls 就执行”，还要看 finish_reason 是否允许。
        例如某些网关会在 ``refusal`` / ``content_filter`` / ``error`` 情况下注入假工具调用，
        这里要显式挡掉。

        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。
        """
        if not self.has_tool_calls:
            return False
        return self.finish_reason in ("tool_calls", "function_call", "stop")


@dataclass(frozen=True)
class GenerationSettings:
    """生成参数默认值集合。"""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


_SYNTHETIC_USER_CONTENT = "(conversation continued)"


class LLMProvider(ABC):
    """所有 LLM Provider 的抽象基类。

    【你可以把它理解成什么】
    它相当于“面向 AgentRunner 的统一模型驱动接口”。
    上层不关心你底下接的是 OpenAI、Anthropic、Azure，还是本地 Ollama，
    只关心：
    - 给你 messages / tools / model
    - 你返回统一的 ``LLMResponse``
    """

    supports_progress_deltas = False

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _PERSISTENT_MAX_DELAY = 60
    _PERSISTENT_IDENTICAL_ERROR_LIMIT = 10
    _RETRY_HEARTBEAT_CHUNK = 30
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
        "速率限制",
        "访问量过大",
    )
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
    _TRANSIENT_ERROR_KINDS = frozenset({"timeout", "connection"})
    _NON_RETRYABLE_429_ERROR_TOKENS = frozenset({
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "insufficient_balance",
        "credit_balance_too_low",
        "billing_not_active",
        "payment_required",
    })
    _RETRYABLE_429_ERROR_TOKENS = frozenset({
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
        "request_limit_exceeded",
        "requests_limit_exceeded",
        "overloaded_error",
    })
    _NON_RETRYABLE_429_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing not active",
        "insufficient balance",
        "insufficient_balance",
        "credit balance too low",
        "payment required",
        "out of credits",
        "out of quota",
        "exceeded your current quota",
    )
    _RETRYABLE_429_TEXT_MARKERS = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "retry after",
        "try again in",
        "temporarily unavailable",
        "overloaded",
        "concurrency limit",
        "速率限制",
    )

    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        """init。
        
        初始化 LLMProvider 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """清洗待发给 Provider 的消息内容。

        主要修两类问题：
        - 空内容块 / 非法空 assistant 内容
        - 内部专用 ``_meta`` 字段

        实现方法：先复制或规范化输入，再移除 provider 或工具无法接受的字段，并保留可安全回放的信息。
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    ):
                        changed = True
                        continue
                    if isinstance(item, dict) and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)
                if changed:
                    clean = dict(msg)
                    if new_items:
                        clean["content"] = new_items
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        """兼容不同 Provider 风格，从工具 schema 中提取工具名。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        name = tool.get("name")
        if isinstance(name, str):
            return name
        fn = tool.get("function")
        if isinstance(fn, dict):
            fname = fn.get("name")
            if isinstance(fname, str):
                return fname
        return ""

    @classmethod
    def _tool_cache_marker_indices(cls, tools: list[dict[str, Any]]) -> list[int]:
        """返回适合做 prompt cache 标记的工具列表边界索引。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if not tools:
            return []

        tail_idx = len(tools) - 1
        last_builtin_idx: int | None = None
        for i in range(tail_idx, -1, -1):
            if not cls._tool_name(tools[i]).startswith("mcp_"):
                last_builtin_idx = i
                break

        ordered_unique: list[int] = []
        for idx in (last_builtin_idx, tail_idx):
            if idx is not None and idx not in ordered_unique:
                ordered_unique.append(idx)
        return ordered_unique

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """只保留对 Provider 安全的消息字段，并规范 assistant 内容。
        
        实现方法：先复制或规范化输入，再移除 provider 或工具无法接受的字段，并保留可安全回放的信息。"""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
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
        """发送一次非流式对话请求。

        这是所有具体 Provider 必须实现的核心接口。

        实现方法：把统一请求参数转换为当前 provider 的 API 调用，并把返回值解析成 LLMResponse。
        """
        pass

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        """is transient error。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def _is_transient_response(cls, response: LLMResponse) -> bool:
        """判断某次错误是否属于“可重试的临时错误”。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)

        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status == 429:
                return cls._is_retryable_429_response(response)
            if status in cls._RETRYABLE_STATUS_CODES or status >= 500:
                return True

        kind = (response.error_kind or "").strip().lower()
        if kind in cls._TRANSIENT_ERROR_KINDS:
            return True

        return cls._is_transient_error(response.content)

    @classmethod
    def is_arrearage_response(cls, response: LLMResponse) -> bool:
        """检测“欠费/配额耗尽/账单异常”这类重试也无意义的错误。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        if response.error_status_code is not None and int(response.error_status_code) == 402:
            return True

        type_token = cls._normalize_error_token(response.error_type)
        code_token = cls._normalize_error_token(response.error_code)
        if any(
            token in cls._NON_RETRYABLE_429_ERROR_TOKENS
            for token in (type_token, code_token)
            if token is not None
        ):
            return True

        content = (response.content or "").lower()
        return any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS)

    @staticmethod
    def _normalize_error_token(value: Any) -> str | None:
        """normalize error token。
        
        实现方法：把输入统一成内部约定格式，处理大小写、空值、别名或 provider 差异。"""
        if value is None:
            return None
        token = str(value).strip().lower()
        return token or None

    @classmethod
    def _extract_error_type_code(cls, payload: Any) -> tuple[str | None, str | None]:
        """extract error type code。
        
        实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
        data: dict[str, Any] | None = None
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            text = payload.strip()
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    data = parsed
        if not isinstance(data, dict):
            return None, None

        error_obj = data.get("error")
        type_value = data.get("type")
        code_value = data.get("code")
        if isinstance(error_obj, dict):
            type_value = error_obj.get("type") or type_value
            code_value = error_obj.get("code") or code_value

        return cls._normalize_error_token(type_value), cls._normalize_error_token(code_value)

    @classmethod
    def _is_retryable_429_response(cls, response: LLMResponse) -> bool:
        """细分 429：区分是真限流，还是余额/配额不足。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        type_token = cls._normalize_error_token(response.error_type)
        code_token = cls._normalize_error_token(response.error_code)
        semantic_tokens = {
            token for token in (type_token, code_token)
            if token is not None
        }
        if any(token in cls._NON_RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return False

        content = (response.content or "").lower()
        if any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS):
            return False

        if any(token in cls._RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return True
        if any(marker in content for marker in cls._RETRYABLE_429_TEXT_MARKERS):
            return True
        # 即使 429 的细分原因未知，也默认按“等待后重试”处理。
        return True

    @staticmethod
    def _enforce_role_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """修正消息角色交替关系，适配更严格的 Provider 协议。

        很多 Provider 不接受：
        - 最后一条消息是 assistant
        - 连续两条非 system 消息角色相同

        所以这里会在真正请求前做一层协议修复。

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        if not messages:
            return messages

        merged: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if (
                merged
                and role != "system"
                and role not in ("tool",)
                and merged[-1].get("role") == role
                and role in ("user", "assistant")
            ):
                prev = merged[-1]
                if role == "assistant":
                    prev_has_tools = bool(prev.get("tool_calls"))
                    curr_has_tools = bool(msg.get("tool_calls"))
                    if curr_has_tools:
                        merged[-1] = dict(msg)
                        continue
                    if prev_has_tools:
                        continue
                prev_content = prev.get("content") or ""
                curr_content = msg.get("content") or ""
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    prev["content"] = (prev_content + "\n\n" + curr_content).strip()
                else:
                    merged[-1] = dict(msg)
            else:
                merged.append(dict(msg))

        last_popped = None
        while merged and merged[-1].get("role") == "assistant":
            last_popped = merged.pop()

        # 如果删掉结尾 assistant 后只剩 system 消息，很多 Provider 又会拒绝。
        # 这时宁可把最后一条 assistant 临时转成 user，也要保证协议合法。
        if (
            merged
            and last_popped is not None
            and not any(m.get("role") in ("user", "tool") for m in merged)
        ):
            recovered = dict(last_popped)
            recovered["role"] = "user"
            merged.append(recovered)

        # 额外兜底：防止截断后变成 system -> assistant 开头。
        # 某些 Provider（如 GLM）会直接拒绝这种序列。
        for i, msg in enumerate(merged):
            if msg.get("role") != "system":
                if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    merged.insert(i, {"role": "user", "content": _SYNTHETIC_USER_CONTENT})
                break

        return merged

    @staticmethod
    def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """把图片块替换成文本占位，必要时用于错误恢复。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = image_placeholder_text(path, empty="[image omitted]")
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    @staticmethod
    def _strip_image_content_inplace(messages: list[dict[str, Any]]) -> bool:
        """原地把图片块替换成文本占位。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        found = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for i, b in enumerate(content):
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = image_placeholder_text(path, empty="[image omitted]")
                        content[i] = {"type": "text", "text": placeholder}
                        found = True
        return found

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        """包装 ``chat()``：把异常转成标准 ``LLMResponse(error)``。
        
        实现方法：把统一请求参数转换为当前 provider 的 API 调用，并把返回值解析成 LLMResponse。"""
        try:
            return await self.chat(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

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
        """发送一次流式对话请求。

        默认实现会退化成非流式 ``chat``，然后把整段内容一次性当作一个 delta 发出去。
        真正支持原生流式的 Provider 应该重写它。

        实现方法：把普通聊天参数转换为流式请求，逐块消费增量文本、思考内容和工具调用，最后汇总成统一响应。
        """
        _ = on_thinking_delta, on_tool_call_delta
        response = await self.chat(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """包装 ``chat_stream()``：把异常转成标准错误响应。
        
        实现方法：把普通聊天参数转换为流式请求，逐块消费增量文本、思考内容和工具调用，最后汇总成统一响应。"""
        try:
            return await self.chat_stream(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """带重试地调用 ``chat_stream()``。
        
        实现方法：把普通聊天参数转换为流式请求，逐块消费增量文本、思考内容和工具调用，最后汇总成统一响应。"""
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        has_streamed_content = False

        async def _tracking_delta(text: str) -> None:
            """tracking delta。
            
            实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
            nonlocal has_streamed_content
            if text:
                has_streamed_content = True
            if on_content_delta:
                await on_content_delta(text)

        async def _recover_stream() -> None:
            """recover stream。
            
            实现方法：逐块读取上游事件，把文本增量、工具调用增量和完成信号分别转发给调用方。"""
            nonlocal has_streamed_content
            if on_stream_recover:
                await on_stream_recover()
            has_streamed_content = False

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            on_content_delta=_tracking_delta if on_content_delta is not None else None,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
        )
        if on_stream_recover and getattr(self, "supports_stream_recover_callback", False):
            kw["on_stream_recover"] = _recover_stream
        return await self._run_with_retry(
            self._safe_chat_stream,
            kw,
            messages,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
            should_retry_guard=lambda: not has_streamed_content,
            on_stream_recover=_recover_stream if on_stream_recover else None,
        )

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """带重试地调用 ``chat()``。

        如果调用方没有显式传 ``max_tokens / temperature / reasoning_effort``，
        这里会自动使用 ``self.generation`` 中的默认值。

        实现方法：把统一请求参数转换为当前 provider 的 API 调用，并把返回值解析成 LLMResponse。
        """
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        return await self._run_with_retry(
            self._safe_chat,
            kw,
            messages,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
        )

    @classmethod
    def _extract_retry_after(cls, content: str | None) -> float | None:
        """从错误文本里提取“建议重试等待时间”。
        
        实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
            r"wait\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)\s*before retry",
            r"retry[_-]?after[\"'\s:=]+(\d+(?:\.\d+)?)",
        )
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2) if idx < 3 else "s"
            return cls._to_retry_seconds(value, unit)
        return None

    @classmethod
    def _to_retry_seconds(cls, value: float, unit: str | None = None) -> float:
        """to retry seconds。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        normalized_unit = (unit or "s").lower()
        if normalized_unit in {"ms", "milliseconds"}:
            return max(0.1, value / 1000.0)
        if normalized_unit in {"m", "min", "minutes"}:
            return max(0.1, value * 60.0)
        return max(0.1, value)

    @classmethod
    def _extract_retry_after_from_headers(cls, headers: Any) -> float | None:
        """从 HTTP 响应头中提取 retry-after。
        
        实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
        if not headers:
            return None

        def _header_value(name: str) -> Any:
            """header value。
            
            实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
            if hasattr(headers, "get"):
                value = headers.get(name) or headers.get(name.title())
                if value is not None:
                    return value
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if isinstance(key, str) and key.lower() == name.lower():
                        return value
            return None

        with suppress(TypeError, ValueError):
            retry_ms = _header_value("retry-after-ms")
            if retry_ms is not None:
                value = float(retry_ms) / 1000.0
                if value > 0:
                    return value

        retry_after = _header_value("retry-after")
        if retry_after is None:
            return None
        retry_after_text = str(retry_after).strip()
        if not retry_after_text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", retry_after_text):
            return cls._to_retry_seconds(float(retry_after_text), "s")
        try:
            retry_at = parsedate_to_datetime(retry_after_text)
        except Exception:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        remaining = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
        return max(0.1, remaining)

    @classmethod
    def _extract_retry_after_from_response(cls, response: LLMResponse) -> float | None:
        """统一从结构化字段或文本中提取 retry-after。
        
        实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
        if response.error_retry_after_s is not None and response.error_retry_after_s > 0:
            return response.error_retry_after_s
        if response.retry_after is not None and response.retry_after > 0:
            return response.retry_after
        return cls._extract_retry_after(response.content)

    async def _sleep_with_heartbeat(
        self,
        delay: float,
        *,
        attempt: int,
        persistent: bool,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        # 对长时间等待做分段 sleep，这样可以周期性向外报告“仍在重试等待中”。
        """sleep with heartbeat。
        
        实现方法：把等待拆成小片段执行，片段之间发送心跳或检查取消状态。"""
        remaining = max(0.0, delay)
        while remaining > 0:
            if on_retry_wait:
                kind = "persistent retry" if persistent else "retry"
                await on_retry_wait(
                    f"Model request failed, {kind} in {max(1, int(round(remaining)))}s "
                    f"(attempt {attempt})."
                )
            chunk = min(remaining, self._RETRY_HEARTBEAT_CHUNK)
            await asyncio.sleep(chunk)
            remaining -= chunk

    async def _run_with_retry(
        self,
        call: Callable[..., Awaitable[LLMResponse]],
        kw: dict[str, Any],
        original_messages: list[dict[str, Any]],
        *,
        retry_mode: str,
        on_retry_wait: Callable[[str], Awaitable[None]] | None,
        should_retry_guard: Callable[[], bool] | None = None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """通用重试主循环。

        【中文名称】带重试的 LLM 调用

        【功能说明】
        这是 Provider 层最重要的公共逻辑。它包装了 any callable（chat 或 chat_stream），
        在遇到可恢复错误时自动重试。

        【两种重试模式】
        1. standard（默认）：最多重试 3 次，指数退避 1s → 2s → 4s
        2. persistent：无限重试，但相同错误超过 10 次后停止；delay 上限 60s

        【重试决策树】
        1. 如果 response.finish_reason != "error" → 成功，直接返回
        2. 如果有 should_retry_guard 且返回 False：
           - 流式 timeout：尝试 stream_recover 或清空 delta 回调后重试
           - 其他情况：跳过重试，直接返回错误
        3. 如果不是临时错误 → 尝试去掉图片重试（有些 provider 因为图片被拒）
        4. persistent 模式下相同错误超过 10 次 → 停止重试
        5. standard 模式下超过 3 次 → 停止重试
        6. 否则：计算延迟时间 → 执行心跳等待 → 重试

        【retry-after 延迟来源（优先级从高到低）】
        1. response.error_retry_after_s（结构化字段）
        2. response.retry_after（结构化字段）
        3. 从 response.content 文本中正则提取
        4. 默认指数退避值

        【心跳等待机制】
        长时间等待会分段 sleep（每段 30s），在每段开始前通过 on_retry_wait
        回调通知用户"模型请求失败，正在重试中"。

        【参数说明】
        - call: Callable → 要重试的实际调用函数（_safe_chat 或 _safe_chat_stream）
        - kw: dict → 传给 call 的关键字参数字典
        - original_messages: list[dict] → 原始消息（用于去图重试）
        - retry_mode: str → "standard" 或 "persistent"
        - on_retry_wait: callback → 重试等待时的进度通知回调
        - should_retry_guard: callback → 返回 False 时跳过重试
        - on_stream_recover: callback → 流式恢复回调

        【返回值】
        - LLMResponse: 最终响应（成功或失败）

        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。
        """
        attempt = 0
        delays = list(self._CHAT_RETRY_DELAYS)
        persistent = retry_mode == "persistent"
        last_response: LLMResponse | None = None
        last_error_key: str | None = None
        identical_error_count = 0
        while True:
            attempt += 1
            response = await call(**kw)
            if response.finish_reason != "error":
                return response
            last_response = response
            if should_retry_guard is not None and not should_retry_guard():
                is_timeout = (response.error_kind or "").lower() == "timeout"
                if is_timeout:
                    if on_stream_recover:
                        logger.warning(
                            "LLM stream stalled after content was emitted; "
                            "starting a new stream segment and retrying"
                        )
                        await on_stream_recover()
                    else:
                        logger.warning(
                            "LLM stream stalled after content was emitted; "
                            "suppressing delta callbacks and retrying"
                        )
                        kw.setdefault("on_content_delta", None)
                        kw["on_content_delta"] = None
                        kw["on_thinking_delta"] = None
                        kw["on_tool_call_delta"] = None
                        should_retry_guard = None
                else:
                    logger.warning(
                        "LLM stream failed after content was emitted; skipping retry"
                    )
                    return response
            error_key = ((response.content or "").strip().lower() or None)
            if error_key and error_key == last_error_key:
                identical_error_count += 1
            else:
                last_error_key = error_key
                identical_error_count = 1 if error_key else 0

            if not self._is_transient_response(response):
                # 某些 Provider 失败只是因为图片内容不被接受；
                # 这时尝试去掉图片再重试一次，尽量保住当前 turn。
                stripped = self._strip_image_content(original_messages)
                if stripped is not None and stripped != kw["messages"]:
                    logger.warning(
                        "Non-transient LLM error with image content, retrying without images"
                    )
                    retry_kw = dict(kw)
                    retry_kw["messages"] = stripped
                    result = await call(**retry_kw)
                    # 如果去图后恢复成功，就永久移除原消息里的图片，
                    # 避免后续迭代反复触发同样的“失败 -> 去图重试”循环。
                    if result.finish_reason != "error":
                        self._strip_image_content_inplace(original_messages)
                    return result
                return response

            if persistent and identical_error_count >= self._PERSISTENT_IDENTICAL_ERROR_LIMIT:
                logger.warning(
                    "Stopping persistent retry after {} identical transient errors: {}",
                    identical_error_count,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(
                        f"Persistent retry stopped after {identical_error_count} identical errors."
                    )
                return response

            if not persistent and attempt > len(delays):
                logger.warning(
                    "LLM request failed after {} retries, giving up: {}",
                    attempt,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(
                        f"Model request failed after {attempt} retries, giving up."
                    )
                break

            base_delay = delays[min(attempt - 1, len(delays) - 1)]
            delay = self._extract_retry_after_from_response(response) or base_delay
            if persistent:
                delay = min(delay, self._PERSISTENT_MAX_DELAY)

            logger.warning(
                "LLM transient error (attempt {}{}), retrying in {}s: {}",
                attempt,
                "+" if persistent and attempt > len(delays) else f"/{len(delays)}",
                int(round(delay)),
                (response.content or "")[:120].lower(),
            )
            await self._sleep_with_heartbeat(
                delay,
                attempt=attempt,
                persistent=persistent,
                on_retry_wait=on_retry_wait,
            )

        return last_response if last_response is not None else await call(**kw)

    @abstractmethod
    def get_default_model(self) -> str:
        """返回该 Provider 的默认模型名。
        
        实现方法：优先从显式参数或实例状态读取目标值，缺失时回退到默认配置，并把结果整理成调用方期望的类型。"""
        pass
