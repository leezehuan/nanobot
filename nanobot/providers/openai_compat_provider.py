"""OpenAI 兼容 Provider：统一接入大量“类 OpenAI”模型服务。

这是 nanobot 里最重要的 Provider 实现之一，因为很多后端都会被收敛到这里，
例如各类 OpenAI-compatible 网关、本地模型服务、以及部分第三方平台。

你可以把这个文件理解成“OpenAI 方言翻译器 + 稳定性增强层”，它主要负责：

1. 把 nanobot 内部统一消息格式转换成 OpenAI / Responses API 请求
2. 兼容不同厂商在 tool call、thinking、reasoning、timeout 上的细微差异
3. 在 Responses API 和传统 chat.completions 之间做策略选择与回退
4. 统一解析返回结果，产出 nanobot 内部标准 ``LLMResponse``
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import secrets
import string
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    parse_tool_arguments,
    tool_arguments_json_for_replay,
)
from nanobot.providers.openai_responses import (
    consume_sdk_stream,
    convert_messages,
    convert_tools,
    parse_response_output,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI as AsyncOpenAIType

    from nanobot.providers.registry import ProviderSpec

# 模块级占位符：真正第一次用到时才懒加载 AsyncOpenAI。
# 之所以保留成模块级名字，而不是藏进类属性里，
# 是为了让测试里的 ``unittest.mock.patch`` 更容易替换它。
AsyncOpenAI: Any = None

_ALLOWED_MSG_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "name",
    "reasoning_content", "extra_content",
})
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}
_KIMI_THINKING_MODELS: frozenset[str] = frozenset({
    "kimi-k2.5",
    "kimi-k2.6",
    "k2.6-code-preview",
})
# 按小米文档整理出的支持 thinking 的 MiMo 模型集合。
# mimo-v2-flash 没放进来，因为它不支持 thinking。
_MIMO_THINKING_MODELS: frozenset[str] = frozenset({
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "mimo-v2-pro",
    "mimo-v2-omni",
})
_OPENAI_COMPAT_REQUEST_TIMEOUT_S = 120.0

# 把 ``ProviderSpec.thinking_style`` 映射到具体的 extra_body 生成函数。
# 这样“thinking 功能如何落到各家 wire format”就集中收敛在这一处，
# 而不是散落在请求构造逻辑各处。
_THINKING_STYLE_MAP: dict[str, Any] = {
    "thinking_type": lambda on: {"thinking": {"type": "enabled" if on else "disabled"}},
    "enable_thinking": lambda on: {"enable_thinking": on},
    "reasoning_split": lambda on: {"reasoning_split": on},
}
_GATEWAY_REASONING_STYLE_MAP: dict[str, Any] = {
    "reasoning_effort": lambda effort: {"reasoning": {"effort": effort}},
}
_MODEL_THINKING_STYLES: dict[str, str] = {
    **dict.fromkeys(_KIMI_THINKING_MODELS, "thinking_type"),
    **dict.fromkeys(_MIMO_THINKING_MODELS, "thinking_type"),
}


def _model_slug(model_name: str) -> str:
    return model_name.lower().rsplit("/", 1)[-1]


def _requires_max_completion_tokens(model_name: str) -> bool:
    """判断某些模型是否必须使用 ``max_completion_tokens`` 而不能用 ``max_tokens``。"""
    slug = _model_slug(model_name)
    return "gpt-5" in slug or any(
        slug == p or slug.startswith((p + "-", p + ".")) for p in ("o1", "o3", "o4")
    )


def _model_thinking_style(model_name: str) -> str:
    return _MODEL_THINKING_STYLES.get(_model_slug(model_name), "")


def _thinking_styles_for(spec: ProviderSpec | None, model_name: str) -> list[str]:
    styles: list[str] = []
    if spec and spec.thinking_style:
        styles.append(spec.thinking_style)
    model_style = _model_thinking_style(model_name)
    if model_style and model_style not in styles:
        styles.append(model_style)
    return styles


def _thinking_extra_body(style: str, thinking_enabled: bool) -> dict[str, Any] | None:
    builder = _THINKING_STYLE_MAP.get(style)
    return builder(thinking_enabled) if builder else None


def _gateway_reasoning_extra_body(style: str, effort: str | None) -> dict[str, Any] | None:
    if not effort:
        return None
    builder = _GATEWAY_REASONING_STYLE_MAP.get(style)
    return builder(effort) if builder else None


def _openai_compat_timeout_s() -> float:
    """返回 OpenAI-compatible Provider 使用的统一请求超时。"""
    return _float_env("NANOBOT_OPENAI_COMPAT_TIMEOUT_S", _OPENAI_COMPAT_REQUEST_TIMEOUT_S)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid {}={!r}; using {}", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive {}={!r}; using {}", name, raw, default)
        return default
    return value


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    """同时兼容 dict / 对象属性两种读取方式。"""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """尽量把任意对象转成 dict；失败或为空时返回 ``None``。"""
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


def _extract_tc_extras(tc: Any) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """提取工具调用中的扩展字段。

    返回三元组：

    - ``extra_content``
    - ``provider_specific_fields``
    - ``function_provider_specific_fields``

    这样 nanobot 即使面对某些厂商的非标准字段，也能尽量无损保存下来。
    """
    extra_content = _coerce_dict(_get(tc, "extra_content"))

    tc_dict = _coerce_dict(tc)
    prov = None
    fn_prov = None
    if tc_dict is not None:
        leftover = {k: v for k, v in tc_dict.items()
                    if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None}
        if leftover:
            prov = leftover
        fn = _coerce_dict(tc_dict.get("function"))
        if fn is not None:
            fn_leftover = {k: v for k, v in fn.items()
                          if k not in _STANDARD_FN_KEYS and v is not None}
            if fn_leftover:
                fn_prov = fn_leftover
    else:
        prov = _coerce_dict(_get(tc, "provider_specific_fields"))
        fn_obj = _get(tc, "function")
        if fn_obj is not None:
            fn_prov = _coerce_dict(_get(fn_obj, "provider_specific_fields"))

    return extra_content, prov, fn_prov


def _uses_openrouter_attribution(spec: "ProviderSpec | None", api_base: str | None) -> bool:
    """判断当前请求是否应该默认带上 OpenRouter attribution headers。"""
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())


_RESPONSES_FAILURE_THRESHOLD = 3
_RESPONSES_PROBE_INTERVAL_S = 300  # 5 minutes


def _is_local_endpoint(
    spec: "ProviderSpec | None",
    api_base: str | None,
) -> bool:
    """判断当前 endpoint 是否是本地或局域网模型服务。"""
    if spec and spec.is_local:
        return True
    if not api_base:
        return False
    raw = api_base.strip().lower()
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:
        host = parsed.hostname
    except ValueError:
        return False
    if host in {"localhost", "host.docker.internal"}:
        return True
    if not host:
        return False
    try:
        addr = ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _is_direct_openai_base(api_base: str | None) -> bool:
    """判断 base URL 是否直连 OpenAI 官方，而不是某个兼容网关。"""
    if not api_base:
        return True
    normalized = api_base.strip().lower().rstrip("/")
    return "api.openai.com" in normalized and "openrouter" not in normalized


def _responses_circuit_key(
    model: str | None,
    default_model: str,
    reasoning_effort: str | None,
) -> str:
    model_name = (model or default_model).lower()
    effort = reasoning_effort.lower() if isinstance(reasoning_effort, str) else ""
    return f"{model_name}:{effort}"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个 dict，返回新对象。"""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_unique_list(base: Any, override: Any) -> Any:
    """合并两个列表，保持顺序并去重。"""
    if not isinstance(base, list) or not isinstance(override, list):
        return override
    result: list[Any] = []
    seen: set[str] = set()
    for value in [*base, *override]:
        try:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _merge_responses_extra_body(
    body: dict[str, Any],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """合并 Responses API 的 extra_body，同时尽量不破坏 tools/include 等关键字段。"""
    reserved = {"include", "tools"}
    regular_extra = {key: value for key, value in extra_body.items() if key not in reserved}
    merged = _deep_merge(body, regular_extra)

    if "include" in extra_body:
        merged["include"] = _merge_unique_list(body.get("include"), extra_body["include"])

    if "tools" in extra_body:
        current_tools = body.get("tools")
        configured_tools = extra_body["tools"]
        if isinstance(current_tools, list) and isinstance(configured_tools, list):
            merged["tools"] = [*current_tools, *configured_tools]
        else:
            merged["tools"] = configured_tools

    return merged


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs.

    【中文名称】OpenAI 兼容 Provider（统一适配器）

    【功能说明】
    这是 nanobot 里覆盖面最广的 Provider 实现。几乎所有走 OpenAI chat completions
    格式的模型服务都走这一个类，包括：
    - OpenAI 官方（GPT-4o、GPT-5、o1/o3/o4 系列）
    - Azure OpenAI
    - 第三方网关：OpenRouter、DeepSeek、Zhipu/GLM、Moonshot、MiniMax 等
    - 本地模型服务：Ollama、vLLM、llama.cpp 等

    【核心策略】
    - chat completions 优先：绝大多数请求走 /v1/chat/completions
    - Responses API 按需：GPT-5 / o 系列推理模型走 /v1/responses（更丰富的推理信息）
    - 熔断保护：Responses API 连续失败 3 次后临时禁用 5 分钟
    - 厂商适配：通过 ProviderSpec 配置文件 + 模型名匹配自动注入各家差异参数
      （如 DeepSeek 的 reasoning_content 回填、Kimi 的 thinking_type、
        MiMo 的 enable_thinking 等）

    【关键内部状态】
    - _spec: ProviderSpec → 定义 provider 的行为特征（thinking_style、strip_model_prefix 等）
    - _extra_body: dict → 用户配置的额外请求体（可覆盖/扩展 thinking 参数）
    - _responses_failures/_responses_tripped_at → Responses API 熔断器状态
    - _is_local: bool → 是否本地端点（影响 keepalive 策略）

    Receives a resolved ``ProviderSpec`` from the caller — no internal
    registry lookups needed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
        extra_body: dict[str, Any] | None = None,
        api_type: str = "auto",
        extra_query: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec
        self._extra_body = extra_body or {}
        self._api_type = api_type if spec and spec.name == "openai" else "auto"
        self._extra_query = extra_query or {}

        if api_key and spec and spec.env_key:
            self._setup_env(api_key, api_base)

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        self._effective_base = effective_base
        self._default_headers = {"x-session-affinity": uuid.uuid4().hex}
        if _uses_openrouter_attribution(spec, effective_base):
            self._default_headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            self._default_headers.update(extra_headers)
        self._api_key_for_client = api_key or "no-key"
        self._is_local = _is_local_endpoint(spec, effective_base)

        # 懒初始化：OpenAI client 连同底层 httpx transport 创建成本不低，
        # 所以等第一次真正请求时再建。
        self._client: AsyncOpenAIType | None = None
        self._client_lock = asyncio.Lock()

        # Responses API 熔断器：
        # 如果某个模型/模式连续失败，就先临时停用 Responses API，
        # 一段时间后再尝试探测恢复。
        self._responses_failures: dict[str, int] = {}
        self._responses_tripped_at: dict[str, float] = {}

    def _build_client(self) -> None:
        """基于当前模块级 ``AsyncOpenAI`` 构建真实 client。"""
        import httpx

        timeout_s = _openai_compat_timeout_s()
        http_client: httpx.AsyncClient | None = None
        if self._is_local:
            # 本地模型服务（Ollama、llama.cpp、vLLM）常常比客户端更早关闭空闲连接。
            # 如果保留 keepalive，下一次请求可能恰好复用到“已经死掉的连接”，
            # 造成首次请求偶发失败。对本地/LAN 服务，关闭 keepalive 成本很低，
            # 但能显著降低这种伪随机连接错误。
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(keepalive_expiry=0),
                timeout=timeout_s,
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key_for_client,
            base_url=self._effective_base,
            default_headers=self._default_headers,
            default_query=self._extra_query or None,
            max_retries=0,
            timeout=timeout_s,
            http_client=http_client,
        )

    async def _ensure_client(self):
        """返回共享 OpenAI client；若不存在则在首次调用时创建。"""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            global AsyncOpenAI
            if AsyncOpenAI is None:
                if os.environ.get("LANGFUSE_SECRET_KEY") and importlib.util.find_spec("langfuse"):
                    from langfuse.openai import AsyncOpenAI as _AsyncOpenAI
                else:
                    if os.environ.get("LANGFUSE_SECRET_KEY"):
                        logger.warning(
                            "LANGFUSE_SECRET_KEY is set but langfuse is not installed; "
                            "install with `pip install langfuse` to enable tracing"
                        )
                    from openai import AsyncOpenAI as _AsyncOpenAI
                AsyncOpenAI = _AsyncOpenAI

            self._build_client()
            return self._client

    def _setup_env(self, api_key: str, api_base: str | None) -> None:
        """按 ProviderSpec 约定补齐环境变量。"""
        spec = self._spec
        if not spec or not spec.env_key:
            return
        if spec.is_gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key).replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    @classmethod
    def _apply_cache_control(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """为支持 prompt caching 的 provider 注入 ``cache_control`` 标记。"""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": cache_marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(content)
                nc[-1] = {**nc[-1], "cache_control": cache_marker}
                return {**msg, "content": nc}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """把工具调用 ID 规范化为 provider 更容易接受的短字母数字形式。"""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _should_normalize_tool_call_ids(self) -> bool:
        """判断当前 provider 是否需要规范化工具调用 ID。"""
        return bool(self._spec and self._spec.name == "mistral")

    @staticmethod
    def _coerce_content_to_string(content: Any) -> str | None:
        """把 block/list 形式内容尽量压平成纯文本，供只接受字符串的 API 使用。"""
        if content is None or isinstance(content, str):
            return content
        text = OpenAICompatProvider._extract_text_content(content)
        if isinstance(text, str) and text:
            return text
        try:
            dumped = json.dumps(content, ensure_ascii=False)
        except Exception:
            dumped = str(content)
        return dumped or "(empty)"

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """清洗消息：去掉非标准字段，并在需要时规范化 tool_call ID。

        【中文名称】消息清洗与规范化

        【功能说明】
        不同 OpenAI 兼容后台对消息格式的容忍度差异很大，此函数统一处理：

        1. 去除非标准字段：只保留 role/content/tool_calls/tool_call_id/name/
           reasoning_content/extra_content，其余字段丢弃
        2. 工具调用 ID 规范化：
           - Mistral 要求 9 字符 alphanumeric → SHA1 截断
           - 去重检测：避免并行 tool_calls 中有重复 ID
        3. 工具结果 ID 映射：确保 tool_result 的 tool_call_id 与其对应的
           tool_call 的 ID 一致（跨消息追踪 pending_tool_ids）
        4. old-style tool_calls：将工具调用中的 arguments 重新序列化为 JSON
        5. DeepSeek 兼容：强制 content 为纯文本（不支持 content block 数组）
        6. assistant 消息净化：tool_calls 存在时清除 content（部分网关拒绝混合）

        【参数说明】
        - messages: list[dict] → 原始消息列表

        【返回值】
        - list[dict] → 清洗后的消息列表，已通过 _enforce_role_alternation 校验
        """
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        id_map: dict[str, str] = {}
        pending_tool_ids: dict[str, deque[str]] = {}
        force_string_content = bool(self._spec and self._spec.name == "deepseek")
        normalize_tool_ids = self._should_normalize_tool_call_ids()

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            if not normalize_tool_ids:
                return value
            return id_map.setdefault(value, self._normalize_tool_call_id(value))

        def unique_tool_id(value: Any, used_ids: set[str], idx: int) -> str:
            if isinstance(value, str) and value:
                base = map_id(value)
            else:
                base = _short_tool_id()
            if not isinstance(base, str) or not base:
                base = _short_tool_id()
            if base not in used_ids:
                return base
            seed = value if isinstance(value, str) and value else base
            salt = 1
            while True:
                candidate = self._normalize_tool_call_id(f"{seed}:{idx}:{salt}")
                if isinstance(candidate, str) and candidate not in used_ids:
                    return candidate
                salt += 1

        def map_tool_result_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            queue = pending_tool_ids.get(value)
            if queue:
                mapped = queue.popleft()
                if not queue:
                    pending_tool_ids.pop(value, None)
                return mapped
            return map_id(value)

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized = []
                used_ids: set[str] = set()
                for idx, tc in enumerate(clean["tool_calls"]):
                    if not isinstance(tc, dict):
                        normalized.append(tc)
                        continue
                    tc_clean = dict(tc)
                    raw_id = tc_clean.get("id")
                    mapped_id = unique_tool_id(raw_id, used_ids, idx)
                    tc_clean["id"] = mapped_id
                    used_ids.add(mapped_id)
                    if isinstance(raw_id, str) and raw_id:
                        pending_tool_ids.setdefault(raw_id, deque()).append(mapped_id)
                    function = tc_clean.get("function")
                    if isinstance(function, dict):
                        function_clean = dict(function)
                        if "arguments" in function_clean:
                            function_clean["arguments"] = tool_arguments_json_for_replay(
                                function_clean.get("arguments")
                            )
                        else:
                            function_clean["arguments"] = "{}"
                        tc_clean["function"] = function_clean
                    normalized.append(tc_clean)
                clean["tool_calls"] = normalized
                if clean.get("role") == "assistant":
                    # Some OpenAI-compatible gateways reject assistant messages
                    # that mix non-empty content with tool_calls.
                    clean["content"] = None
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_tool_result_id(clean["tool_call_id"])
            if (
                force_string_content
                and not (clean.get("role") == "assistant" and clean.get("tool_calls"))
            ):
                clean["content"] = self._coerce_content_to_string(clean.get("content"))
        return self._enforce_role_alternation(sanitized)

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_temperature(
        model_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        """Return True when the model accepts a temperature parameter.

        GPT-5 family and reasoning models (o1/o3/o4) reject temperature
        when reasoning_effort is set to anything other than ``"none"``.
        """
        if reasoning_effort and reasoning_effort.lower() != "none":
            return False
        name = model_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """组装一次完整的 OpenAI chat.completions API 调用参数。

        【中文名称】构建 API 请求参数

        【功能说明】
        这是 chat() 和 chat_stream() 共享的参数组装逻辑，将所有变量收敛为一个
        统一的 kwargs dict。处理以下几个层面的差异：

        1. 模型名规范化：根据 ProviderSpec.strip_model_prefix 决定是否去掉前缀
        2. 消息清洗与重排：_sanitize_messages（去非标准字段 + tool_call ID 规范化
           + role 交替校验）
        3. token 参数：o1/o3/o4 和 GPT-5 必须用 max_completion_tokens
        4. temperature 兼容：GPT-5 / o 系列的 reasoning 模式下不支持 temperature
        5. 推理参数分发（key logic）：
           - thinking 风格映射：通过 _THINKING_STYLE_MAP 把 "thinking_type" /
             "enable_thinking" / "reasoning_split" 映射到各家厂商的具体 extra_body 形状
           - 网关推理映射：DashScope / 其他网关的特殊 reasoning_effort →
             reasoning.effort 转换
           - Kimi 特殊处理：去除冗余的 reasoning_effort 字段（与 thinking_type 冲突）
        6. DeepSeek 推理回填：reasioner 模式下自动补 reasoning_content="" 给每条
           assistant 消息（否则 API 返回 400）
        7. prompt caching：对 Anthropic 系模型注入 cache_control marker
        8. extra_body 合并：用户配置的 extra_body 以深度合并方式覆盖，不会丢失
           系统已设置的 thinking 参数

        【参数说明】
        - messages: 内部消息列表
        - tools: 工具定义列表
        - model: 模型名
        - max_tokens: 最大输出 token
        - temperature: 采样温度
        - reasoning_effort: 推理力度
        - tool_choice: 工具选择策略

        【返回值】
        - dict → 可直接解包传给 client.chat.completions.create(**kwargs) 的参数字典
        """
        model_name = model or self.default_model
        spec = self._spec

        if spec and spec.supports_prompt_caching:
            model_name = model or self.default_model
            if any(model_name.lower().startswith(k) for k in ("anthropic/", "claude")):
                messages, tools = self._apply_cache_control(messages, tools)

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
        }

        # GPT-5 and reasoning models (o1/o3/o4) reject temperature when
        # reasoning_effort is active.  Only include it when safe.
        if self._supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature

        if (
            spec and getattr(spec, "supports_max_completion_tokens", False)
        ) or _requires_max_completion_tokens(model_name):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)

        if spec:
            model_lower = model_name.lower()
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    break

        # Normalize reasoning_effort into a semantic form (OpenAI vocab)
        # used for internal decisions, and a wire form actually sent out.
        # "minimum" is accepted as a DashScope-native alias for "minimal".
        semantic_effort: str | None = None
        if isinstance(reasoning_effort, str):
            semantic_effort = reasoning_effort.lower()
            if semantic_effort == "minimum":
                semantic_effort = "minimal"

        wire_effort = reasoning_effort
        if spec and spec.name == "dashscope" and semantic_effort == "minimal":
            # DashScope accepts none/minimum/low/medium/high/xhigh; "minimal" 400s.
            wire_effort = "minimum"

        if wire_effort and semantic_effort != "none":
            kwargs["reasoning_effort"] = wire_effort

        # Only send thinking controls when reasoning_effort is explicit so
        # omitting the config preserves each provider's default.
        if reasoning_effort is not None:
            thinking_enabled = semantic_effort not in ("none", "minimal")
            for thinking_style in _thinking_styles_for(spec, model_name):
                extra = _thinking_extra_body(thinking_style, thinking_enabled)
                if extra:
                    kwargs.setdefault("extra_body", {}).update(extra)
            gateway_style = getattr(spec, "gateway_reasoning_style", "") if spec else ""
            if gateway_style and _model_thinking_style(model_name):
                extra = _gateway_reasoning_extra_body(gateway_style, semantic_effort)
                if extra:
                    kwargs.setdefault("extra_body", {}).update(extra)

            # Moonshot rejects requests that carry both 'reasoning_effort'
            # and the native 'thinking' param.  We already expressed the
            # user's intent via the provider-native shape, so drop the
            # redundant wire-level kwarg.  Only kimi models need this —
            # Xiaomi's API accepts both params.
            if _model_slug(model_name) in _KIMI_THINKING_MODELS:
                kwargs.pop("reasoning_effort", None)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        # Backfill reasoning_content="" on assistants missing it: DeepSeek
        # thinking mode rejects history otherwise (#3554, #3584); "" reads
        # as "no thinking that turn". DeepSeek-V4/reasoner reason natively,
        # so backfill even without explicit reasoning_effort.
        explicit_thinking = (
            reasoning_effort is not None
            and semantic_effort not in ("none", "minimal")
            and (
                (spec and spec.thinking_style)
                or _model_thinking_style(model_name)
            )
        )
        implicit_deepseek_thinking = (
            spec is not None
            and spec.name == "deepseek"
            and semantic_effort not in ("none", "minimal", "minimum")
            and any(t in model_name.lower() for t in ("deepseek-v4", "deepseek-reasoner"))
        )
        if explicit_thinking or implicit_deepseek_thinking:
            for msg in kwargs["messages"]:
                if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""

        # Merge user-configured extra_body last so it can override or
        # extend provider-specific defaults (e.g. chat_template_kwargs,
        # guided_json, repetition_penalty).  Uses recursive merge so
        # nested dicts like {"chat_template_kwargs": {"enable_thinking": false}}
        # do not clobber sibling keys already set by thinking-style logic.
        if self._extra_body:
            existing = kwargs.get("extra_body", {})
            kwargs["extra_body"] = _deep_merge(existing, self._extra_body)

        return kwargs

    def _should_use_responses_api(
        self,
        model: str | None,
        reasoning_effort: str | None,
    ) -> bool:
        """Use Responses API only for direct OpenAI requests that benefit from it."""
        if self._api_type == "chat_completions":
            return False
        if self._spec and self._spec.name not in ("openai", "github_copilot"):
            return False
        if self._api_type == "responses":
            # Explicit configuration means Responses is mandatory; do not
            # consult the circuit breaker or fall back to Chat Completions.
            return True
        if self._spec is None or self._spec.name != "github_copilot":
            if not _is_direct_openai_base(self._effective_base):
                return False

        model_name = (model or self.default_model).lower()
        wants = False
        if reasoning_effort and reasoning_effort.lower() != "none":
            wants = True
        elif any(token in model_name for token in ("gpt-5", "o1", "o3", "o4")):
            wants = True
        if not wants:
            return False

        return self._responses_circuit_allows_probe(model, reasoning_effort)

    def _responses_circuit_allows_probe(
        self,
        model: str | None,
        reasoning_effort: str | None,
    ) -> bool:
        """Return False when the Responses API circuit breaker is open."""
        key = _responses_circuit_key(model, self.default_model, reasoning_effort)
        failures = self._responses_failures.get(key, 0)
        if failures >= _RESPONSES_FAILURE_THRESHOLD:
            tripped = self._responses_tripped_at.get(key, 0.0)
            if (time.monotonic() - tripped) < _RESPONSES_PROBE_INTERVAL_S:
                return False
            # Half-open: allow one probe attempt
        return True

    def _record_responses_failure(self, model: str | None, reasoning_effort: str | None) -> None:
        key = _responses_circuit_key(model, self.default_model, reasoning_effort)
        count = self._responses_failures.get(key, 0) + 1
        self._responses_failures[key] = count
        if count >= _RESPONSES_FAILURE_THRESHOLD:
            self._responses_tripped_at[key] = time.monotonic()
            logger.warning(
                "Responses API circuit open for {} — falling back to Chat Completions",
                key,
            )

    def _record_responses_success(self, model: str | None, reasoning_effort: str | None) -> None:
        key = _responses_circuit_key(model, self.default_model, reasoning_effort)
        self._responses_failures.pop(key, None)
        self._responses_tripped_at.pop(key, None)

    @staticmethod
    def _should_fallback_from_responses_error(e: Exception) -> bool:
        """Fallback only for likely Responses API compatibility errors."""
        response = getattr(e, "response", None)
        status_code = getattr(e, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code not in {400, 404, 422}:
            return False

        body = (
            getattr(e, "body", None)
            or getattr(e, "doc", None)
            or getattr(response, "text", None)
        )
        body_text = str(body).lower() if body is not None else ""
        compatibility_markers = (
            "responses",
            "response api",
            "max_output_tokens",
            "instructions",
            "previous_response",
            "unsupported",
            "not supported",
            "unknown parameter",
            "unrecognized request argument",
        )
        return any(marker in body_text for marker in compatibility_markers)

    def _build_responses_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a Responses API body for direct OpenAI requests."""
        model_name = model or self.default_model
        if self._spec and self._spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]
        sanitized_messages = self._sanitize_messages(self._sanitize_empty_content(messages))
        instructions, input_items = convert_messages(sanitized_messages)

        body: dict[str, Any] = {
            "model": model_name,
            "instructions": instructions or None,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
            "store": False,
            "stream": False,
        }

        if self._supports_temperature(model_name, reasoning_effort):
            body["temperature"] = temperature

        if reasoning_effort and reasoning_effort.lower() != "none":
            body["reasoning"] = {"effort": reasoning_effort}
            body["include"] = ["reasoning.encrypted_content"]

        if tools:
            body["tools"] = convert_tools(tools)
            body["tool_choice"] = tool_choice or "auto"

        extra_body = getattr(self, "_extra_body", {})
        if extra_body:
            body = _merge_responses_extra_body(body, extra_body)

        return body

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @classmethod
    def _extract_text_content(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                item_map = cls._maybe_mapping(item)
                if item_map:
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, int]:
        """Extract token usage from an OpenAI-compatible response.

        Handles both dict-based (raw JSON) and object-based (SDK Pydantic)
        responses.  Provider-specific ``cached_tokens`` fields are normalised
        under a single key; see the priority chain inside for details.
        """
        # --- resolve usage object ---
        usage_obj = None
        response_map = cls._maybe_mapping(response)
        if response_map is not None:
            usage_obj = response_map.get("usage")
        elif hasattr(response, "usage") and response.usage:
            usage_obj = response.usage

        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            result = {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }
        elif usage_obj:
            result = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        else:
            return {}

        # --- cached_tokens (normalised across providers) ---
        # Try nested paths first (dict), fall back to attribute (SDK object).
        # Priority order ensures the most specific field wins.
        for path in (
            ("prompt_tokens_details", "cached_tokens"),  # OpenAI/Zhipu/MiniMax/Qwen/Mistral/xAI
            ("cached_tokens",),                          # StepFun/Moonshot (top-level)
            ("prompt_cache_hit_tokens",),                # DeepSeek/SiliconFlow
        ):
            cached = cls._get_nested_int(usage_map, path)
            if not cached and usage_obj:
                cached = cls._get_nested_int(usage_obj, path)
            if cached:
                result["cached_tokens"] = cached
                break

        return result

    @staticmethod
    def _get_nested_int(obj: Any, path: tuple[str, ...]) -> int:
        """Drill into *obj* by *path* segments and return an ``int`` value.

        Supports both dict-key access and attribute access so it works
        uniformly with raw JSON dicts **and** SDK Pydantic models.
        """
        current = obj
        for segment in path:
            if current is None:
                return 0
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                current = getattr(current, segment, None)
        return int(current or 0) if current is not None else 0

    def _parse(self, response: Any) -> LLMResponse:
        """把 OpenAI API 的原始响应解析成统一的 LLMResponse。

        【中文名称】响应解析

        【功能说明】
        同时兼容 dict 格式（raw JSON）和 Pydantic 对象格式（SDK 自动解析）的响应：

        【处理流程 —— 3 条路径】
        Path A —— response 是纯字符串：直接返回 LLMResponse(content=string)

        Path B —— response 是 dict（raw JSON）：
          1. 从 choices[0].message 提取 content
          2. 遍历所有 choice 的 tool_calls（支持 parallel tool calls）
          3. 解析每个 tool_call：id + function.name + function.arguments
          4. 提取扩展字段（extra_content/provider_specific_fields）
          5. 提取 reasoning_content（reasoning 字段作为备选，支持 StepFun）

        Path C —— response 是 Pydantic 对象（SDK 模式）：
          同 Path B 但通过属性访问而非字典 key

        【参数说明】
        - response: Any → OpenAI SDK 返回的原始响应对象

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体
        """
        if isinstance(response, str):
            return LLMResponse(content=response, finish_reason="stop")

        response_map = self._maybe_mapping(response)
        if response_map is not None:
            choices = response_map.get("choices") or []
            if not choices:
                content = self._extract_text_content(
                    response_map.get("content") or response_map.get("output_text")
                )
                reasoning_content = self._extract_text_content(
                    response_map.get("reasoning_content")
                )
                if content is not None:
                    return LLMResponse(
                        content=content,
                        reasoning_content=reasoning_content,
                        finish_reason=str(response_map.get("finish_reason") or "stop"),
                        usage=self._extract_usage(response_map),
                    )
                return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

            choice0 = self._maybe_mapping(choices[0]) or {}
            msg0 = self._maybe_mapping(choice0.get("message")) or {}
            content = self._extract_text_content(msg0.get("content"))
            finish_reason = str(choice0.get("finish_reason") or "stop")

            raw_tool_calls: list[Any] = []
            # StepFun: fallback to reasoning field when content is empty
            if not content and msg0.get("reasoning") and self._spec and self._spec.reasoning_as_content:
                content = self._extract_text_content(msg0.get("reasoning"))
            reasoning_content = msg0.get("reasoning_content")
            if reasoning_content is None and msg0.get("reasoning"):
                reasoning_content = self._extract_text_content(msg0.get("reasoning"))
            for ch in choices:
                ch_map = self._maybe_mapping(ch) or {}
                m = self._maybe_mapping(ch_map.get("message")) or {}
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    raw_tool_calls.extend(tool_calls)
                    if ch_map.get("finish_reason") in ("tool_calls", "stop"):
                        finish_reason = str(ch_map["finish_reason"])
                if not content:
                    content = self._extract_text_content(m.get("content"))
                if reasoning_content is None:
                    reasoning_content = m.get("reasoning_content")

            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_map = self._maybe_mapping(tc) or {}
                fn = self._maybe_mapping(tc_map.get("function")) or {}
                args = parse_tool_arguments(fn.get("arguments", {}))
                ec, prov, fn_prov = _extract_tc_extras(tc)
                parsed_tool_calls.append(ToolCallRequest(
                    id=str(tc_map.get("id") or _short_tool_id()),
                    name=str(fn.get("name") or ""),
                    arguments=args,
                    extra_content=ec,
                    provider_specific_fields=prov,
                    function_provider_specific_fields=fn_prov,
                ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response_map),
                reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
            )

        if not response.choices:
            return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        finish_reason = choice.finish_reason

        raw_tool_calls: list[Any] = []
        for ch in response.choices:
            m = ch.message
            if hasattr(m, "tool_calls") and m.tool_calls:
                raw_tool_calls.extend(m.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and m.content:
                content = m.content
            if not content and getattr(m, "reasoning", None) and self._spec and self._spec.reasoning_as_content:
                content = m.reasoning

        tool_calls = []
        for tc in raw_tool_calls:
            args = parse_tool_arguments(tc.function.arguments)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            tool_calls.append(ToolCallRequest(
                id=str(getattr(tc, "id", None) or _short_tool_id()),
                name=tc.function.name,
                arguments=args,
                extra_content=ec,
                provider_specific_fields=prov,
                function_provider_specific_fields=fn_prov,
            ))

        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None and getattr(msg, "reasoning", None):
            reasoning_content = msg.reasoning

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=self._extract_usage(response),
            reasoning_content=reasoning_content,
        )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        """把流式响应的 chunk 列表合并解析成统一的 LLMResponse。

        【中文名称】流式响应合并解析

        【功能说明】
        流式请求先将所有 SSE chunk 收集到列表中，流结束后调用此函数一次性解析。
        之所以不走 OpenAI SDK 的 stream 自动合并，是因为：
        - 某些厂商的 SDK 合并行为不一致
        - 需要在合并过程中做 ID 去重（如 Zhipu/GLM 的 parallel tool calls 复用 ID）
        - 需要无损保留 extra_content 等扩展字段

        【核心逻辑】
        1. 遍历所有 chunk，按 index 分组累积 tool_call 片段
        2. 同时累积 content + reasoning_content 文本
        3. 检测 ID 重复并自动生成新 ID
        4. 兼容旧式 function_call 格式

        【参数说明】
        - chunks: list[Any] → 流式响应收集的所有 chunk

        【返回值】
        - LLMResponse → 合并后的完整响应
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_bufs: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        def _accum_tc(tc: Any, idx_hint: int) -> None:
            """Accumulate one streaming tool-call delta into *tc_bufs*."""
            tc_index: int = _get(tc, "index") if _get(tc, "index") is not None else idx_hint
            buf = tc_bufs.setdefault(tc_index, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            tc_id = _get(tc, "id")
            if tc_id:
                buf["id"] = str(tc_id)
            fn = _get(tc, "function")
            if fn is not None:
                fn_name = _get(fn, "name")
                if fn_name:
                    buf["name"] = str(fn_name)
                fn_args = _get(fn, "arguments")
                if fn_args:
                    buf["arguments"] += str(fn_args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            if ec:
                buf["extra_content"] = ec
            if prov:
                buf["prov"] = prov
            if fn_prov:
                buf["fn_prov"] = fn_prov

        def _accum_legacy_function_call(function_call: Any) -> None:
            """Accumulate legacy ``delta.function_call`` streaming chunks."""
            if not function_call:
                return
            buf = tc_bufs.setdefault(0, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            fn_name = _get(function_call, "name")
            if fn_name:
                buf["name"] = str(fn_name)
            fn_args = _get(function_call, "arguments")
            if fn_args:
                buf["arguments"] += str(fn_args)

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = chunk_map.get("choices") or []
                if not choices:
                    usage = cls._extract_usage(chunk_map) or usage
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                text = cls._extract_text_content(delta.get("content"))
                if text:
                    content_parts.append(text)
                text = cls._extract_text_content(delta.get("reasoning_content"))
                if not text:
                    text = cls._extract_text_content(delta.get("reasoning"))
                if text:
                    reasoning_parts.append(text)
                for idx, tc in enumerate(delta.get("tool_calls") or []):
                    _accum_tc(tc, idx)
                _accum_legacy_function_call(delta.get("function_call"))
                usage = cls._extract_usage(chunk_map) or usage
                continue

            if not chunk.choices:
                usage = cls._extract_usage(chunk) or usage
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
            if delta:
                reasoning = getattr(delta, "reasoning_content", None)
                if not reasoning:
                    reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
            for tc in (getattr(delta, "tool_calls", None) or []) if delta else []:
                _accum_tc(tc, getattr(tc, "index", 0))
            if delta:
                _accum_legacy_function_call(getattr(delta, "function_call", None))

        # Some providers (e.g. Zhipu/GLM) reuse the same tool_call id for
        # parallel tool calls in streaming mode. Deduplicate before building
        # the response so downstream tool messages don't collide.
        _seen_tc_ids: set[str] = set()
        for b in tc_bufs.values():
            if not b["id"] or b["id"] in _seen_tc_ids:
                b["id"] = _short_tool_id()
            _seen_tc_ids.add(b["id"])

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                ToolCallRequest(
                    id=b["id"] or _short_tool_id(),
                    name=b["name"],
                    arguments=parse_tool_arguments(b["arguments"]),
                    extra_content=b.get("extra_content"),
                    provider_specific_fields=b.get("prov"),
                    function_provider_specific_fields=b.get("fn_prov"),
                )
                for b in tc_bufs.values()
            ],
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
        )

    @classmethod
    def _extract_error_metadata(cls, e: Exception) -> dict[str, Any]:
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
        error_type, error_code = LLMProvider._extract_error_type_code(payload)

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

        return {
            "error_status_code": int(status_code) if status_code is not None else None,
            "error_kind": error_kind,
            "error_type": error_type,
            "error_code": error_code,
            "error_retry_after_s": cls._extract_retry_after_from_headers(headers),
            "error_should_retry": should_retry,
        }

    @staticmethod
    def _handle_error(
        e: Exception,
        *,
        spec: ProviderSpec | None = None,
        api_base: str | None = None,
    ) -> LLMResponse:
        body = (
            getattr(e, "doc", None)
            or getattr(e, "body", None)
            or getattr(getattr(e, "response", None), "text", None)
        )
        body_text = body if isinstance(body, str) else str(body) if body is not None else ""
        msg = f"Error: {body_text.strip()[:500]}" if body_text.strip() else f"Error calling LLM: {e}"

        text = f"{body_text} {e}".lower()
        if spec and spec.is_local and ("502" in text or "connection" in text or "refused" in text):
            msg += (
                "\nHint: this is a local model endpoint. Check that the local server is reachable at "
                f"{api_base or spec.default_api_base}, and if you are using a proxy/tunnel, make sure it "
                "can reach your local Ollama/vLLM service instead of routing localhost through the remote host."
            )

        response = getattr(e, "response", None)
        retry_after = LLMProvider._extract_retry_after_from_headers(getattr(response, "headers", None))
        if retry_after is None:
            retry_after = LLMProvider._extract_retry_after(msg)
        return LLMResponse(
            content=msg,
            finish_reason="error",
            retry_after=retry_after,
            **OpenAICompatProvider._extract_error_metadata(e),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """发送非流式对话请求。

        【中文名称】非流式对话

        【功能说明】
        OpenAI Compat Provider 的同步调用入口。一次完整调用流程：

        1. _ensure_client()：懒初始化 OpenAI SDK client（首次调用时才创建）
        2. _should_use_responses_api()：判断是否走 Responses API——
           - 仅 OpenAI 官方 / GitHub Copilot 且模型是 GPT-5 / o 系列时使用
           - 受熔断器保护（连续失败 3 次后禁用 5 分钟）
        3. Responses API 路径：
           - _build_responses_body 组装参数
           - parse_response_output 解析结果
           - 失败时：400/404/422 且错误消息含兼容性标记 → fallback 到 chat.completions
           - 其他错误 → 记录失败次数，触发熔断器
        4. Chat Completions 路径：
           - _build_kwargs 组装参数（含消息清洗 + 厂商适配）
           - 调用 client.chat.completions.create
           - _parse 解析响应
        5. 异常统一走 _handle_error 转换为 LLMResponse

        【参数说明】
        - messages: 消息历史列表
        - tools: 工具定义列表（可选）
        - model: 模型名（可选，默认使用 default_model）
        - max_tokens: 最大输出 token（默认 4096）
        - temperature: 采样温度（默认 0.7）
        - reasoning_effort: 推理力度
        - tool_choice: 工具选择策略

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体
        """
        await self._ensure_client()
        try:
            if self._should_use_responses_api(model, reasoning_effort):
                try:
                    body = self._build_responses_body(
                        messages, tools, model, max_tokens, temperature,
                        reasoning_effort, tool_choice,
                    )
                    result = parse_response_output(await self._client.responses.create(**body))
                    self._record_responses_success(model, reasoning_effort)
                    return result
                except Exception as responses_error:
                    if self._spec and self._spec.name == "github_copilot":
                        # Copilot gateway exposes GPT-5/o-series only via /responses;
                        # falling back to /chat/completions cannot succeed and would
                        # hide the real error.
                        raise
                    if self._api_type == "responses":
                        raise
                    if not self._should_fallback_from_responses_error(responses_error):
                        raise
                    self._record_responses_failure(model, reasoning_effort)

            kwargs = self._build_kwargs(
                messages, tools, model, max_tokens, temperature,
                reasoning_effort, tool_choice,
            )
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return self._handle_error(e, spec=self._spec, api_base=self.api_base)

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
        """发送流式对话请求。

        【中文名称】流式对话

        【功能说明】
        OpenAI Compat Provider 的流式调用入口。与 chat() 不同之处：

        【流式处理流程】
        Phase 1 —— 客户端就位：_ensure_client() 懒初始化 SDK client

        Phase 2 —— Responses API 流式路径（GPT-5 / o 系列）：
          1. _build_responses_body(stream=True)
          2. consume_sdk_stream 消费 SSE 流，期间实时回调 on_content_delta + on_tool_call_delta
          3. 失败策略同 chat()
          4. 直接由 consume_sdk_stream 返回合并好的 content + tool_calls

        Phase 3 —— Chat Completions 流式路径（大多数情况）：
          1. _build_kwargs(stream=True, stream_options=include_usage)
          2. 利用 asyncio.wait_for + idle_timeout 做流空闲保护（默认 90s）
          3. 逐 chunk 遍历：
             - delta.content → on_content_delta 回调（用于打字机效果）
             - delta.reasoning → on_thinking_delta 回调
             - delta.tool_calls → on_tool_call_delta 回调（用于实时文件编辑预览）
          4. 所有 chunk 累积到列表中
          5. 流结束后 _parse_chunks 合并解析（含 ID 去重）

        Phase 4 —— 异常处理：
          - asyncio.TimeoutError → 流空闲超时，返回错误 LLMResponse
          - 其他异常 → _handle_error 统一转换

        【参数说明】
        - messages: 消息历史列表
        - tools: 工具定义列表（可选）
        - model: 模型名
        - max_tokens: 最大输出 token
        - temperature: 采样温度
        - reasoning_effort: 推理力度
        - tool_choice: 工具选择策略
        - on_content_delta: 文本增量回调
        - on_thinking_delta: 思考增量回调
        - on_tool_call_delta: 工具调用增量回调

        【返回值】
        - LLMResponse → 统一格式的 LLM 响应结构体
        """
        await self._ensure_client()
        idle_timeout_s = int(os.environ.get("NANOBOT_STREAM_IDLE_TIMEOUT_S", "90"))
        try:
            if self._should_use_responses_api(model, reasoning_effort):
                try:
                    body = self._build_responses_body(
                        messages, tools, model, max_tokens, temperature,
                        reasoning_effort, tool_choice,
                    )
                    body["stream"] = True
                    stream = await self._client.responses.create(**body)

                    async def _timed_stream():
                        stream_iter = stream.__aiter__()
                        while True:
                            try:
                                yield await asyncio.wait_for(
                                    stream_iter.__anext__(),
                                    timeout=idle_timeout_s,
                                )
                            except StopAsyncIteration:
                                break

                    (
                        content,
                        tool_calls,
                        finish_reason,
                        usage,
                        reasoning_content,
                    ) = await consume_sdk_stream(
                        _timed_stream(),
                        on_content_delta,
                        on_tool_call_delta=on_tool_call_delta,
                    )
                    self._record_responses_success(model, reasoning_effort)
                    return LLMResponse(
                        content=content or None,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                        reasoning_content=reasoning_content,
                    )
                except Exception as responses_error:
                    if self._spec and self._spec.name == "github_copilot":
                        # Copilot gateway exposes GPT-5/o-series only via /responses;
                        # falling back to /chat/completions cannot succeed and would
                        # hide the real error.
                        raise
                    if self._api_type == "responses":
                        raise
                    if not self._should_fallback_from_responses_error(responses_error):
                        raise
                    self._record_responses_failure(model, reasoning_effort)

            kwargs = self._build_kwargs(
                messages, tools, model, max_tokens, temperature,
                reasoning_effort, tool_choice,
            )
            if self._spec and self._spec.name == "zhipu" and tools and on_tool_call_delta:
                # Z.AI/GLM keeps streaming tool-call arguments behind an
                # explicit provider flag.  Pass it through the OpenAI SDK's
                # extra_body escape hatch so the usual delta.tool_calls path
                # can surface live file-edit progress.
                kwargs.setdefault("extra_body", {})["tool_stream"] = True
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            stream = await self._client.chat.completions.create(**kwargs)
            chunks: list[Any] = []
            stream_iter = stream.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=idle_timeout_s,
                    )
                except StopAsyncIteration:
                    break
                chunks.append(chunk)
                if chunk.choices:
                    delta_obj = chunk.choices[0].delta
                    if on_content_delta:
                        text = getattr(delta_obj, "content", None)
                        if text:
                            await on_content_delta(text)
                    if on_thinking_delta:
                        reasoning = getattr(delta_obj, "reasoning_content", None) or getattr(
                            delta_obj, "reasoning", None,
                        )
                        r_text = self._extract_text_content(reasoning)
                        if r_text:
                            await on_thinking_delta(r_text)
                    if on_tool_call_delta:
                        for idx, tool_delta in enumerate(
                            getattr(delta_obj, "tool_calls", None) or []
                        ):
                            fn = _get(tool_delta, "function")
                            tool_index = _get(tool_delta, "index")
                            await on_tool_call_delta({
                                "index": tool_index if tool_index is not None else idx,
                                "call_id": str(_get(tool_delta, "id") or ""),
                                "name": str(_get(fn, "name") or "") if fn is not None else "",
                                "arguments_delta": (
                                    str(_get(fn, "arguments") or "") if fn is not None else ""
                                ),
                            })
                        function_call = getattr(delta_obj, "function_call", None)
                        if function_call:
                            await on_tool_call_delta({
                                "index": 0,
                                "call_id": "",
                                "name": str(_get(function_call, "name") or ""),
                                "arguments_delta": str(_get(function_call, "arguments") or ""),
                            })
            return self._parse_chunks(chunks)
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
            return self._handle_error(e, spec=self._spec, api_base=self.api_base)

    def get_default_model(self) -> str:
        return self.default_model
