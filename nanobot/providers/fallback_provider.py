"""Provider 包装器：主模型失败时自动切换到备用模型。

这个文件解决的是“模型可用性”问题：
- 主模型偶发超时、限流、连接失败时，不让整次请求直接报错
- 自动尝试备用模型，尽量维持服务连续性
- 对持续故障的主模型做简单熔断，避免每次都白白浪费一次请求
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse

# 熔断参数：与 OpenAICompatProvider 的 Responses API 熔断策略保持大致一致。
_PRIMARY_FAILURE_THRESHOLD = 3
_PRIMARY_COOLDOWN_S = 60
_MISSING = object()
_FALLBACK_ERROR_KINDS = frozenset({
    "timeout",
    "connection",
    "server_error",
    "rate_limit",
    "overloaded",
})
_NON_FALLBACK_ERROR_KINDS = frozenset({
    "authentication",
    "auth",
    "permission",
    "content_filter",
    "refusal",
    "context_length",
    "invalid_request",
})
_FALLBACK_ERROR_TOKENS = (
    "rate_limit",
    "rate limit",
    "too_many_requests",
    "too many requests",
    "overloaded",
    "server_error",
    "server error",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection",
    "insufficient_quota",
    "insufficient quota",
    "quota_exceeded",
    "quota exceeded",
    "quota_exhausted",
    "quota exhausted",
    "billing_hard_limit",
    "insufficient_balance",
    "balance",
    "out of credits",
)


class FallbackProvider(LLMProvider):
    """带自动降级能力的 Provider 包装器。

    【中文名称】主备模型切换 Provider

    【核心思想】

    先使用主 provider；如果错误类型属于“临时不可用”而不是“请求本身有问题”，
    就按顺序尝试 fallback 模型。

    【设计重点】

    - failover 是单次请求级的，不跨 turn 保存业务状态
    - 若正文已经流式输出，通常不再切换，避免重复内容
    - “流中途超时”是一个特例：允许新开一个流段继续输出
    - 主模型连续失败达到阈值后会短暂熔断
    """

    supports_stream_recover_callback = True

    def __init__(
        self,
        primary: LLMProvider,
        fallback_presets: list[Any],
        provider_factory: Callable[[Any], LLMProvider],
    ):
        """init。
        
        初始化 FallbackProvider 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._primary = primary
        self._fallback_presets = list(fallback_presets)
        self._provider_factory = provider_factory
        self._has_fallbacks = bool(fallback_presets)
        self._primary_failures = 0
        self._primary_tripped_at: float | None = None

    @property
    def generation(self):
        """generation。
        
        读写底层 provider 的生成参数对象，使 fallback 包装器和真实 provider 保持同一套配置。
        实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._primary.generation

    @generation.setter
    def generation(self, value):
        """generation。
        
        读写底层 provider 的生成参数对象，使 fallback 包装器和真实 provider 保持同一套配置。
        实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        self._primary.generation = value

    def get_default_model(self) -> str:
        """get default model。
        
        实现方法：优先从显式参数或实例状态读取目标值，缺失时回退到默认配置，并把结果整理成调用方期望的类型。"""
        return self._primary.get_default_model()

    @property
    def supports_progress_deltas(self) -> bool:
        """supports progress deltas。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        return bool(getattr(self._primary, "supports_progress_deltas", False))

    def _primary_available(self) -> bool:
        """判断主 provider 当前是否允许再尝试。

        若已进入熔断状态，只有冷却时间过去后才允许进行一次“半开探测”。

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        if self._primary_tripped_at is None:
            return True
        if time.monotonic() - self._primary_tripped_at >= _PRIMARY_COOLDOWN_S:
            # 半开状态：允许再探测一次，看看主模型是否已经恢复。
            return True
        return False

    async def chat(self, **kwargs: Any) -> LLMResponse:
        """chat。
        
        实现方法：把统一请求参数转换为当前 provider 的 API 调用，并把返回值解析成 LLMResponse。"""
        if not self._has_fallbacks:
            return await self._primary.chat(**kwargs)
        return await self._try_with_fallback(
            lambda p, kw: p.chat(**kw), kwargs, has_streamed=None
        )

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        """chat stream。
        
        实现方法：把普通聊天参数转换为流式请求，逐块消费增量文本、思考内容和工具调用，最后汇总成统一响应。"""
        on_stream_recover = kwargs.pop("on_stream_recover", None)
        if not self._has_fallbacks:
            return await self._primary.chat_stream(**kwargs)

        has_streamed: list[bool] = [False]
        original_delta = kwargs.get("on_content_delta")

        async def _tracking_delta(text: str) -> None:
            """tracking delta。
            
            实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
            if text:
                has_streamed[0] = True
            if original_delta:
                await original_delta(text)

        kwargs["on_content_delta"] = _tracking_delta
        return await self._try_with_fallback(
            lambda p, kw: p.chat_stream(**kw),
            kwargs,
            has_streamed=has_streamed,
            on_stream_recover=on_stream_recover,
        )

    async def _try_with_fallback(
        self,
        call: Callable[[LLMProvider, dict[str, Any]], Awaitable[LLMResponse]],
        kwargs: dict[str, Any],
        has_streamed: list[bool] | None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """先试主模型，必要时按顺序切换到 fallback 模型。
        
        实现方法：在主 provider 失败或返回可降级错误时，按预设顺序切换备用 provider 重试。"""
        primary_model = kwargs.get("model") or self._primary.get_default_model()

        if self._primary_available():
            response = await call(self._primary, kwargs)
            if response.finish_reason != "error":
                self._primary_failures = 0
                self._primary_tripped_at = None
                return response

            if has_streamed is not None and has_streamed[0]:
                # 流式输出已经把正文发给用户后，普通失败不应再切换，否则会重复回答。
                # 只有“中途超时”这种情况，才允许结束当前流段后再继续。
                is_timeout = (response.error_kind or "").lower() == "timeout"
                if is_timeout:
                    logger.warning(
                        "Primary model '{}' stream stalled after content was emitted; "
                        "attempting failover anyway",
                        primary_model,
                    )
                    has_streamed[0] = False
                    if on_stream_recover:
                        await on_stream_recover()
                    else:
                        kwargs["on_content_delta"] = None
                else:
                    logger.warning(
                        "Primary model error but content already streamed; skipping failover"
                    )
                    return response

            if not self._should_fallback(response):
                logger.warning(
                    "Primary model '{}' returned non-fallbackable error: {}",
                    primary_model,
                    (response.content or "")[:120],
                )
                return response

            self._primary_failures += 1
            if self._primary_failures >= _PRIMARY_FAILURE_THRESHOLD:
                self._primary_tripped_at = time.monotonic()
                logger.warning(
                    "Primary model '{}' circuit open after {} consecutive failures",
                    primary_model, self._primary_failures,
                )
        else:
            logger.debug("Primary model '{}' circuit open; skipping", primary_model)

        last_response: LLMResponse | None = None
        primary_skipped = not self._primary_available()
        for idx, fallback in enumerate(self._fallback_presets):
            fallback_model = fallback.model
            if has_streamed is not None and has_streamed[0]:
                is_timeout = (
                    last_response is not None
                    and (last_response.error_kind or "").lower() == "timeout"
                )
                if is_timeout and on_stream_recover:
                    logger.warning(
                        "Fallback model '{}' stream stalled after content was emitted; "
                        "starting a new stream segment and trying next fallback",
                        self._fallback_presets[idx - 1].model if idx > 0 else primary_model,
                    )
                    has_streamed[0] = False
                    await on_stream_recover()
                else:
                    break
            if idx == 0 and primary_skipped:
                logger.info(
                    "Primary model '{}' circuit open, trying fallback '{}'",
                    primary_model, fallback_model,
                )
            elif idx == 0:
                logger.info(
                    "Primary model '{}' failed, trying fallback '{}'",
                    primary_model, fallback_model,
                )
            else:
                logger.info(
                    "Fallback '{}' also failed, trying next fallback '{}'",
                    self._fallback_presets[idx - 1].model, fallback_model,
                )
            try:
                fallback_provider = self._provider_factory(fallback)
            except Exception as exc:
                logger.warning(
                    "Failed to create provider for fallback '{}': {}", fallback_model, exc
                )
                continue

            # 临时把模型名、max_tokens、temperature 等参数切换为 fallback 的值，
            # 调用结束后再恢复，避免污染调用方原始 kwargs。
            original_values = {
                name: kwargs.get(name, _MISSING)
                for name in ("model", "max_tokens", "temperature", "reasoning_effort")
            }
            kwargs["model"] = fallback_model
            kwargs["max_tokens"] = fallback.max_tokens
            kwargs["temperature"] = fallback.temperature
            if fallback.reasoning_effort is None:
                kwargs.pop("reasoning_effort", None)
            else:
                kwargs["reasoning_effort"] = fallback.reasoning_effort
            try:
                fallback_response = await call(fallback_provider, kwargs)
            finally:
                for name, value in original_values.items():
                    if value is _MISSING:
                        kwargs.pop(name, None)
                    else:
                        kwargs[name] = value

            if fallback_response.finish_reason != "error":
                logger.info(
                    "Fallback '{}' succeeded after primary '{}' failed",
                    fallback_model, primary_model,
                )
                return fallback_response

            last_response = fallback_response
            logger.warning(
                "Fallback '{}' also failed: {}",
                fallback_model,
                (fallback_response.content or "")[:120],
            )

        logger.warning(
            "All {} fallback model(s) failed",
            len(self._fallback_presets),
        )
        # 全部失败时，优先返回最后一次真实失败的响应对象。
        if last_response is not None:
            return last_response
        # 如果主模型已经熔断且没有 fallback，就构造一个明确错误。
        return LLMResponse(
            content=f"Primary model '{primary_model}' circuit open and no fallbacks available",
            finish_reason="error",
        )

    @staticmethod
    def _should_fallback(response: LLMResponse) -> bool:
        """判断当前错误是否属于“应该尝试降级”的类型。

        大致原则：
        - 权限、认证、内容过滤、请求参数错误：不降级
        - 超时、限流、连接错误、5xx、服务过载：允许降级

        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。
        """
        if response.error_should_retry is False:
            return False
        status = response.error_status_code
        kind = (response.error_kind or "").lower()
        error_type = (response.error_type or "").lower()
        code = (response.error_code or "").lower()
        text = (response.content or "").lower()

        if status in {400, 401, 403, 404, 422}:
            return False
        if kind in _NON_FALLBACK_ERROR_KINDS:
            return False
        if any(token in value for value in (kind, error_type, code) for token in _NON_FALLBACK_ERROR_KINDS):
            return False
        if response.error_should_retry is True:
            return True
        if status is not None and (status in {408, 409, 429} or 500 <= status <= 599):
            return True
        if kind in _FALLBACK_ERROR_KINDS:
            return True
        return any(token in value for value in (kind, error_type, code, text) for token in _FALLBACK_ERROR_TOKENS)
