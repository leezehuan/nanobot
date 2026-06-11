"""Azure OpenAI Provider：通过 OpenAI Python SDK 调用 Azure 上的 Responses API。

这个 provider 复用了 OpenAI SDK，但把 ``base_url`` 指向 Azure 的：
``https://{endpoint}/openai/v1/``

它支持两种认证模式：
1. 传统 API Key
2. Microsoft Entra ID（AAD）动态换取 bearer token
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.openai_responses import (
    consume_sdk_stream,
    convert_messages,
    convert_tools,
    parse_response_output,
)

_AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


class _AzureTokenProvider:
    """为 Azure AAD 认证提供异步 bearer token 的小包装器。

    OpenAI SDK 允许把 ``api_key`` 传成一个异步可调用对象；
    SDK 会在每次请求时调用它，拿到最新 token。
    这个类就是为此服务的。
    """

    def __init__(self, scope: str = _AZURE_OPENAI_SCOPE) -> None:
        try:
            from azure.identity.aio import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError(
                "Azure OpenAI AAD authentication requires the 'azure-identity' package. "
                "Install it with: pip install 'nanobot-ai[azure]'"
            ) from exc

        self._scope = scope
        self._credential = DefaultAzureCredential()

    async def __call__(self) -> str:
        """返回当前 scope 对应的 bearer token。"""
        access_token = await self._credential.get_token(self._scope)
        return access_token.token

    async def aclose(self) -> None:
        """释放凭证对象占用的资源；重复调用也是安全的。"""
        close = getattr(self._credential, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


class AzureOpenAIProvider(LLMProvider):
    """【中文名称】Azure OpenAI Provider

    【功能说明】
    基于 Azure Responses API 调用 Azure 托管的 OpenAI 模型。复用 OpenAI SDK，
    但将 base_url 指向 Azure 的 ``https://{endpoint}/openai/v1/`` 格式。

    支持两种认证模式：
    1. 传统 API Key
    2. Microsoft Entra ID（AAD）动态换取 bearer token（通过 _AzureTokenProvider）

    基于 Azure Responses API 的 Azure OpenAI Provider。"""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        default_model: str = "gpt-5.2-chat",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model

        if not api_base:
            raise ValueError("Azure OpenAI api_base is required")

        # 统一补齐末尾斜杠，后面构造 base_url 时更稳定。
        if not api_base.endswith("/"):
            api_base += "/"
        self.api_base = api_base

        # 认证模式选择：
        # - 若传了 api_key，就走传统静态密钥
        # - 否则退回到 AAD，按请求动态取 bearer token
        self._token_provider: _AzureTokenProvider | None = None
        client_api_key: str | Callable[[], Awaitable[str]]
        if api_key:
            client_api_key = api_key
        else:
            self._token_provider = _AzureTokenProvider()
            client_api_key = self._token_provider

        # 底层仍然用 OpenAI SDK client，只是目标端点改成 Azure。
        base_url = f"{api_base.rstrip('/')}/openai/v1/"
        self._client = AsyncOpenAI(
            api_key=client_api_key,
            base_url=base_url,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
            max_retries=0,
        )

    # 辅助方法：负责识别能力差异、组装请求体、统一错误处理。

    @staticmethod
    def _supports_temperature(
        deployment_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        """判断当前 Azure deployment 大概率是否支持 ``temperature`` 参数。"""
        if reasoning_effort and reasoning_effort.lower() != "none":
            return False
        name = deployment_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """把 Chat-Completions 风格参数组装成 Azure Responses API 请求体。"""
        deployment = model or self.default_model
        instructions, input_items = convert_messages(self._sanitize_empty_content(messages))

        body: dict[str, Any] = {
            "model": deployment,
            "instructions": instructions or None,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
            "store": False,
            "stream": False,
        }

        if self._supports_temperature(deployment, reasoning_effort):
            body["temperature"] = temperature

        if reasoning_effort and reasoning_effort.lower() != "none":
            body["reasoning"] = {"effort": reasoning_effort}
            body["include"] = ["reasoning.encrypted_content"]

        if tools:
            body["tools"] = convert_tools(tools)
            body["tool_choice"] = tool_choice or "auto"

        return body

    @staticmethod
    def _handle_error(e: Exception) -> LLMResponse:
        """把 SDK / HTTP 异常统一转换成 ``LLMResponse`` 错误对象。"""
        response = getattr(e, "response", None)
        body = getattr(e, "body", None) or getattr(response, "text", None)
        body_text = str(body).strip() if body is not None else ""
        msg = f"Error: {body_text[:500]}" if body_text else f"Error calling Azure OpenAI: {e}"
        retry_after = LLMProvider._extract_retry_after_from_headers(getattr(response, "headers", None))
        if retry_after is None:
            retry_after = LLMProvider._extract_retry_after(msg)
        return LLMResponse(content=msg, finish_reason="error", retry_after=retry_after)

    # 对外公开接口：实现 LLMProvider 规定的 chat / chat_stream / get_default_model。

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
        """非流式调用 Azure Responses API。"""
        body = self._build_body(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        try:
            response = await self._client.responses.create(**body)
            return parse_response_output(response)
        except Exception as e:
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
        _ = on_thinking_delta
        """流式调用 Azure Responses API，并消费 SDK stream。"""
        body = self._build_body(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        body["stream"] = True

        try:
            stream = await self._client.responses.create(**body)
            content, tool_calls, finish_reason, usage, reasoning_content = (
                await consume_sdk_stream(stream, on_content_delta, on_tool_call_delta)
            )
            return LLMResponse(
                content=content or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model
