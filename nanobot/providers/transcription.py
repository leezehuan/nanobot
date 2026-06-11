"""语音转写 Provider 适配层。

【中文名称】语音转写适配层

【功能说明】
这个模块封装了 6 种语音转写 API 的调用细节，所有 provider 都通过
统一的 ``transcribe(file_path) -> str`` 接口对外提供服务：

1. OpenAI Whisper：标准的 ``/audio/transcriptions`` multipart 上传
2. Groq Whisper：与 OpenAI 兼容，但独立管理 API key 和 base URL
3. OpenRouter Whisper：走 JSON body 格式（base64 音频），非 multipart
4. Xiaomi MiMo ASR：chat-completions 风格的 ASR 接口
5. StepFun ASR：Server-Sent Events (SSE) 流式转写
6. AssemblyAI：异步模式——上传 → 轮询 → 获取结果

【重试策略】
所有请求都通过 _post_with_retry 或 _request_json_with_retry 包装，
处理以下瞬时故障：
- 网络异常：TimeoutException / ConnectError / ReadError 等
- HTTP 状态码：408 / 429 / 500 / 502 / 503 / 504
- 最多重试 3 次，指数退避：1s → 2s → 4s

【架构位置】
channel 适配器（如 telegram.py）收到语音消息后，调用 base.py 的
transcribe_audio 方法，后者再根据配置匹配对应的 provider 类来执行转写。

这个模块只负责“怎么调用外部转写 API”，例如：

- OpenAI Whisper
- Groq Whisper
- OpenRouter
- Xiaomi MiMo ASR
- AssemblyAI

它不负责产品层面的上传校验、WebUI 交互或渠道集成；
那些更上层的逻辑在别的模块里。
"""

import asyncio
import base64
import json
import mimetypes
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

_CHAT_COMPLETIONS_PATH = "chat/completions"
_TRANSCRIPTIONS_PATH = "audio/transcriptions"
_STEPFUN_ASR_PATH = "audio/asr/sse"
_ASSEMBLYAI_DEFAULT_API_BASE = "https://api.assemblyai.com/v2"
_ASSEMBLYAI_POLL_ATTEMPTS = 60
_ASSEMBLYAI_POLL_INTERVAL_S = 2.0
_AUDIO_MIME_OVERRIDES = {
    ".m4a": "audio/mp4",
    ".mpga": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".weba": "audio/webm",
    ".webm": "audio/webm",
}
_FORMAT_ALIASES = {
    "oga": "ogg",
    "opus": "ogg",
    "mpga": "mp3",
    "mpeg": "mp3",
    "mp4": "m4a",
}


def _resolve_transcription_url(api_base: str | None, default_url: str) -> str:
    """解析完整的 transcription endpoint URL。

    这里既支持传入 chat 风格 base URL，也支持直接传完整转写 endpoint。
    如果用户给的是 base URL，就自动补上 ``/audio/transcriptions``。
    """
    if not api_base:
        return default_url
    base = api_base.rstrip("/")
    if base.endswith(_TRANSCRIPTIONS_PATH):
        return base
    return f"{base}/{_TRANSCRIPTIONS_PATH}"


def _resolve_chat_completions_url(api_base: str | None, default_url: str) -> str:
    """为使用 chat payload 的 ASR provider 解析 chat-completions endpoint。"""
    if not api_base:
        return default_url
    base = api_base.rstrip("/")
    if base.endswith(_CHAT_COMPLETIONS_PATH):
        return base
    return f"{base}/{_CHAT_COMPLETIONS_PATH}"


def _resolve_api_path(api_base: str | None, default_base: str, path: str) -> str:
    base = (api_base or default_base).rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _resolve_stepfun_asr_url(api_base: str | None) -> str:
    base = (api_base or "https://api.stepfun.com/v1").rstrip("/")
    if base.endswith(_STEPFUN_ASR_PATH):
        return base
    return f"{base}/{_STEPFUN_ASR_PATH}"


def _audio_mime_type(path: Path) -> str:
    return (
        _AUDIO_MIME_OVERRIDES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _audio_format(path: Path) -> str:
    """把音频文件扩展名映射成 OpenRouter 所需的 ``format`` 值。"""
    ext = path.suffix.lstrip(".").lower()
    return _FORMAT_ALIASES.get(ext, ext)


# 最多重试 3 次（共 4 次尝试），并采用指数退避。
# 语音转写接口在高负载或移动网络环境下经常出现瞬时失败；
# 如果完全不重试，一条语音消息很容易直接变成空字符串。
_MAX_RETRIES = 3
_BACKOFF_S = (1.0, 2.0, 4.0)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


async def _request_json_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider_label: str,
    **kwargs: object,
) -> dict[str, Any] | None:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            request = getattr(client, method.lower(), None)
            if request is None:
                response = await client.request(method, url, **kwargs)
            else:
                response = await request(url, **kwargs)
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "{} transcription transient error (attempt {}/{}): {}",
                    provider_label,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    e,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue
            logger.exception(
                "{} transcription error after {} attempts: {}",
                provider_label,
                _MAX_RETRIES + 1,
                e,
            )
            return None
        except Exception as e:
            logger.exception("{} transcription error: {}", provider_label, e)
            return None

        if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
            logger.warning(
                "{} transcription transient HTTP {} (attempt {}/{})",
                provider_label,
                response.status_code,
                attempt + 1,
                _MAX_RETRIES + 1,
            )
            await asyncio.sleep(_BACKOFF_S[attempt])
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            body = response.text.strip().replace("\n", " ")[:500]
            logger.error(
                "{} transcription HTTP {}{}{}",
                provider_label,
                response.status_code,
                f" {response.reason_phrase}" if response.reason_phrase else "",
                f": {body}" if body else "",
            )
            return None
        except Exception as e:
            logger.exception("{} transcription error: {}", provider_label, e)
            return None

        try:
            payload = response.json()
        except Exception as e:
            logger.exception(
                "{} transcription error: malformed response body: {}",
                provider_label,
                e,
            )
            return None
        if not isinstance(payload, dict):
            logger.error(
                "{} transcription error: unexpected response shape: {!r}",
                provider_label,
                type(payload).__name__,
            )
            return None
        return payload
    return None


async def _post_transcription_with_retry(
    url: str,
    *,
    api_key: str | None,
    path: Path,
    model: str,
    provider_label: str,
    language: str | None = None,
) -> str:
    """以 multipart 方式上传音频做转写，并在瞬时错误时自动重试。

    会重试连接/读取/超时错误，以及 408/429/5xx 响应；
    对于 401/403 这类明显配置错误，不会盲目重试。
    """
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""
    headers = {"Authorization": f"Bearer {api_key}"}

    def build_request() -> dict[str, Any]:
        files = {
            "file": (path.name, data, _audio_mime_type(path)),
            "model": (None, model),
        }
        if language:
            files["language"] = (None, language)
        return {"url": url, "headers": headers, "files": files, "timeout": 60.0}

    return await _post_with_retry(build_request, provider_label, _text_from_transcription_payload)


async def _post_json_transcription_with_retry(
    url: str,
    *,
    api_key: str | None,
    path: Path,
    model: str,
    provider_label: str,
    language: str | None = None,
) -> str:
    """给不支持 multipart 的 provider 发送 base64 JSON 音频。"""
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def build_request() -> dict[str, Any]:
        body: dict[str, object] = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(data).decode(),
                "format": _audio_format(path),
            },
        }
        if language:
            body["language"] = language
        return {"url": url, "headers": headers, "json": body, "timeout": 60.0}

    return await _post_with_retry(build_request, provider_label, _text_from_transcription_payload)


async def _post_xiaomi_mimo_asr_with_retry(
    url: str,
    *,
    api_key: str | None,
    path: Path,
    model: str,
    provider_label: str,
    language: str | None = None,
) -> str:
    """调用 Xiaomi MiMo 的 chat-completions 风格 ASR 接口。"""
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": (
                                f"data:{_audio_mime_type(path)};base64,"
                                f"{base64.b64encode(data).decode('ascii')}"
                            ),
                        },
                    }
                ],
            }
        ],
    }
    if language:
        body["asr_options"] = {"language": language}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def build_request() -> dict[str, Any]:
        return {"url": url, "headers": headers, "json": body, "timeout": 60.0}

    return await _post_with_retry(build_request, provider_label, _text_from_chat_payload)


async def _post_stepfun_asr_with_retry(
    url: str,
    *,
    api_key: str | None,
    path: Path,
    model: str,
    provider_label: str,
    language: str | None = None,
) -> str:
    """调用 StepFun 的 SSE 语音转写接口，并收集最终文本。"""
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""

    suffix = path.suffix.lstrip(".").lower()
    audio_type = suffix if suffix in ("ogg", "mp3", "wav", "pcm") else "wav"

    body: dict[str, Any] = {
        "audio": {
            "data": base64.b64encode(data).decode("ascii"),
            "input": {
                "transcription": {
                    "model": model,
                    "enable_itn": True,
                },
                "format": {"type": audio_type},
            },
        },
    }
    if language:
        body["audio"]["input"]["transcription"]["language"] = language

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with client.stream(
                    "POST", url, headers=headers, json=body, timeout=60.0
                ) as resp:
                    if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                        logger.warning(
                            "{} transcription transient HTTP {} (attempt {}/{})",
                            provider_label,
                            resp.status_code,
                            attempt + 1,
                            _MAX_RETRIES + 1,
                        )
                        await asyncio.sleep(_BACKOFF_S[attempt])
                        continue
                    resp.raise_for_status()
                    final_text = None
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[len("data:") :].strip()
                        if not payload_str:
                            continue
                        try:
                            payload = json.loads(payload_str)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        event_type = payload.get("type", "")
                        if event_type == "error":
                            msg = payload.get("message", "unknown error")
                            logger.error("{} ASR error: {}", provider_label, msg)
                            return ""
                        if event_type == "transcript.text.done":
                            final_text = payload.get("text", "")
                            break
                    if final_text is not None:
                        return final_text
                    # 流提前结束但没有 final event，若还有次数就重试。
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "{} transcription: no final event (attempt {}/{})",
                            provider_label,
                            attempt + 1,
                            _MAX_RETRIES + 1,
                        )
                        await asyncio.sleep(_BACKOFF_S[attempt])
                        continue
                    logger.error(
                        "{} transcription: stream ended without final text after {} attempts",
                        provider_label,
                        _MAX_RETRIES + 1,
                    )
                    return ""
            except httpx.HTTPStatusError as e:
                if e.response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.error(
                    "{} transcription HTTP {}{}",
                    provider_label,
                    e.response.status_code,
                    f" {e.response.reason_phrase}" if e.response.reason_phrase else "",
                )
                return ""
            except (httpx.RequestError, Exception):
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.exception("{} transcription request error", provider_label)
                return ""
    return ""


async def _post_with_retry(
    build_request: Callable[[], dict[str, Any]],
    provider_label: str,
    extract_text: Callable[[dict[str, Any]], str],
) -> str:
    """带重试的通用 HTTP POST 转写请求包装器。

    【中文名称】转写请求（带重试）

    【功能说明】
    所有转写 provider 共享的重试逻辑层。build_request 返回参数字典后，
    用 httpx 发送请求，在以下情况重试（最多 3 次+1 次初始尝试，指数退避）：

    1. 网络异常（TimeoutException / ConnectError / ReadError 等）
    2. 可重试 HTTP 状态码（408 / 429 / 500 / 502 / 503 / 504）
    3. 非重试错误（401 / 403 / 400 等）直接返回 ""，不做无用重试

    成功后通过 extract_text 回调从 payload 中提取转写文本。

    【参数说明】
    - build_request: () -> dict → 每次重试都重新调用以构建请求参数
      （因为 httpx.AsyncClient 可能连接过期）
    - provider_label: str → provider 名称（用于日志前缀）
    - extract_text: dict -> str → 从 payload 中提取纯文本的回调

    【返回值】
    - str → 转写文本，失败时返回空字符串
    """
    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.post(**build_request())
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "{} transcription transient error (attempt {}/{}): {}",
                        provider_label,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        e,
                    )
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.exception(
                    "{} transcription error after {} attempts: {}",
                    provider_label,
                    _MAX_RETRIES + 1,
                    e,
                )
                return ""
            except Exception as e:
                logger.exception("{} transcription error: {}", provider_label, e)
                return ""

            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                logger.warning(
                    "{} transcription transient HTTP {} (attempt {}/{})",
                    provider_label,
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                body = response.text.strip().replace("\n", " ")[:500]
                logger.error(
                    "{} transcription HTTP {}{}{}",
                    provider_label,
                    response.status_code,
                    f" {response.reason_phrase}" if response.reason_phrase else "",
                    f": {body}" if body else "",
                )
                return ""
            except Exception as e:
                logger.exception("{} transcription error: {}", provider_label, e)
                return ""

            try:
                payload = response.json()
            except Exception as e:
                logger.exception(
                    "{} transcription error: malformed response body: {}",
                    provider_label,
                    e,
                )
                return ""
            if not isinstance(payload, dict):
                logger.error(
                    "{} transcription error: unexpected response shape: {!r}",
                    provider_label,
                    type(payload).__name__,
                )
                return ""
            return extract_text(payload)
    return ""


def _text_from_transcription_payload(payload: dict[str, Any]) -> str:
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def _text_from_chat_payload(payload: dict[str, Any]) -> str:
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return text if isinstance(text, str) else ""


def _assemblyai_speech_models(model: str | None) -> list[str]:
    return [part for part in (part.strip() for part in (model or "").split(",")) if part]


class AssemblyAITranscriptionProvider:
    """AssemblyAI 语音转写客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        base = api_base or os.environ.get("ASSEMBLYAI_BASE_URL")
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        self.upload_url = _resolve_api_path(base, _ASSEMBLYAI_DEFAULT_API_BASE, "upload")
        self.transcript_url = _resolve_api_path(base, _ASSEMBLYAI_DEFAULT_API_BASE, "transcript")
        self.language = language or None
        self.model = model or "universal-3-pro,universal-2"
        logger.debug("AssemblyAI transcription endpoint: {}", self.transcript_url)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("AssemblyAI API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.exception("AssemblyAI transcription error: cannot read audio file: {}", e)
            return ""

        headers = {"Authorization": self.api_key}
        async with httpx.AsyncClient() as client:
            upload = await _request_json_with_retry(
                client,
                "POST",
                self.upload_url,
                provider_label="AssemblyAI",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=data,
                timeout=60.0,
            )
            upload_url = upload.get("upload_url") if upload else None
            if not isinstance(upload_url, str) or not upload_url:
                logger.error("AssemblyAI transcription error: upload_url missing")
                return ""

            body: dict[str, object] = {"audio_url": upload_url}
            speech_models = _assemblyai_speech_models(self.model)
            if speech_models:
                body["speech_models"] = speech_models
            if self.language:
                body["language_code"] = self.language

            transcript = await _request_json_with_retry(
                client,
                "POST",
                self.transcript_url,
                provider_label="AssemblyAI",
                headers=headers,
                json=body,
                timeout=30.0,
            )
            transcript_id = transcript.get("id") if transcript else None
            if not isinstance(transcript_id, str) or not transcript_id:
                logger.error("AssemblyAI transcription error: transcript id missing")
                return ""

            poll_url = f"{self.transcript_url.rstrip('/')}/{transcript_id}"
            for attempt in range(_ASSEMBLYAI_POLL_ATTEMPTS):
                payload = await _request_json_with_retry(
                    client,
                    "GET",
                    poll_url,
                    provider_label="AssemblyAI",
                    headers=headers,
                    timeout=30.0,
                )
                if not payload:
                    return ""
                status = str(payload.get("status") or "").lower()
                if status == "completed":
                    text = payload.get("text")
                    return text if isinstance(text, str) else ""
                if status in {"error", "failed"}:
                    logger.error(
                        "AssemblyAI transcription failed: {}",
                        payload.get("error") or payload,
                    )
                    return ""
                if attempt < _ASSEMBLYAI_POLL_ATTEMPTS - 1:
                    await asyncio.sleep(_ASSEMBLYAI_POLL_INTERVAL_S)
            logger.error("AssemblyAI transcription timed out while polling transcript")
            return ""


class OpenAITranscriptionProvider:
    """OpenAI Whisper 语音转写客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = _resolve_transcription_url(
            api_base or os.environ.get("OPENAI_TRANSCRIPTION_BASE_URL"),
            "https://api.openai.com/v1/audio/transcriptions",
        )
        self.language = language or None
        self.model = model or "whisper-1"
        logger.debug("OpenAI transcription endpoint: {}", self.api_url)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenAI API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        return await _post_transcription_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            provider_label="OpenAI",
            language=self.language,
        )


class GroqTranscriptionProvider:
    """Groq Whisper 语音转写客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = _resolve_transcription_url(
            api_base or os.environ.get("GROQ_BASE_URL"),
            "https://api.groq.com/openai/v1/audio/transcriptions",
        )
        self.language = language or None
        self.model = model or "whisper-large-v3"
        logger.debug("Groq transcription endpoint: {}", self.api_url)

    async def transcribe(self, file_path: str | Path) -> str:
        """把音频文件交给 Groq 转写，并返回文本。"""
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        return await _post_transcription_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            provider_label="Groq",
            language=self.language,
        )


class OpenRouterTranscriptionProvider:
    """OpenRouter 语音转写客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_url = _resolve_transcription_url(
            api_base or os.environ.get("OPENROUTER_BASE_URL"),
            "https://openrouter.ai/api/v1/audio/transcriptions",
        )
        self.language = language or None
        self.model = model or "openai/whisper-1"
        logger.debug("OpenRouter transcription endpoint: {}", self.api_url)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenRouter API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        return await _post_json_transcription_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            provider_label="OpenRouter",
            language=self.language,
        )


class XiaomiMiMoTranscriptionProvider:
    """Xiaomi MiMo ASR 语音转写客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("MIMO_API_KEY")
        self.api_url = _resolve_chat_completions_url(
            api_base or os.environ.get("MIMO_API_BASE"),
            "https://api.xiaomimimo.com/v1/chat/completions",
        )
        self.language = language or None
        self.model = model or "mimo-v2.5-asr"
        logger.debug("Xiaomi MiMo transcription endpoint: {}", self.api_url)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("Xiaomi MiMo API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        return await _post_xiaomi_mimo_asr_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            provider_label="Xiaomi MiMo",
            language=self.language,
        )


class StepFunTranscriptionProvider:
    """StepFun SSE 语音转写客户端。"""

    _DEFAULT_URL = "https://api.stepfun.com/v1/audio/asr/sse"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("STEPFUN_API_KEY")
        # api_base 既可以是 StepFun 的 base URL，也可以直接是完整 SSE endpoint。
        self.api_url = _resolve_stepfun_asr_url(api_base)
        self.language = language or None
        self.model = model or "stepaudio-2.5-asr"
        logger.debug("StepFun transcription endpoint: {}", self.api_url)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("StepFun API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        return await _post_stepfun_asr_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            provider_label="StepFun",
            language=self.language,
        )
