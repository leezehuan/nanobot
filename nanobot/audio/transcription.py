"""应用层的音频转写服务。

【中文名称】音频转文字服务

这个模块负责的是“转写流程编排”，而不是具体厂商 API 的细节。
它主要完成下面几件事：

1. 从总配置里解析出“当前应该用哪个转写 provider”；
2. 兼容旧版 channel 配置字段；
3. 校验 WebUI 上传的音频数据是否合法；
4. 把 data URL 落盘成临时文件；
5. 调用 provider 适配器去真正转写；
6. 在结束后清理临时文件。

真正和 OpenAI / Groq / StepFun 等厂商 HTTP 协议打交道的代码，
在 `nanobot.providers.transcription`。
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.audio.transcription_registry import (
    get_transcription_provider,
    resolve_transcription_provider,
)
from nanobot.config.paths import get_media_dir
from nanobot.utils.media_decode import FileSizeExceeded, save_base64_data_url

TranscriptionProviderName = str

_DEFAULT_PROVIDER: TranscriptionProviderName = "groq"
_MAX_AUDIO_BYTES_FALLBACK = 25 * 1024 * 1024
_AUDIO_MIME_ALLOWED: frozenset[str] = frozenset({
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
})


@dataclass(frozen=True)
class EffectiveTranscriptionConfig:
    """转写阶段真正生效的一份配置快照。

    顶层原始配置通常比较分散，还要兼容旧字段。
    这个 dataclass 的意义，就是把“已经解析完”的配置收束成一份
    干净、可直接执行的对象，后面的转写流程只读它，不再反复查原配置。
    """
    enabled: bool
    provider: TranscriptionProviderName
    model: str
    language: str | None
    api_key: str = field(repr=False)
    api_base: str
    max_duration_sec: int
    max_upload_mb: int

    @property
    def configured(self) -> bool:
        """判断当前配置是否具备“可以真正调用 provider”的最小条件。"""
        return bool(self.api_key)


class TranscriptionIngressError(Exception):
    """暴露给 WebUI 的稳定转写入口错误。

    这里不用随意抛裸异常，而是统一用一组可预期的错误码语义，例如：

    - `missing_audio`
    - `disabled`
    - `not_configured`
    - `mime`
    - `size`

    前端就可以根据这些稳定标识，给用户展示明确提示。
    """

    def __init__(self, detail: str, **extra: Any):
        super().__init__(detail)
        self.detail = detail
        self.extra = extra


def _as_provider(value: Any) -> TranscriptionProviderName | None:
    """把任意输入规范化为 provider 名称。"""
    spec = resolve_transcription_provider(value)
    return spec.name if spec else None


def _provider_config(config: Any, provider: str) -> Any:
    """从总配置对象中取出某个 provider 的专属配置块。"""
    return getattr(getattr(config, "providers", None), provider, None)


def _extract_data_url_mime(url: str) -> str | None:
    """从 `data:` URL 头部提取 MIME 类型。"""
    header, _, _ = url.partition(",")
    if not header.startswith("data:") or ";base64" not in header:
        return None
    return header[5:].split(";", 1)[0].strip().lower() or None


def resolve_transcription_config(config: Any) -> EffectiveTranscriptionConfig:
    """解析出最终生效的转写配置。

    【解析顺序】

    provider 会按下面顺序兜底：

    1. `config.transcription.provider`
    2. 旧版 `config.channels.transcription_provider`
    3. 默认值 `_DEFAULT_PROVIDER`

    这样做是为了既支持新配置结构，也兼容老用户配置。
    """
    top = getattr(config, "transcription", None)
    channels = getattr(config, "channels", None)
    provider = (
        _as_provider(getattr(top, "provider", None))
        or _as_provider(getattr(channels, "transcription_provider", None))
        or _DEFAULT_PROVIDER
    )
    spec = get_transcription_provider(provider)
    if spec is None:
        logger.warning("Unknown transcription provider {}; falling back to {}", provider, _DEFAULT_PROVIDER)
        provider = _DEFAULT_PROVIDER
        spec = get_transcription_provider(provider)
    default_model = spec.default_model if spec else ""
    provider_cfg = _provider_config(config, provider)
    return EffectiveTranscriptionConfig(
        enabled=bool(getattr(top, "enabled", True)),
        provider=provider,
        model=(getattr(top, "model", None) or default_model).strip(),
        language=getattr(top, "language", None) or getattr(channels, "transcription_language", None),
        api_key=getattr(provider_cfg, "api_key", None) or "",
        api_base=getattr(provider_cfg, "api_base", None) or "",
        max_duration_sec=int(getattr(top, "max_duration_sec", 120)),
        max_upload_mb=int(getattr(top, "max_upload_mb", 25)),
    )


async def transcribe_audio_data_url(
    data_url: Any,
    config: EffectiveTranscriptionConfig,
    *,
    duration_ms: Any = None,
) -> str:
    """把 WebUI 传来的音频 data URL 转成文字。

    【完整流程】

    1. 校验 data URL 是否存在；
    2. 检查转写功能是否启用、是否已配置密钥；
    3. 检查录音时长和 MIME 类型；
    4. 把 base64 音频保存到临时目录；
    5. 调用 `transcribe_audio_file` 做真正转写；
    6. 无论成功失败，都清理临时文件；
    7. 如果结果为空，抛出 `empty` 错误。
    """
    if not isinstance(data_url, str) or not data_url:
        raise TranscriptionIngressError("missing_audio")
    if not config.enabled:
        raise TranscriptionIngressError("disabled")
    if not config.configured:
        raise TranscriptionIngressError("not_configured", provider=config.provider)
    if (
        isinstance(duration_ms, (int, float))
        and duration_ms > (config.max_duration_sec * 1000 + 1000)
    ):
        raise TranscriptionIngressError("duration")
    if _extract_data_url_mime(data_url) not in _AUDIO_MIME_ALLOWED:
        raise TranscriptionIngressError("mime")

    audio_path: str | None = None
    max_bytes = max(
        1,
        config.max_upload_mb * 1024 * 1024 if config.max_upload_mb else _MAX_AUDIO_BYTES_FALLBACK,
    )
    try:
        audio_path = save_base64_data_url(
            data_url,
            get_media_dir("webui-transcription"),
            max_bytes=max_bytes,
        )
    except FileSizeExceeded as exc:
        raise TranscriptionIngressError("size") from exc
    except Exception as exc:
        logger.warning("transcription audio decode failed: {}", exc)
    if not audio_path:
        raise TranscriptionIngressError("decode")

    try:
        text = await transcribe_audio_file(audio_path, config)
    finally:
        with suppress(OSError):
            Path(audio_path).unlink(missing_ok=True)
    if not text:
        raise TranscriptionIngressError("empty")
    return text


async def transcribe_audio_file(
    file_path: str | Path,
    config: EffectiveTranscriptionConfig,
) -> str:
    """使用已经解析好的配置转写本地音频文件。

    这个函数假设：

    - 路径已经准备好；
    - provider / model / api_key 已经在 `config` 中解析完毕。

    它的职责只剩两步：

    1. 根据 provider 名称装载对应适配器类；
    2. 实例化适配器并调用 `transcribe`。
    """
    if not config.enabled or not config.configured:
        return ""
    spec = get_transcription_provider(config.provider)
    if spec is None:
        logger.warning("Unknown transcription provider: {}", config.provider)
        return ""
    provider = spec.load_adapter()(
        api_key=config.api_key,
        api_base=config.api_base or None,
        language=config.language,
        model=config.model,
    )
    return await provider.transcribe(file_path)
