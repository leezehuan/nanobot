"""WebUI 音频转写 WebSocket 事件处理器。

【中文名称】语音转写事件入口

WebSocket 通道本身只负责消息运输和广播分发，
而“某个 envelope 表示一次语音转写请求”这层业务语义由本模块负责。

也就是说，这里处理的是：

- 解析前端发来的转写请求 envelope
- 校验 request_id
- 调用音频转写服务
- 产出统一的成功/失败事件
"""

from __future__ import annotations

from typing import Any

from nanobot.audio.transcription import (
    TranscriptionIngressError,
    resolve_transcription_config,
    transcribe_audio_data_url,
)
from nanobot.config.loader import load_config

_MAX_REQUEST_ID_LENGTH = 80


async def webui_transcription_event(envelope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """处理一条 WebUI 转写请求，并返回 `(事件名, 事件载荷)`。

    返回的事件名有两种：

    - `transcription_result`
    - `transcription_error`
    """
    request_id = envelope.get("request_id")
    valid_request_id = (
        isinstance(request_id, str)
        and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH
    )

    def error(detail: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        """统一构造转写失败事件。"""
        payload: dict[str, Any] = {"detail": detail, **extra}
        if valid_request_id:
            payload["request_id"] = request_id
        return "transcription_error", payload

    if not valid_request_id:
        return error("invalid_request")

    try:
        text = await transcribe_audio_data_url(
            envelope.get("data_url"),
            resolve_transcription_config(load_config()),
            duration_ms=envelope.get("duration_ms"),
        )
    except TranscriptionIngressError as exc:
        return error(exc.detail, **exc.extra)
    return "transcription_result", {"request_id": request_id, "text": text}
