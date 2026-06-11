"""重启通知辅助工具：让旧进程把“重启完成后要通知谁”传给新进程。"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

RESTART_NOTIFY_CHANNEL_ENV = "NANOBOT_RESTART_NOTIFY_CHANNEL"
RESTART_NOTIFY_CHAT_ID_ENV = "NANOBOT_RESTART_NOTIFY_CHAT_ID"
RESTART_NOTIFY_METADATA_ENV = "NANOBOT_RESTART_NOTIFY_METADATA"
RESTART_STARTED_AT_ENV = "NANOBOT_RESTART_STARTED_AT"


@dataclass(frozen=True)
class RestartNotice:
    """描述一次重启完成后要投递给哪个会话的通知信息。"""
    channel: str
    chat_id: str
    started_at_raw: str
    metadata: dict[str, Any] = field(default_factory=dict)


def format_restart_completed_message(started_at_raw: str) -> str:
    """构造“重启完成”提示文本，并在可行时附上耗时。"""
    elapsed_suffix = ""
    if started_at_raw:
        with suppress(ValueError):
            elapsed_s = max(0.0, time.time() - float(started_at_raw))
            elapsed_suffix = f" in {elapsed_s:.1f}s"
    return f"Restart completed{elapsed_suffix}."


def set_restart_notice_to_env(
    *, channel: str, chat_id: str, metadata: dict[str, Any] | None = None,
) -> None:
    """把重启通知写入环境变量，供下一个进程启动后读取。"""
    os.environ[RESTART_NOTIFY_CHANNEL_ENV] = channel
    os.environ[RESTART_NOTIFY_CHAT_ID_ENV] = chat_id
    os.environ[RESTART_STARTED_AT_ENV] = str(time.time())
    if metadata:
        try:
            os.environ[RESTART_NOTIFY_METADATA_ENV] = json.dumps(metadata, default=str)
        except (TypeError, ValueError):
            os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)
    else:
        os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)


def consume_restart_notice_from_env() -> RestartNotice | None:
    """读取并清空一次性重启通知环境变量。"""
    channel = os.environ.pop(RESTART_NOTIFY_CHANNEL_ENV, "").strip()
    chat_id = os.environ.pop(RESTART_NOTIFY_CHAT_ID_ENV, "").strip()
    started_at_raw = os.environ.pop(RESTART_STARTED_AT_ENV, "").strip()
    metadata_raw = os.environ.pop(RESTART_NOTIFY_METADATA_ENV, "").strip()
    if not (channel and chat_id):
        return None
    metadata: dict[str, Any] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            metadata = parsed
    return RestartNotice(
        channel=channel,
        chat_id=chat_id,
        started_at_raw=started_at_raw,
        metadata=metadata,
    )


def should_show_cli_restart_notice(notice: RestartNotice, session_id: str) -> bool:
    """判断当前 CLI 会话是否应该展示这条重启完成通知。"""
    if notice.channel != "cli":
        return False
    if ":" in session_id:
        _, cli_chat_id = session_id.split(":", 1)
    else:
        cli_chat_id = session_id
    return not notice.chat_id or notice.chat_id == cli_chat_id
