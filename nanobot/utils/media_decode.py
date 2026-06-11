"""媒体解码工具：把 ``data:...;base64,...`` URL 保存成真实文件。

这个模块被 API server 与 WebSocket 渠道共享，保证不同入口都使用同一套：
- data URL 解析规则
- 文件大小限制
- 文件落盘命名方式
"""

from __future__ import annotations

import base64
import mimetypes
import re
import uuid
from pathlib import Path

from nanobot.utils.helpers import safe_filename

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
MAX_FILE_SIZE = DEFAULT_MAX_BYTES

_DATA_URL_RE = re.compile(r"^data:([^;,]+)(?:;[^,]*)*;base64,(.+)$", re.DOTALL)
_MIME_EXTENSION_OVERRIDES = {
    # Python 的 mimetypes 在某些平台上会返回比较冷门的扩展名，
    # 但部分转写 API 会严格校验后缀，所以这里手动改成更常见的容器后缀。
    "application/ogg": ".ogg",
    "audio/ogg": ".ogg",
    "audio/mpga": ".mpga",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "audio/vnd.wave": ".wav",
    "video/webm": ".webm",
}


class FileSizeExceededError(Exception):
    """解码后的 payload 超过大小上限时抛出的异常。"""


FileSizeExceeded = FileSizeExceededError


def save_base64_data_url(
    data_url: str,
    media_dir: Path,
    *,
    max_bytes: int | None = None,
) -> str | None:
    """解码 ``data:<mime>;base64,<payload>`` URL，并把结果写入磁盘。

    返回：
    - 成功：保存后的绝对路径
    - URL 形状或 base64 内容非法：``None``
    - 超过大小限制：抛 ``FileSizeExceeded``
    """
    m = _DATA_URL_RE.match(data_url)
    if not m:
        return None
    mime_type, b64_payload = m.group(1).strip().lower(), m.group(2)
    try:
        raw = base64.b64decode(b64_payload)
    except Exception:
        return None
    limit = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    if len(raw) > limit:
        raise FileSizeExceeded(f"File exceeds {limit // (1024 * 1024)}MB limit")
    ext = _MIME_EXTENSION_OVERRIDES.get(mime_type) or mimetypes.guess_extension(mime_type) or ".bin"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    dest = media_dir / safe_filename(filename)
    dest.write_bytes(raw)
    return str(dest)
