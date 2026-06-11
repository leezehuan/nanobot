"""按需检查 `nanobot-ai` 是否有新版本。

【中文名称】版本更新检查器

这个模块不会后台轮询，也不会偷偷常驻请求网络。
只有在显式调用时，它才会去 PyPI 看看：

- 当前本地版本是多少
- PyPI 最新版本是多少
- 是否值得提示用户升级
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from nanobot import __version__

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/nanobot-ai/json"
_CACHE_TTL_S = 300  # 5 minutes cache to avoid hammering PyPI

_cache: tuple[float, str | None] = (0.0, None)


def check_for_update() -> dict[str, Any] | None:
    """检查 PyPI 是否存在比当前版本更高的新版本。

    返回：

    - `dict`：存在新版本时，返回当前版本、最新版本和 PyPI 地址
    - `None`：已经是最新版本，或者检查失败

    这里用了短 TTL 缓存，避免前端频繁点击时反复打 PyPI。
    """
    global _cache
    now = time.monotonic()
    cached_at, cached_val = _cache
    if now - cached_at < _CACHE_TTL_S and cached_val is not None:
        latest = cached_val
    else:
        try:
            resp = httpx.get(_PYPI_URL, timeout=5.0, follow_redirects=True)
            resp.raise_for_status()
            latest = resp.json().get("info", {}).get("version")
        except Exception:
            logger.debug("PyPI version check failed", exc_info=True)
            return None
        _cache = (now, latest)

    if not latest or latest == __version__:
        return None
    return {
        "currentVersion": __version__,
        "latestVersion": latest,
        "pypiUrl": "https://pypi.org/project/nanobot-ai/",
    }
