"""旧版 WebUI 线程快照文件辅助函数。

【中文名称】旧线程 JSON 快照清理器

nanobot 现在主要依赖 transcript 记录会话，但历史上 WebUI 还会额外维护
JSON 快照文件。这个模块就是围绕那套旧文件的路径计算与删除逻辑。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from nanobot.config.paths import get_webui_dir
from nanobot.session.manager import SessionManager
from nanobot.webui.transcript import delete_webui_transcript


def webui_thread_file_path(session_key: str) -> Path:
    """根据 session_key 生成旧版 WebUI JSON 快照文件路径。"""
    stem = SessionManager.safe_key(session_key)
    return get_webui_dir() / f"{stem}.json"


def delete_webui_thread(session_key: str) -> bool:
    """删除某个会话的旧版 JSON 快照和新的 transcript。

    返回 `True` 表示至少删除成功了一项。
    """
    removed = False
    path = webui_thread_file_path(session_key)
    if path.is_file():
        try:
            path.unlink()
            removed = True
        except OSError as e:
            logger.warning("Failed to delete webui thread file {}: {}", path, e)
    if delete_webui_transcript(session_key):
        removed = True
    return removed
