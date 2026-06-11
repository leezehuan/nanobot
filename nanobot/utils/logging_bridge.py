"""日志桥接工具：把标准库 logging 的日志重定向到 loguru。"""
from __future__ import annotations

import logging

from loguru import logger


class _LoguruBridge(logging.Handler):
    """把标准 logging 的 LogRecord 转发到 loguru，并统一格式。"""

    _LEVEL_MAP: dict[int, str] = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, lib_name: str) -> None:
        super().__init__()
        self.lib_name = lib_name

    def emit(self, record: logging.LogRecord) -> None:
        """处理一条标准库日志记录，并按 loguru 方式输出。"""
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame, depth = frame.f_back, depth + 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, "[{lib}] {message}", lib=self.lib_name, message=record.getMessage()
        )


def redirect_lib_logging(name: str, level: str | None = None) -> None:
    """把指定库的 stdlib logging 重定向到 loguru。

    这样第三方库日志就能和 nanobot 自己的 loguru 日志使用同一套展示风格。
    """
    lib_logger = logging.getLogger(name)
    if not any(isinstance(h, _LoguruBridge) for h in lib_logger.handlers):
        handler = _LoguruBridge(name)
        if level is not None:
            handler.setLevel(getattr(logging, level.upper(), logging.WARNING))
        lib_logger.handlers = [handler]
        lib_logger.propagate = False
