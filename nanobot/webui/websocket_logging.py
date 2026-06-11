"""WebUI WebSocket 服务的日志降噪辅助函数。

【中文名称】WebSocket 日志过滤器

浏览器在页面刷新、标签页关闭、服务重启时，WebSocket 握手失败日志经常很多。
其中有一类“客户端其实已经断开”的异常，本质上只是噪声，不值得把日志刷爆。

这个模块就是专门过滤这类噪声的。
"""

from __future__ import annotations

import logging

from websockets.exceptions import ConnectionClosed

OPENING_HANDSHAKE_FAILED_MESSAGE = "opening handshake failed"


def _exception_chain_has_disconnect(exc: BaseException | None) -> bool:
    """沿着异常链向下找，判断是否包含“对端已断开”类异常。"""
    seen: set[int] = set()
    while exc is not None:
        ident = id(exc)
        if ident in seen:
            return False
        seen.add(ident)
        if isinstance(exc, (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            ConnectionClosed,
        )):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class WebSocketHandshakeNoiseFilter(logging.Filter):
    """过滤浏览器已断开时产生的握手失败噪声日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """返回 `False` 表示这条日志应该被抑制。"""
        if record.getMessage() != OPENING_HANDSHAKE_FAILED_MESSAGE:
            return True
        exc_info = record.exc_info
        exc = exc_info[1] if isinstance(exc_info, tuple) and len(exc_info) >= 2 else None
        return not _exception_chain_has_disconnect(exc)


def websockets_server_logger() -> logging.Logger:
    """获取并配置 `websockets.server` logger。"""
    ws_logger = logging.getLogger("websockets.server")
    if not any(isinstance(f, WebSocketHandshakeNoiseFilter) for f in ws_logger.filters):
        ws_logger.addFilter(WebSocketHandshakeNoiseFilter())
    return ws_logger
