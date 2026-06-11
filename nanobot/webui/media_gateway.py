"""WebUI 媒体网关服务。

【中文名称】媒体签名与改写服务

WebUI 里经常会遇到本地图片、上传媒体、会话附件这类资源。
这些资源不能简单把绝对路径暴露给前端，所以需要一层专门的媒体网关来做：

- 媒体路径签名
- 本地 markdown 图片地址改写
- transcript 里的附件增强
- 安全的媒体回放入口
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.config.paths import get_media_dir
from nanobot.webui.media_api import (
    attach_signed_media_urls,
    serve_signed_media,
    sign_media_path,
    sign_or_stage_media_path,
    signed_media_attachments,
)
from nanobot.webui.transcript import rewrite_local_markdown_images


class WebUIMediaGateway:
    """负责媒体签名、媒体 URL 增强和 markdown 图片改写。"""

    def __init__(
        self,
        *,
        workspace_path: Path,
        logger: Any,
        media_dir: Callable[[str | None], Path] | None = None,
        secret: bytes | None = None,
    ) -> None:
        """初始化媒体网关。

        `secret` 用于生成签名 URL；如果没有传入，就在当前进程里随机生成。
        """
        self.workspace_path = workspace_path
        self.logger = logger
        self._media_dir = media_dir or (lambda channel=None: get_media_dir(channel))
        self.secret = secret or secrets.token_bytes(32)

    def serve_signed_media(
        self,
        sig: str,
        payload: str,
        *,
        request: WsRequest | None = None,
    ) -> Response:
        """根据签名信息回放媒体文件。"""
        return serve_signed_media(
            sig,
            payload,
            secret=self.secret,
            request=request,
            media_dir=self._media_dir,
        )

    def sign_media_path(self, abs_path: Path) -> str | None:
        """为一个绝对路径媒体文件生成可安全访问的签名 URL。"""
        return sign_media_path(
            abs_path,
            secret=self.secret,
            media_dir=self._media_dir,
        )

    def sign_or_stage_media_path(self, path: Path) -> dict[str, str] | None:
        """为媒体生成签名地址；必要时先做 staging。"""
        return sign_or_stage_media_path(
            path,
            secret=self.secret,
            media_dir=self._media_dir,
            logger=self.logger,
        )

    def rewrite_local_markdown_images(
        self,
        text: str,
        *,
        workspace_path: Path | None = None,
    ) -> str:
        """把 markdown 文本里的本地图片引用改写成可由 WebUI 访问的地址。"""
        return rewrite_local_markdown_images(
            text,
            workspace_path=workspace_path or self.workspace_path,
            sign_path=self.sign_or_stage_media_path,
        )

    def augment_media_urls(self, payload: dict[str, Any]) -> None:
        """给 payload 中的媒体字段附加签名 URL。"""
        attach_signed_media_urls(payload, sign_path=self.sign_media_path)

    def augment_transcript_media(self, paths: list[str]) -> list[dict[str, Any]]:
        """把 transcript 中的媒体路径转成前端可直接展示的附件对象。"""
        return signed_media_attachments(
            paths,
            sign_path=self.sign_or_stage_media_path,
        )

    def augment_transcript_user_media(self, paths: list[str]) -> list[dict[str, Any]]:
        """当前用户媒体和 transcript 媒体走同一套增强逻辑。"""
        return self.augment_transcript_media(paths)
