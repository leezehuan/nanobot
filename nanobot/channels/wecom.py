"""企业微信（WeCom）渠道实现：使用 ``wecom_aibot_sdk`` 接入 AI Bot。

这个文件展示了另一种常见的渠道接入方式：
- 不是直接 HTTP webhook
- 也不是自己手写协议
- 而是依赖第三方 SDK 提供的 WebSocket 长连接和事件回调

本文件的核心职责是把企业微信 SDK 的事件回调，转换成 nanobot 统一的消息流。
"""

import asyncio
import base64
import hashlib
import importlib.util
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base

WECOM_AVAILABLE = importlib.util.find_spec("wecom_aibot_sdk") is not None

# 上传安全大小限制，和 QQ 渠道保持相近默认值。
WECOM_UPLOAD_MAX_BYTES = 1024 * 1024 * 200  # 200MB

# 用于清洗文件名：把危险字符替换成 "_"，同时保留中文和常见安全标点。
_SAFE_NAME_RE = re.compile(r"[^\w.\-()\[\]（）【】\u4e00-\u9fff]+", re.UNICODE)


def _sanitize_filename(name: str) -> str:
    """清洗文件名，避免路径穿越和奇怪字符导致落盘失败。"""
    name = (name or "").strip()
    name = Path(name).name
    name = _SAFE_NAME_RE.sub("_", name).strip("._ ")
    return name


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov"}
_AUDIO_EXTS = {".amr", ".mp3", ".wav", ".ogg"}


def _guess_wecom_media_type(filename: str) -> str:
    """根据扩展名推断企业微信发送时要使用的媒体类型。"""
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "voice"
    return "file"

class WecomConfig(Base):
    """企业微信 AI Bot 渠道配置模型。"""

    enabled: bool = False
    bot_id: str = ""
    secret: str = ""
    allow_from: list[str] = Field(default_factory=list)
    welcome_message: str = ""


# 当 mixed 消息里出现我们不做深度解析的子类型时，用这些占位文本保持可读性。
MSG_TYPE_MAP = {
    "image": "[image]",
    "voice": "[voice]",
    "file": "[file]",
    "mixed": "[mixed content]",
}


class WecomChannel(BaseChannel):
    """基于企业微信 WebSocket 长连接的渠道适配器。

    【中文名称】企业微信渠道适配器

    【特点】
    - 不需要公网 webhook
    - 可接收文本、图片、语音、文件、mixed 混合消息
    - 支持基于原始 frame 的 reply / reply_stream 语义
    """

    name = "wecom"
    display_name = "WeCom"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WecomConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WecomConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WecomConfig = config
        self._client: Any = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generate_req_id = None
        # 保存每个 chat 最近一次的原始 frame，后续回复消息时需要它维持企业微信上下文。
        self._chat_frames: dict[str, Any] = {}

    async def start(self) -> None:
        """启动企业微信机器人并注册所有事件回调。"""
        if not WECOM_AVAILABLE:
            self.logger.error("SDK not installed. Run: pip install nanobot-ai[wecom]")
            return

        if not self.config.bot_id or not self.config.secret:
            self.logger.error("bot_id and secret not configured")
            return

        from wecom_aibot_sdk import WSClient, generate_req_id

        self._running = True
        self._loop = asyncio.get_running_loop()
        self._generate_req_id = generate_req_id

        # 创建 SDK WebSocket client；自动重连与心跳主要由 SDK 内部维护。
        self._client = WSClient({
            "bot_id": self.config.bot_id,
            "secret": self.config.secret,
            "reconnect_interval": 1000,
            "max_reconnect_attempts": -1,  # Infinite reconnect
            "heartbeat_interval": 30000,
        })

        # 把不同类型的企业微信事件分别绑定到对应处理函数。
        self._client.on("connected", self._on_connected)
        self._client.on("authenticated", self._on_authenticated)
        self._client.on("disconnected", self._on_disconnected)
        self._client.on("error", self._on_error)
        self._client.on("message.text", self._on_text_message)
        self._client.on("message.image", self._on_image_message)
        self._client.on("message.voice", self._on_voice_message)
        self._client.on("message.file", self._on_file_message)
        self._client.on("message.mixed", self._on_mixed_message)
        self._client.on("event.enter_chat", self._on_enter_chat)

        self.logger.info("bot starting with WebSocket long connection")
        self.logger.info("No public IP required - using WebSocket to receive events")

        # 真正建立 WebSocket 长连接。
        await self._client.connect_async()

        # 渠道对象保持存活，直到外部调用 stop()。
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """停止企业微信机器人并断开连接。"""
        self._running = False
        if self._client:
            await self._client.disconnect()
        self.logger.info("bot stopped")

    async def _on_connected(self, frame: Any) -> None:
        """处理 WebSocket 已连接事件。"""
        self.logger.info("WebSocket connected")

    async def _on_authenticated(self, frame: Any) -> None:
        """处理鉴权成功事件。"""
        self.logger.info("authenticated successfully")

    async def _on_disconnected(self, frame: Any) -> None:
        """处理连接断开事件。"""
        reason = frame.body if hasattr(frame, 'body') else str(frame)
        self.logger.warning("WebSocket disconnected: {}", reason)

    async def _on_error(self, frame: Any) -> None:
        """处理 SDK 层错误事件。"""
        self.logger.error("error: {}", frame)

    async def _on_text_message(self, frame: Any) -> None:
        """处理文本消息事件。"""
        await self._process_message(frame, "text")

    async def _on_image_message(self, frame: Any) -> None:
        """处理图片消息事件。"""
        await self._process_message(frame, "image")

    async def _on_voice_message(self, frame: Any) -> None:
        """处理语音消息事件。"""
        await self._process_message(frame, "voice")

    async def _on_file_message(self, frame: Any) -> None:
        """处理文件消息事件。"""
        await self._process_message(frame, "file")

    async def _on_mixed_message(self, frame: Any) -> None:
        """处理 mixed 混合消息事件。"""
        await self._process_message(frame, "mixed")

    async def _on_enter_chat(self, frame: Any) -> None:
        """处理用户打开机器人会话窗口事件。

        企业微信支持“进入会话”这一事件，这里可用于发送欢迎语。
        """
        try:
            # SDK 不同版本里 frame 可能是对象也可能是 dict，这里统一兼容读取 body。
            if hasattr(frame, 'body'):
                body = frame.body or {}
            elif isinstance(frame, dict):
                body = frame.get("body", frame)
            else:
                body = {}

            chat_id = body.get("chatid", "") if isinstance(body, dict) else ""

            if chat_id and not self.is_allowed(chat_id):
                return

            if chat_id and self.config.welcome_message:
                await self._client.reply_welcome(frame, {
                    "msgtype": "text",
                    "text": {"content": self.config.welcome_message},
                })
        except Exception:
            self.logger.exception("Error handling enter_chat")

    async def _process_message(self, frame: Any, msg_type: str) -> None:
        """解析企业微信入站消息，并转发到统一消息总线。

        这是本文件最重要的入站转换函数。它负责：
        1. 从 SDK frame 中提取 body
        2. 做权限判断和消息去重
        3. 按消息类型抽取文本与媒体
        4. 需要时下载企业微信加密附件
        5. 调用 ``_handle_message()`` 进入 nanobot 标准处理链
        """
        try:
            # 兼容 SDK 不同 frame 形态。
            if hasattr(frame, 'body'):
                body = frame.body or {}
            elif isinstance(frame, dict):
                body = frame.get("body", frame)
            else:
                body = {}

            # 后续逻辑默认 body 是 dict，这里先防御性检查。
            if not isinstance(body, dict):
                self.logger.warning("Invalid body type: {}", type(body))
                return

            # 若平台没有给标准 msgid，就退化成 chatid + sendertime 组合键。
            msg_id = body.get("msgid", "")
            if not msg_id:
                msg_id = f"{body.get('chatid', '')}_{body.get('sendertime', '')}"

            # 企业微信 SDK 会把发送者身份放在 from 字段里。
            from_info = body.get("from", {})
            sender_id = from_info.get("userid", "unknown") if isinstance(from_info, dict) else "unknown"
            if not self.is_allowed(sender_id):
                return

            # 通过一个小缓存做去重，避免平台重试导致同一条消息反复触发。
            if msg_id in self._processed_message_ids:
                return
            self._processed_message_ids[msg_id] = None

            # 控制缓存上限，避免常驻进程无限增长。
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            # 单聊里 chatid 往往就是发送者 userid；群聊则通常由 body 显式提供。
            chat_type = body.get("chattype", "single")
            chat_id = body.get("chatid", sender_id)

            content_parts = []
            media_paths: list[str] = []

            if msg_type == "text":
                text = body.get("text", {}).get("content", "")
                if text:
                    content_parts.append(text)

            elif msg_type == "image":
                image_info = body.get("image", {})
                file_url = image_info.get("url", "")
                aes_key = image_info.get("aeskey", "")

                if file_url and aes_key:
                    file_path = await self._download_and_save_media(file_url, aes_key, "image")
                    if file_path:
                        filename = os.path.basename(file_path)
                        content_parts.append(f"[image: {filename}]")
                        media_paths.append(file_path)
                    else:
                        content_parts.append("[image: download failed]")
                else:
                    content_parts.append("[image: download failed]")

            elif msg_type == "voice":
                voice_info = body.get("voice", {})
                # 企业微信语音消息通常已经带有平台侧转写文本。
                voice_content = voice_info.get("content", "")
                if voice_content:
                    content_parts.append(f"[voice] {voice_content}")
                else:
                    content_parts.append("[voice]")

            elif msg_type == "file":
                file_info = body.get("file", {})
                file_url = file_info.get("url", "")
                aes_key = file_info.get("aeskey", "")
                file_name = file_info.get("name") or None

                if file_url and aes_key:
                    file_path = await self._download_and_save_media(file_url, aes_key, "file", file_name)
                    if file_path:
                        display_name = os.path.basename(file_path)
                        content_parts.append(f"[file: {display_name}]")
                        media_paths.append(file_path)
                    else:
                        content_parts.append(f"[file: {file_name or 'unknown'}: download failed]")
                else:
                    content_parts.append(f"[file: {file_name or 'unknown'}: download failed]")

            elif msg_type == "mixed":
                # mixed 消息本质上是多个子消息片段的容器。
                msg_items = body.get("mixed", {}).get("msg_item", [])
                for item in msg_items:
                    item_type = item.get("msgtype", "")
                    if item_type == "text":
                        text = item.get("text", {}).get("content", "")
                        if text:
                            content_parts.append(text)
                    elif item_type == "image":
                        file_url = item.get("image", {}).get("url", "")
                        aes_key = item.get("image", {}).get("aeskey", "")
                        if file_url and aes_key:
                            file_path = await self._download_and_save_media(file_url, aes_key, "image")
                            if file_path:
                                filename = os.path.basename(file_path)
                                content_parts.append(f"[image: {filename}]")
                                media_paths.append(file_path)
                    else:
                        content_parts.append(MSG_TYPE_MAP.get(item_type, f"[{item_type}]"))

            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            content = "\n".join(content_parts) if content_parts else ""

            if not content:
                return

            # 保存最近原始 frame，后续 reply / reply_stream 发送时需要它。
            self._chat_frames[chat_id] = frame

            # 统一投递到 nanobot 的消息总线。
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media_paths or None,
                metadata={
                    "message_id": msg_id,
                    "msg_type": msg_type,
                    "chat_type": chat_type,
                }
            )

        except Exception:
            self.logger.exception("Error processing message")

    async def _download_and_save_media(
        self,
        file_url: str,
        aes_key: str,
        media_type: str,
        filename: str | None = None,
    ) -> str | None:
        """下载并保存企业微信媒体文件。

        企业微信媒体不是普通公开 URL，通常需要结合 ``aes_key`` 解密。
        SDK 已封装了下载与解密，本函数负责把结果落到本地媒体目录。
        """
        try:
            data, fname = await self._client.download_file(file_url, aes_key)

            if not data:
                self.logger.warning("Failed to download media")
                return None

            if len(data) > WECOM_UPLOAD_MAX_BYTES:
                self.logger.warning(
                    "inbound media too large: {} bytes (max {})",
                    len(data),
                    WECOM_UPLOAD_MAX_BYTES,
                )
                return None

            media_dir = get_media_dir("wecom")
            if not filename:
                filename = fname or f"{media_type}_{hash(file_url) % 100000}"
            filename = _sanitize_filename(filename)

            file_path = media_dir / filename
            await asyncio.to_thread(file_path.write_bytes, data)
            self.logger.debug("Downloaded {} to {}", media_type, file_path)
            return str(file_path)

        except Exception:
            self.logger.exception("Error downloading media")
            return None

    async def _upload_media_ws(
        self, client: Any, file_path: str,
    ) -> "tuple[str, str] | tuple[None, None]":
        """通过企业微信 WebSocket 三段式协议上传本地媒体。

        上传流程：
        1. ``aibot_upload_media_init``：申请 ``upload_id``
        2. ``aibot_upload_media_chunk``：按块上传内容
        3. ``aibot_upload_media_finish``：结束上传并换回 ``media_id``
        """
        from wecom_aibot_sdk.utils import generate_req_id as _gen_req_id

        try:
            fname = os.path.basename(file_path)
            media_type = _guess_wecom_media_type(fname)

            # 读文件属于阻塞 I/O，放到线程里执行，避免卡住 asyncio 事件循环。
            def _read_file():
                file_size = os.path.getsize(file_path)
                if file_size > WECOM_UPLOAD_MAX_BYTES:
                    raise ValueError(
                        f"File too large: {file_size} bytes (max {WECOM_UPLOAD_MAX_BYTES})"
                    )
                with open(file_path, "rb") as f:
                    return file_size, f.read()

            file_size, data = await asyncio.to_thread(_read_file)
            # 这里的 MD5 只是协议要求的完整性校验，并非安全场景哈希。
            md5_hash = hashlib.md5(data).hexdigest()

            chunk_size = 512 * 1024  # 512 KB raw (before base64)
            mv = memoryview(data)
            chunk_list = [bytes(mv[i : i + chunk_size]) for i in range(0, file_size, chunk_size)]
            n_chunks = len(chunk_list)
            del mv, data

            # 第 1 步：初始化上传，换回 upload_id。
            req_id = _gen_req_id("upload_init")
            resp = await client._ws_manager.send_reply(req_id, {
                "type": media_type,
                "filename": fname,
                "total_size": file_size,
                "total_chunks": n_chunks,
                "md5": md5_hash,
            }, "aibot_upload_media_init")
            if resp.errcode != 0:
                self.logger.warning("upload init failed ({}): {}", resp.errcode, resp.errmsg)
                return None, None
            upload_id = resp.body.get("upload_id") if resp.body else None
            if not upload_id:
                self.logger.warning("upload init: no upload_id in response")
                return None, None

            # 第 2 步：按顺序发送所有分片。
            for i, chunk in enumerate(chunk_list):
                req_id = _gen_req_id("upload_chunk")
                resp = await client._ws_manager.send_reply(req_id, {
                    "upload_id": upload_id,
                    "chunk_index": i,
                    "base64_data": base64.b64encode(chunk).decode(),
                }, "aibot_upload_media_chunk")
                if resp.errcode != 0:
                    self.logger.warning("upload chunk {} failed ({}): {}", i, resp.errcode, resp.errmsg)
                    return None, None

            # 第 3 步：提交完成，换回最终 media_id。
            req_id = _gen_req_id("upload_finish")
            resp = await client._ws_manager.send_reply(req_id, {
                "upload_id": upload_id,
            }, "aibot_upload_media_finish")
            if resp.errcode != 0:
                self.logger.warning("upload finish failed ({}): {}", resp.errcode, resp.errmsg)
                return None, None

            media_id = resp.body.get("media_id") if resp.body else None
            if not media_id:
                self.logger.warning("upload finish: no media_id in response body={}", resp.body)
                return None, None

            suffix = "..." if len(media_id) > 16 else ""
            self.logger.debug("uploaded {} ({}) → media_id={}", fname, media_type, media_id[:16] + suffix)
            return media_id, media_type

        except ValueError as e:
            self.logger.warning("upload skipped for {}: {}", file_path, e)
            return None, None
        except Exception:
            self.logger.exception("_upload_media_ws error for {}", file_path)
            return None, None

    async def send(self, msg: OutboundMessage) -> None:
        """向企业微信发送文本和媒体。

        企业微信有两种常见发送语境：
        - 有原始 frame：表示在回复某条已有消息
        - 没有原始 frame：表示主动推送
        两者使用的 SDK 接口略有区别。
        """
        if not self._client:
            self.logger.warning("client not initialized")
            return

        try:
            content = (msg.content or "").strip()
            is_progress = bool(msg.metadata.get("_progress"))

            # 拿到该 chat 最近一次原始 frame，reply_stream 发送会依赖它。
            frame = self._chat_frames.get(msg.chat_id)

            # 媒体文件需要先上传为 media_id，随后再引用 media_id 发送。
            for file_path in msg.media or []:
                if not os.path.isfile(file_path):
                    self.logger.warning("media file not found: {}", file_path)
                    continue
                media_id, media_type = await self._upload_media_ws(self._client, file_path)
                if media_id:
                    if frame:
                        await self._client.reply(frame, {
                            "msgtype": media_type,
                            media_type: {"media_id": media_id},
                        })
                    else:
                        await self._client.send_message(msg.chat_id, {
                            "msgtype": media_type,
                            media_type: {"media_id": media_id},
                        })
                    self.logger.debug("sent {} → {}", media_type, msg.chat_id)
                else:
                    content += f"\n[file upload failed: {os.path.basename(file_path)}]"

            if not content:
                return

            if frame:
                # 不论进度消息还是最终消息，都统一用 reply_stream。
                # 普通 reply() 在发送 text 时会触发企业微信侧错误。
                stream_id = self._generate_req_id("stream")
                await self._client.reply_stream(
                    frame,
                    stream_id,
                    content,
                    finish=not is_progress,
                )
                self.logger.debug(
                    "{} sent to {}",
                    "progress" if is_progress else "message",
                    msg.chat_id,
                )
            else:
                # 没有 frame 时，多半是主动推送，例如 cron 消息，只能走主动发送接口。
                await self._client.send_message(msg.chat_id, {
                    "msgtype": "markdown",
                    "markdown": {"content": content},
                })
                self.logger.info("proactive send to {}", msg.chat_id)

        except Exception:
            self.logger.exception("Error sending message to chat_id={}", msg.chat_id)
