"""WhatsApp 渠道实现：通过 Node.js bridge 接入 WhatsApp Web。

这个文件的定位是“平台适配器”：

1. Python 主进程并不直接实现 WhatsApp Web 协议
2. 复杂协议细节交给 Node.js bridge 处理
3. 本文件只负责在 bridge 消息格式 与 nanobot 统一消息格式之间做转换

对 Agent 初学者来说，这类文件很重要，因为它展示了：
- 外部聊天平台如何接入消息总线
- 多语言 bridge（Python + Node.js）如何协作
- 渠道层如何把平台特定字段整理成统一的 ``InboundMessage`` / ``OutboundMessage``
"""

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class WhatsAppConfig(Base):
    """WhatsApp 渠道配置模型。

    这是“静态配置”而不是“运行时状态”：
    - ``bridge_url``：Python 连接 bridge 的地址
    - ``bridge_token``：Python 与 bridge 之间共享的鉴权密钥
    - ``group_policy``：群聊里是全量响应，还是仅在被 @ 时响应
    """

    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    bridge_token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "open"  # "open" responds to all, "mention" only when @mentioned


def _bridge_token_path() -> Path:
    """返回 bridge token 的本地保存路径。

    当用户没有手工配置 ``bridge_token`` 时，nanobot 会在本地自动生成一个共享密钥，
    Python 和 Node.js bridge 后续都使用它进行简单鉴权。
    """
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("whatsapp-auth") / "bridge-token"


def _load_or_create_bridge_token(path: Path) -> str:
    """读取已保存的 bridge token；若不存在则首次创建。

    这样可以保证：
    - 首次运行时开箱即用
    - 后续重启仍能复用同一个 bridge 身份
    """
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    with suppress(OSError):
        path.chmod(0o600)
    return token


class WhatsAppChannel(BaseChannel):
    """连接 Node.js bridge 的 WhatsApp 渠道适配器。

    bridge 侧使用 ``@whiskeysockets/baileys`` 处理 WhatsApp Web 协议，
    本类负责：

    - 建立和 bridge 的 WebSocket 连接
    - 把 bridge 推来的事件翻译成统一消息
    - 把 Agent 回复转成 bridge 可识别的发送命令
    """

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WhatsAppConfig.model_validate(config)
        super().__init__(config, bus)
        self._ws = None
        self._connected = False
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._lid_to_phone: dict[str, str] = {}
        self._bridge_token: str | None = None

    def _effective_bridge_token(self) -> str:
        """得到当前应使用的 bridge token。

        读取顺序是：
        1. 进程内缓存
        2. 配置文件里显式提供的 token
        3. 本地自动生成并持久化的 token
        """
        if self._bridge_token is not None:
            return self._bridge_token
        configured = self.config.bridge_token.strip()
        if configured:
            self._bridge_token = configured
        else:
            self._bridge_token = _load_or_create_bridge_token(_bridge_token_path())
        return self._bridge_token

    async def login(self, force: bool = False) -> bool:
        """启动 bridge，并进入二维码登录流程。

        这里的“登录”不是 Python 直接请求 WhatsApp，而是拉起 Node.js bridge，
        由 bridge 在终端中展示二维码并等待用户扫码。
        """
        try:
            bridge_dir = _ensure_bridge_setup()
        except RuntimeError:
            self.logger.exception("bridge setup failed")
            return False

        env = {**os.environ}
        env["BRIDGE_TOKEN"] = self._effective_bridge_token()
        env["AUTH_DIR"] = str(_bridge_token_path().parent)

        self.logger.info("Starting WhatsApp bridge for QR login...")
        try:
            subprocess.run(
                [shutil.which("npm"), "start"], cwd=bridge_dir, check=True, env=env
            )
        except subprocess.CalledProcessError:
            return False

        return True

    async def start(self) -> None:
        """启动渠道并持续连接 bridge。

        这是运行期主循环：
        - 建立 WebSocket 连接
        - 先发送鉴权消息
        - 持续监听 bridge 事件
        - 出错后自动重连
        """
        import websockets

        bridge_url = self.config.bridge_url

        self.logger.info("Connecting to WhatsApp bridge at {}...", bridge_url)

        self._running = True

        while self._running:
            try:
                async with websockets.connect(bridge_url) as ws:
                    self._ws = ws
                    await ws.send(
                        json.dumps({"type": "auth", "token": self._effective_bridge_token()})
                    )
                    self._connected = True
                    self.logger.info("Connected to WhatsApp bridge")

                    # 持续接收入站事件，bridge 会把 WhatsApp 平台消息编码成 JSON 推给我们。
                    async for message in ws:
                        try:
                            await self._handle_bridge_message(message)
                        except Exception:
                            self.logger.exception("Error handling bridge message")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                self.logger.warning("WhatsApp bridge connection error: {}", e)

                if self._running:
                    self.logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """停止渠道并关闭当前 WebSocket 连接。"""
        self._running = False
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send(self, msg: OutboundMessage) -> None:
        """通过 bridge 向 WhatsApp 发送文本或媒体。"""
        if not self._ws or not self._connected:
            self.logger.warning("WhatsApp bridge not connected")
            return

        chat_id = msg.chat_id

        if msg.content:
            try:
                payload = {"type": "send", "to": chat_id, "text": msg.content}
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception:
                self.logger.exception("Error sending message")
                raise

        for media_path in msg.media or []:
            try:
                mime, _ = mimetypes.guess_type(media_path)
                payload = {
                    "type": "send_media",
                    "to": chat_id,
                    "filePath": media_path,
                    "mimetype": mime or "application/octet-stream",
                    "fileName": media_path.rsplit("/", 1)[-1],
                }
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception:
                self.logger.exception("Error sending media {}", media_path)
                raise

    async def _handle_bridge_message(self, raw: str) -> None:
        """处理 bridge 发来的一条原始 JSON 事件。

        这是本文件最关键的入站转换函数。它会：
        1. 解析 bridge payload
        2. 区分消息/状态/二维码/错误事件
        3. 从 WhatsApp 的多种身份字段中推导统一 ``sender_id``
        4. 补充媒体标签与语音转写
        5. 调用 ``_handle_message()`` 投递到消息总线
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Invalid JSON from bridge: {}", raw[:100])
            return

        msg_type = data.get("type")

        if msg_type == "message":
            # 入站消息。
            # pn 是较旧的手机号风格身份，典型形式：<phone>@s.whatsapp.net
            pn = data.get("pn", "")
            # sender 往往是较新的 LID 风格身份。
            sender = data.get("sender", "")
            content = data.get("content", "")
            message_id = data.get("id", "")

            # 群聊是否需要仅在被 @ 时才响应，由配置控制。
            is_group = data.get("isGroup", False)
            was_mentioned = bool(data.get("wasMentioned", False) or data.get("isReplyToBot", False))

            if is_group and getattr(self.config, "group_policy", "open") == "mention":
                if not was_mentioned:
                    return

            # 通过 JID 后缀识别“手机号身份”和“LID 身份”。
            # bridge 不同版本的字段映射不完全稳定，所以这里做兼容推断。
            raw_a = pn or ""
            participant = data.get("participant", "")
            raw_b = participant or sender or ""
            id_a = raw_a.split("@")[0] if "@" in raw_a else raw_a
            id_b = raw_b.split("@")[0] if "@" in raw_b else raw_b

            phone_id = ""
            lid_id = ""
            for raw, extracted in [(raw_a, id_a), (raw_b, id_b)]:
                if "@s.whatsapp.net" in raw:
                    phone_id = extracted
                elif "@lid.whatsapp.net" in raw:
                    lid_id = extracted
                elif extracted and not phone_id:
                    phone_id = extracted  # 对无后缀裸值做保守兜底

            sender_id = phone_id or self._lid_to_phone.get(lid_id, "") or lid_id or id_a or id_b
            if not self.is_allowed(sender_id):
                return

            if message_id:
                if message_id in self._processed_message_ids:
                    return
                # 用一个小型有序缓存做去重，避免 bridge 重发导致 Agent 重复执行。
                self._processed_message_ids[message_id] = None
                while len(self._processed_message_ids) > 1000:
                    self._processed_message_ids.popitem(last=False)

            if phone_id and lid_id:
                # 记住 LID -> 手机号映射，后续只拿到 LID 时仍可做权限判断。
                self._lid_to_phone[lid_id] = phone_id

            self.logger.info("Sender phone={} lid={} → sender_id={}", phone_id or "(empty)", lid_id or "(empty)", sender_id)

            # bridge 侧已下载好的媒体路径会直接附在消息里。
            media_paths = data.get("media") or []

            # 尽量把语音消息统一转成文本，这样后续 Agent 不用区分文本输入与语音输入。
            if content == "[Voice Message]":
                if media_paths:
                    self.logger.info("Transcribing voice message from {}...", sender_id)
                    transcription = await self.transcribe_audio(media_paths[0])
                    if transcription:
                        content = transcription
                        media_paths = []
                        self.logger.info("Transcribed voice from {}: {}...", sender_id, transcription[:50])
                    else:
                        content = "[Voice Message: Transcription failed]"
                else:
                    content = "[Voice Message: Audio not available]"

            # 把媒体附加成统一标签，保持和其他渠道相似的上下文格式。
            if media_paths:
                for p in media_paths:
                    mime, _ = mimetypes.guess_type(p)
                    media_type = "image" if mime and mime.startswith("image/") else "file"
                    media_tag = f"[{media_type}: {p}]"
                    content = f"{content}\n{media_tag}" if content else media_tag

            await self._handle_message(
                sender_id=sender_id,
                chat_id=sender,  # 回复时尽量沿用平台原生会话标识，避免只用手机号导致定位不准
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "timestamp": data.get("timestamp"),
                    "is_group": data.get("isGroup", False),
                    "participant": participant or None,
                    "is_reply_to_bot": data.get("isReplyToBot", False),
                },
            )

        elif msg_type == "status":
            # bridge 连接状态变化事件。
            status = data.get("status")
            self.logger.info("Status: {}", status)

            if status == "connected":
                self._connected = True
            elif status == "disconnected":
                self._connected = False

        elif msg_type == "qr":
            # 二维码本身通常在 bridge 终端里显示，这里只提示用户去扫码。
            self.logger.info("Scan QR code in the bridge terminal to connect WhatsApp")

        elif msg_type == "error":
            self.logger.error("Bridge error: {}", data.get("error"))


def _ensure_bridge_setup() -> Path:
    """确保 WhatsApp bridge 已复制、安装依赖并构建完成。

    这一步处理的是“bridge 运行环境准备”，不是“连接 WhatsApp”本身。
    它会：
    1. 定位 bridge 源码目录
    2. 计算源码哈希，判断本地缓存是否过期
    3. 需要时重新复制 bridge、执行 ``npm install`` 和 ``npm run build``
    """
    from nanobot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()
    stamp_file = user_bridge / ".nanobot-bridge-source-hash"

    # 优先使用打包进安装产物里的 bridge；若不存在，则回退到源码仓库目录。
    current_file = Path(__file__)
    pkg_bridge = current_file.parent.parent / "bridge"
    src_bridge = current_file.parent.parent.parent / "bridge"

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        raise RuntimeError(
            "WhatsApp bridge source not found. "
            "Try reinstalling: pip install --force-reinstall nanobot"
        )

    def source_hash(root: Path) -> str:
        """计算 bridge 源目录内容哈希，用于判断是否需要重建。"""
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if rel.parts and rel.parts[0] in {"node_modules", "dist"}:
                continue
            digest.update(rel.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    expected_hash = source_hash(source)
    current_hash = stamp_file.read_text().strip() if stamp_file.exists() else None

    # dist 存在且源码哈希一致，说明 bridge 已是最新构建结果，可直接复用。
    if (user_bridge / "dist" / "index.js").exists() and current_hash == expected_hash:
        return user_bridge

    if (user_bridge / "dist" / "index.js").exists() and current_hash != expected_hash:
        logger.info("WhatsApp bridge source changed; rebuilding bridge...")

    npm_path = shutil.which("npm")
    if not npm_path:
        raise RuntimeError("npm not found. Please install Node.js >= 18.")

    logger.info("Setting up WhatsApp bridge...")
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    logger.info("  Installing dependencies...")
    subprocess.run([npm_path, "install"], cwd=user_bridge, check=True, capture_output=True)

    logger.info("  Building...")
    subprocess.run([npm_path, "run", "build"], cwd=user_bridge, check=True, capture_output=True)
    stamp_file.write_text(expected_hash + "\n")

    logger.info("Bridge ready")
    return user_bridge
