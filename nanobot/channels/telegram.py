"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReactionTypeEmoji,
    ReplyParameters,
    Update,
)
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.command.builtin import build_help_text
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.security.network import validate_url_target
from nanobot.utils.helpers import split_message

TELEGRAM_MAX_MESSAGE_LEN = 4000  # Telegram message character limit
# Telegram's actual API limit is 4096; we split raw markdown at 4000 as a
# safety margin for mid-stream edits (plain text).  For _stream_end, we
# convert to HTML first and then split at the true 4096-char boundary so
# the final rendered message never overflows.
TELEGRAM_HTML_MAX_LEN = 4096
TELEGRAM_REPLY_CONTEXT_MAX_LEN = TELEGRAM_MAX_MESSAGE_LEN  # Max length for reply context in user message


def _escape_telegram_html(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tool_hint_to_telegram_blockquote(text: str) -> str:
    """Render tool hints as an expandable blockquote (collapsed by default)."""
    return f"<blockquote expandable>{_escape_telegram_html(text)}</blockquote>" if text else ""


def _strip_md(s: str) -> str:
    """Strip markdown inline formatting from text."""
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'__(.+?)__', r'\1', s)
    s = re.sub(r'~~(.+?)~~', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    return s.strip()


def _strip_md_block(text: str) -> str:
    """Strip block-level and inline markdown for readable plain-text preview.

    Used during streaming mid-edits so users see clean text instead of raw
    markdown syntax while the response is still being generated.
    """
    # Code blocks -> just the code
    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', r'\1', text)
    # Headers -> plain text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)
    # Bold / italic / strikethrough
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Bullet lists
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    # Numbered lists (normalize spacing)
    text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
    return text


def _render_table_box(table_lines: list[str]) -> str:
    """Convert markdown pipe-table to compact aligned text for <pre> display."""

    def dw(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

    rows: list[list[str]] = []
    has_sep = False
    for line in table_lines:
        cells = [_strip_md(c) for c in line.strip().strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            has_sep = True
            continue
        rows.append(cells)
    if not rows or not has_sep:
        return '\n'.join(table_lines)

    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([''] * (ncols - len(r)))
    widths = [max(dw(r[c]) for r in rows) for c in range(ncols)]

    def dr(cells: list[str]) -> str:
        return '  '.join(f'{c}{" " * (w - dw(c))}' for c, w in zip(cells, widths))

    out = [dr(rows[0])]
    out.append('  '.join('─' * w for w in widths))
    for row in rows[1:]:
        out.append(dr(row))
    return '\n'.join(out)


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', save_code_block, text)

    # 1.5. Convert markdown tables to box-drawing (reuse code_block placeholders)
    lines = text.split('\n')
    rebuilt: list[str] = []
    li = 0
    while li < len(lines):
        if re.match(r'^\s*\|.+\|', lines[li]):
            tbl: list[str] = []
            while li < len(lines) and re.match(r'^\s*\|.+\|', lines[li]):
                tbl.append(lines[li])
                li += 1
            box = _render_table_box(tbl)
            if box != '\n'.join(tbl):
                code_blocks.append(box)
                rebuilt.append(f"\x00CB{len(code_blocks) - 1}\x00")
            else:
                rebuilt.extend(tbl)
        else:
            rebuilt.append(lines[li])
            li += 1
    text = '\n'.join(rebuilt)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r'`([^`]+)`', save_inline_code, text)

    # 3. Headers # Title -> <b>Title</b> (preserve visual hierarchy)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'⟪B⟫\1⟪/B⟫', text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = _escape_telegram_html(text)

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)

    # 10.5. Numbered lists  1. item -> 1. item (keep number, normalize indent)
    text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = _escape_telegram_html(code)
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = _escape_telegram_html(code)
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    # 13. Restore header bold markers (inserted in step 3, after HTML escaping)
    text = text.replace('⟪B⟫', '<b>').replace('⟪/B⟫', '</b>')

    return text


_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE_DELAY = 0.5  # seconds, doubled each retry
_STREAM_EDIT_INTERVAL_DEFAULT = 0.6  # min seconds between edit_message_text calls


@dataclass
class _StreamBuf:
    """Per-chat streaming accumulator for progressive message editing."""
    text: str = ""
    message_id: int | None = None
    last_edit: float = 0.0
    stream_id: str | None = None


@dataclass
class _QueuedTelegramUpdate:
    """Telegram update staged for per-session ordered processing."""

    kind: Literal["command", "message"]
    update: Update
    context: Any
    sort_key: tuple[int, int]


class TelegramConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    mode: Literal["polling", "webhook"] = "polling"
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None
    reply_to_message: bool = False
    react_emoji: str = "👀"
    group_policy: Literal["open", "mention"] = "mention"
    connection_pool_size: int = 32
    pool_timeout: float = 5.0
    streaming: bool = True
    # Enable inline keyboard buttons in Telegram messages.
    inline_keyboards: bool = False
    stream_edit_interval: float = Field(default=_STREAM_EDIT_INTERVAL_DEFAULT, ge=0.1)
    webhook_url: str = ""
    webhook_listen_host: str = "127.0.0.1"
    webhook_listen_port: int = Field(default=8081, ge=1, le=65535)
    webhook_path: str = "/telegram"
    webhook_secret_token: str = ""
    webhook_max_connections: int = Field(default=4, ge=1, le=100)

    @field_validator("webhook_path")
    @classmethod
    def webhook_path_must_start_with_slash(cls, value: str) -> str:
        value = value.strip() or "/telegram"
        if not value.startswith("/"):
            raise ValueError('webhook_path must start with "/"')
        return value

    @model_validator(mode="after")
    def validate_webhook_config(self) -> "TelegramConfig":
        if self.mode != "webhook":
            return self

        url = self.webhook_url.strip()
        if not url:
            raise ValueError("webhook_url is required when Telegram mode is webhook")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("webhook_url must be a public HTTPS URL")
        secret = self.webhook_secret_token.strip()
        if not secret:
            raise ValueError("webhook_secret_token is required when Telegram mode is webhook")
        if len(secret) > 256 or re.match(r"^[A-Za-z0-9_-]+$", secret) is None:
            raise ValueError(
                "webhook_secret_token must be 1-256 characters using only A-Z, a-z, 0-9, _ and -"
            )
        return self


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling or webhook mode.

    【中文名称】Telegram 渠道适配器

    【功能说明】
    将 nanobot 的 MessageBus 与 Telegram Bot API 桥接起来的完整适配器。

    【支持的运行模式】
    - polling 模式（默认）：bot 定期向 Telegram 服务器拉取新消息，适合开发环境
    - webhook 模式：Telegram 主动推送消息到 nanobot 的 HTTP 端点，适合生产环境

    【入站消息处理流程】
    1. python-telegram-bot 的 Handler 捕获消息 / 命令
    2. 消息/命令进入有序缓冲队列（_enqueue_ordered_update），按 message_id 排序后序列处理，保证同一会话内消息顺序不乱
    3. 图片/视频/语音/文档等媒体文件下载到本地 media 目录
    4. 语音消息自动转写为文本（transcribe_audio）
    5. 最终通过 self._handle_message() 发送到 MessageBus，进入 AgentLoop

    【出站消息处理流程】
    1. MessageBus 输出 OutboundMessage 到达 send() 方法
    2. Markdown 文本转换为 Telegram HTML 格式（_markdown_to_telegram_html）
    3. 支持流式消息编辑（send_delta），让用户在 LLM 生成过程中看到实时预览
    4. 超长消息自动分片（split_message），处理 4000 字符限制

    【关键内部状态】
    - _stream_bufs: 按 chat_id 存储流式缓冲区，支持渐近式消息编辑
    - _inbound_buffers/_inbound_workers: 有序入站队列，保证消息时序
    - _media_group_buffers: 媒体组缓冲（0.6 秒等待窗口），多张图合并为一次 turn
    - _message_threads: 论坛话题 ID 缓存，确保回复在正确的 thread 内

    Long polling is the default. Webhook mode requires a public HTTPS URL and a
    Telegram secret token.
    """

    name = "telegram"
    display_name = "Telegram"

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("restart", "Restart the bot"),
        BotCommand("status", "Show bot status"),
        BotCommand("history", "Show recent conversation messages"),
        BotCommand("goal", "Start a sustained objective (long-running task)"),
        BotCommand("pairing", "Manage DM pairing (approve/deny/list)"),
        BotCommand("model", "Switch runtime model preset"),
        BotCommand("skill", "List enabled skills"),
        BotCommand("dream", "Run Dream memory consolidation now"),
        BotCommand("dream_log", "Show the latest Dream memory change"),
        BotCommand("dream_restore", "Restore Dream memory to an earlier version"),
        BotCommand("help", "Show available commands"),
    ]

    # Regex for slash commands routed to AgentLoop via ``_forward_command``.
    # Hyphenated ``dream-*`` commands stay on a separate handler (below).
    TELEGRAM_BUS_SLASH_COMMAND_RE = re.compile(
        r"^/(?:new|stop|restart|status|dream|history|goal|pairing|model|skill)(?:@\w+)?(?:\s+.*)?$"
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return TelegramConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = TelegramConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
        self._media_group_buffers: dict[str, dict] = {}
        self._media_group_tasks: dict[str, asyncio.Task] = {}
        self._message_threads: dict[tuple[str, int], int] = {}
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None
        self._stream_bufs: dict[str, _StreamBuf] = {}  # chat_id -> streaming state
        self._inbound_buffers: dict[str, list[_QueuedTelegramUpdate]] = {}
        self._inbound_workers: dict[str, asyncio.Task] = {}

    def is_allowed(self, sender_id: str) -> bool:
        """Preserve Telegram's legacy id|username allowlist matching."""
        if super().is_allowed(sender_id):
            return True

        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list or "*" in allow_list:
            return False

        sender_str = str(sender_id)
        if sender_str.count("|") != 1:
            return False

        sid, username = sender_str.split("|", 1)
        if not sid.isdigit() or not username:
            return False

        return sid in allow_list or username in allow_list

    @staticmethod
    def _normalize_telegram_command(content: str) -> str:
        """Map Telegram-safe command aliases back to canonical nanobot commands."""
        if not content.startswith("/"):
            return content
        if content == "/dream_log" or content.startswith("/dream_log "):
            return content.replace("/dream_log", "/dream-log", 1)
        if content == "/dream_restore" or content.startswith("/dream_restore "):
            return content.replace("/dream_restore", "/dream-restore", 1)
        return content

    async def start(self) -> None:
        """启动 Telegram Bot，进入消息接收循环。

        【中文名称】启动 Telegram Bot

        【功能说明】
        这是 Telegram 渠道的完整生命周期入口，执行以下步骤：

        【启动流程 —— 6 个阶段】
        Phase 1 —— 创建 HTTPX 请求池：
          - api_request 池：处理出站消息发送（connection_pool_size 可配置）
          - poll_request 池：独立的长轮询连接（固定 4 连接），避免出站消息占用轮询通道

        Phase 2 —— 注册消息处理器：
          - /start 命令 → _on_start（打招呼）
          - 业务命令（/new /stop /status /history 等）→ _forward_command（转发到 AgentLoop）
          - /help 命令 → _on_help
          - 普通消息（文本/图片/视频/语音/文档/位置）→ _on_message
          - 可选：inline keyboard 回调 → _on_callback_query

        Phase 3 —— 初始化并获取 Bot 身份：
          - 调用 get_me() 获取 bot 的 username 和 user_id
          - 注册 Telegram 命令菜单（set_my_commands）

        Phase 4 —— 启动更新接收：
          - polling 模式：start_polling，进入长轮询循环
          - webhook 模式：start_webhook，在本地端口上监听 Telegram 推送

        Phase 5 —— 运行循环：只要 self._running 为 True，就保持 asyncio.sleep(1) 循环

        【两种模式的网络架构差异】
        polling:  nanobot → [主动拉取] → Telegram 服务器（每几秒一次）
        webhook:  Telegram 服务器 → [主动推送] → nanobot HTTP 端点

        【参数说明】
        无显式参数，所有配置从 self.config (TelegramConfig) 读取。
        """
        if not self.config.token:
            self.logger.error("bot token not configured")
            return

        self._running = True

        proxy = self.config.proxy or None

        # Separate pools so long-polling (getUpdates) never starves outbound sends.
        api_request = HTTPXRequest(
            connection_pool_size=self.config.connection_pool_size,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        poll_request = HTTPXRequest(
            connection_pool_size=4,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        builder = (
            Application.builder()
            .token(self.config.token)
            .request(api_request)
            .get_updates_request(poll_request)
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers (using Regex to support @username suffixes before bot initialization)
        self._app.add_handler(MessageHandler(filters.Regex(r"^/start(?:@\w+)?$"), self._on_start))
        self._app.add_handler(
            MessageHandler(
                filters.Regex(TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE),
                self._forward_command,
            )
        )
        self._app.add_handler(
            MessageHandler(
                filters.Regex(r"^/(dream-log|dream_log|dream-restore|dream_restore)(?:@\w+)?(?:\s+.*)?$"),
                self._forward_command,
            )
        )
        self._app.add_handler(MessageHandler(filters.Regex(r"^/help(?:@\w+)?$"), self._on_help))

        # Add message handler for text, photos, video, voice, documents, and locations
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.VIDEO_NOTE
                 | filters.ANIMATION | filters.VOICE | filters.AUDIO
                 | filters.Document.ALL | filters.LOCATION)
                & ~filters.COMMAND,
                self._on_message
            )
        )

        # Conditionally register inline keyboard callback handler
        if self.config.inline_keyboards:
            self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
            allowed_updates = ["message", "callback_query"]
            self.logger.debug("inline keyboards enabled")
        else:
            allowed_updates = ["message"]

        if self.config.mode == "webhook":
            self.logger.info("Starting bot (webhook mode)...")
        else:
            self.logger.info("Starting bot (polling mode)...")

        # Initialize and start receiving updates
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        self.logger.info("bot @{} connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            self.logger.debug("bot commands registered")
        except Exception as e:
            self.logger.warning("Failed to register bot commands: {}", e)

        if self.config.mode == "webhook":
            # ``url_path`` is the local HTTP route. ``webhook_url`` is the
            # public HTTPS URL Telegram calls; reverse proxies may rewrite it.
            await self._app.updater.start_webhook(
                listen=self.config.webhook_listen_host,
                port=self.config.webhook_listen_port,
                url_path=self.config.webhook_path.lstrip("/"),
                webhook_url=self.config.webhook_url.strip(),
                allowed_updates=allowed_updates,
                drop_pending_updates=False,
                secret_token=self.config.webhook_secret_token.strip(),
                max_connections=self.config.webhook_max_connections,
            )
        else:
            # Start polling (this runs until stopped)
            await self._app.updater.start_polling(
                allowed_updates=allowed_updates,
                drop_pending_updates=False,  # Process pending messages on startup
                error_callback=self._on_polling_error,
            )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """停止 Telegram Bot，清理所有运行中任务。

        【中文名称】停止 Telegram Bot

        【功能说明】
        优雅关闭 Telegram 渠道的所有活动资源：

        1. 取消所有 typing indicator 任务（_typing_tasks）
        2. 取消所有媒体组缓冲任务（_media_group_tasks）
        3. 取消所有入站消息工作协程（_inbound_workers）
        4. 关闭 polling/webhook updater
        5. 停止并关闭 Application，释放 HTTPX 连接池

        【注意】
        该方法会被 ChannelManager 在 nanobot 关闭时调用。
        停止后 _app 会被置为 None，新消息将无法发送。
        """
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()

        for task in self._inbound_workers.values():
            task.cancel()
        self._inbound_workers.clear()
        self._inbound_buffers.clear()

        if self._app:
            self.logger.info("Stopping bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext in ("mp4", "mov", "avi", "mkv", "webm", "3gp"):
            return "video"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    @staticmethod
    def _is_remote_media_url(path: str) -> bool:
        return path.startswith(("http://", "https://"))

    async def send(self, msg: OutboundMessage) -> None:
        """通过 Telegram 发送一条完整消息。

        【中文名称】发送出站消息

        【功能说明】
        这是 ChannelManager 调用 send() 分发 OutboundMessage 到 Telegram 的入口。
        一次完整的 send 调用执行以下步骤：

        【发送流程 —— 4 个阶段】
        Phase 1 —— 清理状态：
          - 如果不是进度中间态（_progress 为 False），停止 typing indicator
          - 移除消息上的 emoji 反应标记

        Phase 2 —— 发送媒体文件：
          - 遍历 msg.media 列表，根据文件扩展名判断类型（photo/video/voice/audio/document）
          - 远程 URL 直接传给 Telegram API（不做本地下载）
          - 本地文件读取为 bytes 后上传
          - 通过 _call_with_retry 包装，自动处理网络超时和 Flood Control

        Phase 3 —— 发送文本内容：
          - 跳过 "[empty message]" 占位符
          - 工具提示（_tool_hint）渲染为可折叠 blockquote
          - 内联键盘（inline_keyboards）作为最后一条消息的 reply_markup 附加
          - 超长消息通过 split_message() 分片为 4000 字以内的多个 chunk
          - Markdown 转 HTML（_markdown_to_telegram_html），失败时回退到纯文本

        【参数说明】
        - msg: OutboundMessage → 从 MessageBus 分发的出站消息，包含：
          - content: 回复文本
          - media: 媒体文件路径列表
          - chat_id: 目标聊天 ID
          - metadata: 元数据（message_thread_id, message_id, _tool_hint, _progress 等）
        """
        if not self._app:
            self.logger.warning("bot not running")
            return

        # Only stop typing indicator and remove reaction for final responses
        if not msg.metadata.get("_progress", False):
            self._stop_typing(msg.chat_id)
            if reply_to_message_id := msg.metadata.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(msg.chat_id, int(reply_to_message_id))

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            self.logger.exception("Invalid chat_id: {}", msg.chat_id)
            return
        reply_to_message_id = msg.metadata.get("message_id")
        message_thread_id = msg.metadata.get("message_thread_id")
        if message_thread_id is None and reply_to_message_id is not None:
            message_thread_id = self._message_threads.get((msg.chat_id, reply_to_message_id))
        thread_kwargs = {}
        if message_thread_id is not None:
            thread_kwargs["message_thread_id"] = message_thread_id

        reply_params = None
        if self.config.reply_to_message:
            if reply_to_message_id:
                reply_params = ReplyParameters(
                    message_id=reply_to_message_id,
                    allow_sending_without_reply=True
                )

        # Send media files
        for media_path in (msg.media or []):
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": self._app.bot.send_photo,
                    "video": self._app.bot.send_video,
                    "voice": self._app.bot.send_voice,
                    "audio": self._app.bot.send_audio,
                }.get(media_type, self._app.bot.send_document)
                param = {
                    "photo": "photo",
                    "video": "video",
                    "voice": "voice",
                    "audio": "audio",
                }.get(media_type, "document")
                extra: dict[str, Any] = {}
                if media_type == "video":
                    extra["supports_streaming"] = True

                # Telegram Bot API accepts HTTP(S) URLs directly for media params.
                if self._is_remote_media_url(media_path):
                    ok, error = validate_url_target(media_path)
                    if not ok:
                        raise ValueError(f"unsafe media URL: {error}")
                    await self._call_with_retry(
                        sender,
                        chat_id=chat_id,
                        **{param: media_path},
                        reply_parameters=reply_params,
                        **thread_kwargs,
                        **extra,
                    )
                    continue

                media_bytes = Path(media_path).read_bytes()
                filename = Path(media_path).name
                send_kwargs = {param: media_bytes, "filename": filename}
                await self._call_with_retry(
                    sender,
                    chat_id=chat_id,
                    reply_parameters=reply_params,
                    **thread_kwargs,
                    **extra,
                    **send_kwargs,
                )
            except Exception:
                filename = media_path.rsplit("/", 1)[-1]
                self.logger.exception("Failed to send media {}", media_path)
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_params,
                    **thread_kwargs,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            render_as_blockquote = bool(msg.metadata.get("_tool_hint"))
            buttons = getattr(msg, "buttons", None) or []
            reply_markup = self._build_keyboard(buttons) if buttons else None
            text = msg.content
            # Fallback: no native keyboard → splice labels into the message so the choices survive.
            if buttons and reply_markup is None:
                text = f"{text}\n\n{self._buttons_as_text(buttons)}"
            chunks = split_message(text, TELEGRAM_MAX_MESSAGE_LEN)
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                await self._send_text(
                    chat_id, chunk, reply_params, thread_kwargs,
                    render_as_blockquote=render_as_blockquote,
                    reply_markup=reply_markup if is_last else None,
                )

    async def _call_with_retry(self, fn, *args, **kwargs):
        """带重试机制的 Telegram API 调用包装器。

        【中文名称】带重试的 API 调用

        【功能说明】
        对 Telegram Bot API 调用做 2 种错误的重试处理：

        1. TimedOut（网络超时）→ 指数退避重试（0.5s → 1s → 2s），最多 3 次
        2. RetryAfter（Flood Control / 频率限制）→ 按 Telegram 返回的 retry_after
           秒数等待后重试，最多 3 次

        3 次重试后仍然失败则向上抛出异常，由上层调用者（如 ChannelManager）处理。

        【参数说明】
        - fn: 异步可调用对象 → Telegram Bot API 方法（如 send_message, edit_message_text）
        - *args, **kwargs: 传给 fn 的参数

        【返回值】
        fn 的原始返回值
        """
        from telegram.error import RetryAfter

        for attempt in range(1, _SEND_MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except TimedOut:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = _SEND_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    "timeout (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except RetryAfter as e:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = float(e.retry_after)
                self.logger.warning(
                    "Flood Control (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        reply_params=None,
        thread_kwargs: dict | None = None,
        render_as_blockquote: bool = False,
        reply_markup=None,
    ) -> None:
        """Send a plain text message with HTML fallback."""
        try:
            html = _tool_hint_to_telegram_blockquote(text) if render_as_blockquote else _markdown_to_telegram_html(text)
            await self._call_with_retry(
                self._app.bot.send_message,
                chat_id=chat_id, text=html, parse_mode="HTML",
                reply_parameters=reply_params,
                reply_markup=reply_markup,
                **(thread_kwargs or {}),
            )
        except BadRequest as e:
            self.logger.warning("HTML parse failed, falling back to plain text: {}", e)
            try:
                await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=chat_id,
                    text=text,
                    reply_parameters=reply_params,
                    reply_markup=reply_markup,
                    **(thread_kwargs or {}),
                )
            except Exception:
                self.logger.exception("Error sending message")
                raise

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return isinstance(exc, BadRequest) and "message is not modified" in str(exc).lower()

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """渐进式消息编辑：首条发送，后续编辑同一条消息。

        【中文名称】流式增量发送

        【功能说明】
        这是 Telegram 渠道实现"打字机效果"的核心方法。ChannelManager 在收到
        LLM 的流式 token 时不断调用此方法，让用户在 LLM 生成过程中就能看到
        实时更新的消息内容。

        【完整流程 —— 3 种情况】
        情况 A —— 流结束（_stream_end 为 True）：
          1. 停止 typing indicator，移除 reaction
          2. 将完整的 markdown 文本转换为 Telegram HTML
          3. 如果 HTML 长度超过 4096 字符：分片发送 ——
             第一条编辑进原消息，后续从第二条开始作为独立消息追加
          4. Edit 失败时回退到发送原始 markdown 文本
          5. 回退也失败时使用 _send_text 做 HTML→plain text 双保险

        情况 B —— 首次 delta（buf.message_id 为 None）：
          1. 创建新的 _StreamBuf，存储 chat_id 对应的流式状态
          2. 去掉 markdown 标记（_strip_md_block），以纯文本形式发送首条消息
          3. 记录 message_id，后续 delta 都编辑同一条消息

        情况 C —— 中途 delta：
          1. 追加 delta 到 buffer
          2. 仅在距离上次编辑超过 stream_edit_interval 秒时才发送编辑
          3. 如果 buffer 超过 4000 字符，调用 _flush_stream_overflow 溢出处理
          4. 将 buffer 去掉 markdown 标记后编辑消息

        【参数说明】
        - chat_id: str → Telegram 聊天 ID
        - delta: str → 增量文本片段
        - metadata: dict → 可能包含 _stream_id（标记不同流）、_stream_end（流结束标记）
          message_thread_id（话题 ID）、message_id（原始消息 ID 用于移除 reaction）
        """
        if not self._app:
            return
        meta = metadata or {}
        int_chat_id = int(chat_id)
        stream_id = meta.get("_stream_id")

        if meta.get("_stream_end"):
            buf = self._stream_bufs.get(chat_id)
            if not buf or not buf.message_id or not buf.text:
                return
            if stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id:
                return
            self._stop_typing(chat_id)
            if reply_to_message_id := meta.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(chat_id, int(reply_to_message_id))
            thread_kwargs = {}
            if message_thread_id := meta.get("message_thread_id"):
                thread_kwargs["message_thread_id"] = message_thread_id
            raw_text = buf.text
            html = _markdown_to_telegram_html(raw_text)
            if len(html) <= TELEGRAM_HTML_MAX_LEN:
                primary_html = html
                extra_html_chunks = []
            else:
                html_chunks = split_message(html, TELEGRAM_HTML_MAX_LEN)
                primary_html = html_chunks[0]
                extra_html_chunks = html_chunks[1:]
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=primary_html, parse_mode="HTML",
                )
            except BadRequest as e:
                # Only fall back to plain text on actual HTML parse/format errors.
                # Network errors (TimedOut, NetworkError) should propagate immediately
                # to avoid doubling connection demand during pool exhaustion.
                if self._is_not_modified_error(e):
                    self.logger.debug("Final stream edit already applied for {}", chat_id)
                    self._stream_bufs.pop(chat_id, None)
                    return
                self.logger.debug("Final stream edit failed (HTML), trying plain: {}", e)
                # Fall back to raw markdown (not HTML) so users don't see raw tags.
                primary_plain = split_message(raw_text, TELEGRAM_MAX_MESSAGE_LEN)[0] if len(raw_text) > TELEGRAM_MAX_MESSAGE_LEN else raw_text
                try:
                    await self._call_with_retry(
                        self._app.bot.edit_message_text,
                        chat_id=int_chat_id, message_id=buf.message_id,
                        text=primary_plain,
                    )
                except Exception as e2:
                    if self._is_not_modified_error(e2):
                        self.logger.debug("Final stream plain edit already applied for {}", chat_id)
                    else:
                        self.logger.warning("Final stream edit failed: {}", e2)
                        raise  # Let ChannelManager handle retry
            for extra_html_chunk in extra_html_chunks:
                try:
                    await self._call_with_retry(
                        self._app.bot.send_message,
                        chat_id=int_chat_id, text=extra_html_chunk,
                        parse_mode="HTML",
                        **thread_kwargs,
                    )
                except Exception:
                    # Fall back to _send_text which handles HTML→plain gracefully.
                    await self._send_text(int_chat_id, extra_html_chunk)
            self._stream_bufs.pop(chat_id, None)
            return

        buf = self._stream_bufs.get(chat_id)
        if buf is None or (stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id):
            buf = _StreamBuf(stream_id=stream_id)
            self._stream_bufs[chat_id] = buf
        elif buf.stream_id is None:
            buf.stream_id = stream_id
        buf.text += delta

        if not buf.text.strip():
            return

        now = time.monotonic()
        thread_kwargs = {}
        if message_thread_id := meta.get("message_thread_id"):
            thread_kwargs["message_thread_id"] = message_thread_id
        if buf.message_id is None:
            preview = _strip_md_block(buf.text)
            try:
                sent = await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=int_chat_id, text=preview,
                    **thread_kwargs,
                )
                buf.message_id = sent.message_id
                buf.last_edit = now
            except Exception as e:
                self.logger.warning("Stream initial send failed: {}", e)
                raise  # Let ChannelManager handle retry
        elif (now - buf.last_edit) >= self.config.stream_edit_interval:
            if len(buf.text) > TELEGRAM_MAX_MESSAGE_LEN:
                await self._flush_stream_overflow(int_chat_id, buf, thread_kwargs)
                buf.last_edit = now
                return
            preview = _strip_md_block(buf.text)
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=preview,
                )
                buf.last_edit = now
            except Exception as e:
                if self._is_not_modified_error(e):
                    buf.last_edit = now
                    return
                self.logger.warning("Stream edit failed: {}", e)
                raise  # Let ChannelManager handle retry

    async def _flush_stream_overflow(
        self,
        chat_id: int,
        buf: "_StreamBuf",
        thread_kwargs: dict,
    ) -> None:
        """Split an oversized stream buffer mid-flight.

        Edits the current stream message with the first chunk, sends any
        intermediate chunks as standalone messages, then opens a new message
        for the tail so subsequent deltas continue streaming into it.
        """
        chunks = split_message(buf.text, TELEGRAM_MAX_MESSAGE_LEN)
        if len(chunks) <= 1:
            return
        try:
            await self._call_with_retry(
                self._app.bot.edit_message_text,
                chat_id=chat_id, message_id=buf.message_id,
                text=chunks[0],
            )
        except Exception as e:
            if not self._is_not_modified_error(e):
                self.logger.warning("Stream overflow edit failed: {}", e)
                raise
        for chunk in chunks[1:-1]:
            await self._call_with_retry(
                self._app.bot.send_message,
                chat_id=chat_id, text=chunk, **thread_kwargs,
            )
        tail = chunks[-1]
        sent = await self._call_with_retry(
            self._app.bot.send_message,
            chat_id=chat_id, text=tail, **thread_kwargs,
        )
        buf.message_id = sent.message_id
        buf.text = tail

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, update.message, user)
            return
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command for allowed users only."""
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, update.message, user)
            return
        await update.message.reply_text(build_help_text())

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _send_pairing_code_if_private(self, sender_id: str, message, user) -> None:
        if message.chat.type != "private":
            return
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content="",
            metadata=self._build_message_metadata(message, user),
            is_dm=True,
        )

    @staticmethod
    def _derive_topic_session_key(message) -> str | None:
        """Derive topic-scoped session key for Telegram chats with threads."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return None
        return f"telegram:{message.chat_id}:topic:{message_thread_id}"

    @staticmethod
    def _build_message_metadata(message, user) -> dict:
        """Build common Telegram inbound metadata payload."""
        reply_to = getattr(message, "reply_to_message", None)
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
            "message_thread_id": getattr(message, "message_thread_id", None),
            "is_forum": bool(getattr(message.chat, "is_forum", False)),
            "reply_to_message_id": getattr(reply_to, "message_id", None) if reply_to else None,
        }

    async def _extract_reply_context(self, message) -> str | None:
        """Extract text from the message being replied to, if any."""
        reply = getattr(message, "reply_to_message", None)
        if not reply:
            return None
        text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
        if len(text) > TELEGRAM_REPLY_CONTEXT_MAX_LEN:
            text = text[:TELEGRAM_REPLY_CONTEXT_MAX_LEN] + "..."

        if not text:
            return None

        bot_id, _ = await self._ensure_bot_identity()
        reply_user = getattr(reply, "from_user", None)

        if bot_id and reply_user and getattr(reply_user, "id", None) == bot_id:
            return f"[Reply to bot: {text}]"
        elif reply_user and getattr(reply_user, "username", None):
            return f"[Reply to @{reply_user.username}: {text}]"
        elif reply_user and getattr(reply_user, "first_name", None):
            return f"[Reply to {reply_user.first_name}: {text}]"
        else:
            return f"[Reply to: {text}]"

    async def _download_message_media(
        self, msg, *, add_failure_content: bool = False
    ) -> tuple[list[str], list[str]]:
        """Download media from a message (current or reply). Returns (media_paths, content_parts)."""
        media_file = None
        media_type = None
        if getattr(msg, "photo", None):
            media_file = msg.photo[-1]
            media_type = "image"
        elif getattr(msg, "voice", None):
            media_file = msg.voice
            media_type = "voice"
        elif getattr(msg, "audio", None):
            media_file = msg.audio
            media_type = "audio"
        elif getattr(msg, "document", None):
            media_file = msg.document
            media_type = "file"
        elif getattr(msg, "video", None):
            media_file = msg.video
            media_type = "video"
        elif getattr(msg, "video_note", None):
            media_file = msg.video_note
            media_type = "video"
        elif getattr(msg, "animation", None):
            media_file = msg.animation
            media_type = "animation"
        if not media_file or not self._app:
            return [], []
        try:
            file = await self._app.bot.get_file(media_file.file_id)
            ext = self._get_extension(
                media_type,
                getattr(media_file, "mime_type", None),
                getattr(media_file, "file_name", None),
            )
            media_dir = get_media_dir("telegram")
            unique_id = getattr(media_file, "file_unique_id", media_file.file_id)
            file_path = media_dir / f"{unique_id}{ext}"
            await file.download_to_drive(str(file_path))
            path_str = str(file_path)
            if media_type in ("voice", "audio"):
                transcription = await self.transcribe_audio(file_path)
                if transcription:
                    self.logger.info("Transcribed {}: {}...", media_type, transcription[:50])
                    return [path_str], [f"[transcription: {transcription}]"]
                return [path_str], [f"[{media_type}: {path_str}]"]
            return [path_str], [f"[{media_type}: {path_str}]"]
        except Exception as e:
            self.logger.warning("Failed to download message media: {}", e)
            if add_failure_content:
                return [], [f"[{media_type}: download failed]"]
            return [], []

    async def _ensure_bot_identity(self) -> tuple[int | None, str | None]:
        """Load bot identity once and reuse it for mention/reply checks."""
        if self._bot_user_id is not None or self._bot_username is not None:
            return self._bot_user_id, self._bot_username
        if not self._app:
            return None, None
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        return self._bot_user_id, self._bot_username

    @staticmethod
    def _has_mention_entity(
        text: str,
        entities,
        bot_username: str,
        bot_id: int | None,
    ) -> bool:
        """Check Telegram mention entities against the bot username."""
        handle = f"@{bot_username}".lower()
        for entity in entities or []:
            entity_type = getattr(entity, "type", None)
            if entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and bot_id is not None and getattr(user, "id", None) == bot_id:
                    return True
                continue
            if entity_type != "mention":
                continue
            offset = getattr(entity, "offset", None)
            length = getattr(entity, "length", None)
            if offset is None or length is None:
                continue
            if text[offset : offset + length].lower() == handle:
                return True
        return handle in text.lower()

    async def _is_group_message_for_bot(self, message) -> bool:
        """Allow group messages when policy is open, @mentioned, or replying to the bot."""
        if message.chat.type == "private" or self.config.group_policy == "open":
            return True

        bot_id, bot_username = await self._ensure_bot_identity()
        if bot_username:
            text = message.text or ""
            caption = message.caption or ""
            if self._has_mention_entity(
                text,
                getattr(message, "entities", None),
                bot_username,
                bot_id,
            ):
                return True
            if self._has_mention_entity(
                caption,
                getattr(message, "caption_entities", None),
                bot_username,
                bot_id,
            ):
                return True

        reply_user = getattr(getattr(message, "reply_to_message", None), "from_user", None)
        return bool(bot_id and reply_user and reply_user.id == bot_id)

    def _remember_thread_context(self, message) -> None:
        """Cache Telegram thread context by chat/message id for follow-up replies."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return
        key = (str(message.chat_id), message.message_id)
        self._message_threads[key] = message_thread_id
        if len(self._message_threads) > 1000:
            self._message_threads.pop(next(iter(self._message_threads)))

    @staticmethod
    def _queue_key_for_message(message) -> str:
        """Return the final nanobot session key used for ordered Telegram ingress."""
        return TelegramChannel._derive_topic_session_key(message) or f"telegram:{message.chat_id}"

    @staticmethod
    def _sort_key_for_update(update: Update) -> tuple[int, int]:
        """Sort by chat message id first, then Telegram update id."""
        message = getattr(update, "message", None)
        message_id = int(getattr(message, "message_id", 0) or 0)
        update_id = int(getattr(update, "update_id", 0) or 0)
        return (message_id, update_id)

    def _enqueue_ordered_update(
        self,
        *,
        kind: Literal["command", "message"],
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Stage a Telegram update behind a short per-session reorder window."""
        message = update.message
        key = self._queue_key_for_message(message)
        self._inbound_buffers.setdefault(key, []).append(
            _QueuedTelegramUpdate(
                kind=kind,
                update=update,
                context=context,
                sort_key=self._sort_key_for_update(update),
            )
        )
        if key not in self._inbound_workers:
            self._inbound_workers[key] = asyncio.create_task(
                self._drain_ordered_updates(key)
            )

    async def _drain_ordered_updates(self, key: str) -> None:
        """Drain one Telegram session buffer in stable message order."""
        try:
            while self._running:
                await asyncio.sleep(0.2)
                batch = self._inbound_buffers.get(key, [])
                if not batch:
                    break
                self._inbound_buffers[key] = []
                batch.sort(key=lambda item: item.sort_key)
                for item in batch:
                    try:
                        if item.kind == "command":
                            await self._process_forward_command(item.update, item.context)
                        else:
                            await self._process_message_update(item.update, item.context)
                    except Exception as e:
                        self.logger.warning(
                            "Telegram queued update handling failed for {}: {}",
                            key,
                            e,
                        )
            if not self._inbound_buffers.get(key):
                self._inbound_buffers.pop(key, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("Telegram ordered update worker failed for {}: {}", key, e)
        finally:
            if not self._inbound_buffers.get(key):
                self._inbound_workers.pop(key, None)

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        if not self._running:
            await self._process_forward_command(update, context)
            return
        self._enqueue_ordered_update(kind="command", update=update, context=context)

    async def _process_forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理一条排队的斜杠命令。

        【中文名称】处理命令

        【功能说明】
        将 Telegram 斜杠命令转发到消息总线，由 AgentLoop 统一处理。

        与普通消息的关键区别：
        - 命令内容是斜杠指令（如 /new, /stop, /status），直接作为 content 发送
        - 自动剥离 @bot_username 后缀（如 /new@MyBot → /new）
        - 自动做命令别名映射（如 /dream_log → /dream-log）

        【参数说明】
        - update: Update → python-telegram-bot 的命令更新对象
        - context: ContextTypes.DEFAULT_TYPE → Handler 上下文
        """
        message = update.message
        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, message, user)
            return
        self._remember_thread_context(message)

        # Strip @bot_username suffix if present
        content = message.text or ""
        if content.startswith("/") and "@" in content:
            cmd_part, *rest = content.split(" ", 1)
            cmd_part = cmd_part.split("@")[0]
            content = f"{cmd_part} {rest[0]}" if rest else cmd_part
        content = self._normalize_telegram_command(content)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content=content,
            metadata=self._build_message_metadata(message, user),
            session_key=self._derive_topic_session_key(message),
            is_dm=message.chat.type == "private",
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        if not self._running:
            await self._process_message_update(update, context)
            return
        self._enqueue_ordered_update(kind="message", update=update, context=context)

    async def _process_message_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理一条排队的 Telegram 消息更新。

        【中文名称】处理消息更新

        【功能说明】
        这是普通消息（非命令）的完整入站处理流程：

        【处理流程 —— 7 个步骤】
        Step 1 —— 权限校验：检查 sender_id 是否在 allow_from 白名单中，不允许则发配对码
        Step 2 —— 线程上下文：缓存 message_thread_id，确保后续回复在正确的论坛话题下
        Step 3 —— 群聊策略：非私聊消息需检查群聊策略——
                  "open" 模式直接通过，"mention" 模式只响应 @bot 或回复 bot 的消息
        Step 4 —— 收集文本内容：正文 + 标题（caption）+ 位置信息
        Step 5 —— 下载当前消息的媒体（图片/语音/视频/文档）
        Step 6 —— 提取回复上下文：被回复消息的文本 + 媒体，拼接到 content 头部
        Step 7 —— 媒体组聚合（media_group_id 存在时）：
                  0.6 秒缓冲窗口内将同一 media_group 的多条消息合并为一次 turn
        Step 8 —— 最终通过 _handle_message 发送到 MessageBus，进入 AgentLoop

        【参数说明】
        - update: Update → python-telegram-bot 的消息更新对象
        - context: ContextTypes.DEFAULT_TYPE → Handler 上下文
        """

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, message, user)
            return
        self._remember_thread_context(message)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        if not await self._is_group_message_for_bot(message):
            return

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Location content
        if message.location:
            lat = message.location.latitude
            lon = message.location.longitude
            content_parts.append(f"[location: {lat}, {lon}]")

        # Download current message media
        current_media_paths, current_media_parts = await self._download_message_media(
            message, add_failure_content=True
        )
        media_paths.extend(current_media_paths)
        content_parts.extend(current_media_parts)
        if current_media_paths:
            self.logger.debug("Downloaded message media to {}", current_media_paths[0])

        # Reply context: text and/or media from the replied-to message
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            reply_ctx = await self._extract_reply_context(message)
            reply_media, reply_media_parts = await self._download_message_media(reply)
            if reply_media:
                media_paths = reply_media + media_paths
                self.logger.debug("Attached replied-to media: {}", reply_media[0])
            tag = reply_ctx or (f"[Reply to: {reply_media_parts[0]}]" if reply_media_parts else None)
            if tag:
                content_parts.insert(0, tag)
        content = "\n".join(content_parts) if content_parts else "[empty message]"

        self.logger.debug("message from {}: {}...", sender_id, content[:50])

        str_chat_id = str(chat_id)
        metadata = self._build_message_metadata(message, user)
        session_key = self._derive_topic_session_key(message)

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id, "chat_id": str_chat_id,
                    "contents": [], "media": [],
                    "metadata": metadata,
                    "session_key": session_key,
                }
                self._start_typing(str_chat_id)
                await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)
            buf = self._media_group_buffers[key]
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata=metadata,
            session_key=session_key,
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            await self._handle_message(
                sender_id=buf["sender_id"], chat_id=buf["chat_id"],
                content=content, media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
                session_key=buf.get("session_key"),
            )
        finally:
            self._media_group_tasks.pop(key, None)

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _add_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        """Add emoji reaction to a message (best-effort, non-blocking)."""
        if not self._app or not emoji:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            self.logger.debug("reaction failed: {}", e)

    async def _remove_reaction(self, chat_id: str, message_id: int) -> None:
        """Remove emoji reaction from a message (best-effort, non-blocking)."""
        if not self._app:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[],
            )
        except Exception as e:
            self.logger.debug("reaction removal failed: {}", e)

    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            with suppress(asyncio.CancelledError):
                while self._app:
                    await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                    await asyncio.sleep(4)
        except Exception as e:
            self.logger.debug("Typing indicator stopped for {}: {}", chat_id, e)

    @staticmethod
    def _format_telegram_error(exc: Exception) -> str:
        """Return a short, readable error summary for logs."""
        text = str(exc).strip()
        if text:
            return text
        if exc.__cause__ is not None:
            cause = exc.__cause__
            cause_text = str(cause).strip()
            if cause_text:
                return f"{exc.__class__.__name__} ({cause_text})"
            return f"{exc.__class__.__name__} ({cause.__class__.__name__})"
        return exc.__class__.__name__

    def _on_polling_error(self, exc: Exception) -> None:
        """Keep long-polling network failures to a single readable line."""
        summary = self._format_telegram_error(exc)
        if isinstance(exc, (NetworkError, TimedOut)):
            self.logger.warning("polling network issue: {}", summary)
        else:
            self.logger.error("polling error: {}", summary)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        summary = self._format_telegram_error(context.error)

        if isinstance(context.error, (NetworkError, TimedOut)):
            self.logger.warning("network issue: {}", summary)
        else:
            self.logger.error("error: {}", summary)

    def _get_extension(
        self,
        media_type: str,
        mime_type: str | None,
        filename: str | None = None,
    ) -> str:
        """Get file extension based on media type or original filename."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "image/webp": ".webp",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
                "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
                "video/x-matroska": ".mkv", "video/3gpp": ".3gp",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "video": ".mp4", "file": ""}
        if ext := type_map.get(media_type, ""):
            return ext

        if filename:
            return "".join(Path(filename).suffixes)

        return ""

    def _build_keyboard(self, buttons: list) -> InlineKeyboardMarkup | None:
        """Build inline keyboard markup if inline_keyboards is enabled."""
        if not buttons or not self.config.inline_keyboards:
            return None
        keyboard = [
            [InlineKeyboardButton(label, callback_data=self._safe_callback_data(label)) for label in row]
            for row in buttons
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _safe_callback_data(label: str) -> str:
        # Telegram caps callback_data at 64 bytes UTF-8; truncate at a char boundary so the keyboard still sends.
        encoded = label.encode("utf-8")
        if len(encoded) <= 64:
            return label
        return encoded[:64].decode("utf-8", errors="ignore")

    @staticmethod
    def _buttons_as_text(buttons: list[list[str]]) -> str:
        # Buttons are semantic options; when we can't render a keyboard, the user still needs to see them.
        return "\n".join(" ".join(f"[{label}]" for label in row) for row in buttons if row)

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button clicks (callback queries)."""
        if not update.callback_query or not update.effective_user:
            return
        query = update.callback_query
        user = update.effective_user
        chat_id = query.message.chat_id if query.message else None
        sender_id = self._sender_id(user)
        if not chat_id:
            self.logger.warning("Callback query without chat_id")
            return
        if not self.is_allowed(sender_id):
            return
        button_label = query.data or ""
        await query.answer()
        if query.message:
            with suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)
        self.logger.debug("Inline button tap from {}: {}", sender_id, button_label)
        self._start_typing(str(chat_id))
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=button_label,
            metadata={
                "callback_query_id": query.id,
                "button_label": button_label,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_callback": True,
            },
        )
