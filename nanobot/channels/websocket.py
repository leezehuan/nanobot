"""WebSocket server channel: nanobot acts as a WebSocket server and serves connected clients."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import ssl
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator
from websockets.asyncio.server import ServerConnection, serve, unix_serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request as WsRequest

from nanobot.bus.events import OUTBOUND_META_AGENT_UI, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScopeError,
)
from nanobot.session.goal_state import goal_state_ws_blob
from nanobot.session.webui_turns import websocket_turn_wall_started_at
from nanobot.utils.media_decode import (
    FileSizeExceeded,
    save_base64_data_url,
)
from nanobot.webui.cli_apps_api import normalize_cli_app_mentions
from nanobot.webui.forking import handle_webui_fork_chat
from nanobot.webui.gateway_services import GatewayServices
from nanobot.webui.http_utils import (
    normalize_config_path as _normalize_config_path,
)
from nanobot.webui.http_utils import (
    parse_request_path as _parse_request_path,
)
from nanobot.webui.http_utils import (
    query_first as _query_first,
)
from nanobot.webui.mcp_presets_api import normalize_mcp_preset_mentions
from nanobot.webui.transcription_ws import webui_transcription_event
from nanobot.webui.websocket_logging import websockets_server_logger


class WebSocketConfig(Base):
    """WebSocket server channel configuration.

    Clients connect with URLs like ``ws://{host}:{port}{path}?client_id=...&token=...``.
    - ``client_id``: Used for ``allow_from`` authorization; if omitted, a value is generated and logged.
    - ``token``: If non-empty, the ``token`` query param may match this static secret; short-lived tokens
      from ``token_issue_path`` are also accepted.
    - ``token_issue_path``: If non-empty, **GET** (HTTP/1.1) to this path returns JSON
      ``{"token": "...", "expires_in": <seconds>}``; use ``?token=...`` when opening the WebSocket.
      Must differ from ``path`` (the WS upgrade path). If the client runs in the **same process** as
      nanobot and shares the asyncio loop, use a thread or async HTTP client for GET—do not call
      blocking ``urllib`` or synchronous ``httpx`` from inside a coroutine.
    - ``token_issue_secret``: If non-empty, token requests must send ``Authorization: Bearer <secret>`` or
      ``X-Nanobot-Auth: <secret>``.
    - ``websocket_requires_token``: If True, the handshake must include a valid token (static or issued and not expired).
    - Each connection has its own session: a unique ``chat_id`` maps to the agent session internally.
    - ``media`` field in outbound messages contains local filesystem paths; remote clients need a
      shared filesystem or an HTTP file server to access these files.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    unix_socket_path: str = ""
    path: str = "/"
    token: str = ""
    token_issue_path: str = ""
    token_issue_secret: str = ""
    token_ttl_s: int = Field(default=300, ge=30, le=86_400)
    websocket_requires_token: bool = True
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    streaming: bool = True
    # Default 36 MB, upper 40 MB: supports up to 4 images at ~6 MB each after
    # client-side Worker normalization (see webui Composer). 4 × 6 MB × 1.37
    # (base64 overhead) + envelope framing stays under 36 MB; the 40 MB ceiling
    # leaves a small margin for sender slop without opening a DoS avenue.
    max_message_bytes: int = Field(default=37_748_736, ge=1024, le=41_943_040)
    ping_interval_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ping_timeout_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ssl_certfile: str = ""
    ssl_keyfile: str = ""

    @field_validator("unix_socket_path")
    @classmethod
    def unix_socket_path_format(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        if "\x00" in value:
            raise ValueError("unix_socket_path must not contain NUL bytes")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("unix_socket_path must be an absolute path")
        return str(path)

    @field_validator("path")
    @classmethod
    def path_must_start_with_slash(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError('path must start with "/"')
        return _normalize_config_path(value)

    @field_validator("token_issue_path")
    @classmethod
    def token_issue_path_format(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        if not value.startswith("/"):
            raise ValueError('token_issue_path must start with "/"')
        return _normalize_config_path(value)

    @model_validator(mode="after")
    def token_issue_path_differs_from_ws_path(self) -> Self:
        if not self.token_issue_path:
            return self
        if _normalize_config_path(self.token_issue_path) == _normalize_config_path(self.path):
            raise ValueError("token_issue_path must differ from path (the WebSocket upgrade path)")
        return self

    @model_validator(mode="after")
    def wildcard_host_requires_auth(self) -> Self:
        if self.host not in ("0.0.0.0", "::"):
            return self
        if self.token.strip() or self.token_issue_secret.strip():
            return self
        raise ValueError(
            "host is 0.0.0.0 (all interfaces) but neither token nor "
            "token_issue_secret is set — set one to prevent unauthenticated access"
        )


def publish_runtime_model_update(
    bus: MessageBus,
    model: str,
    model_preset: str | None,
) -> None:
    """Enqueue a runtime model snapshot for websocket subscribers (fan-out in-channel)."""
    bus.outbound.put_nowait(OutboundMessage(
        channel="websocket",
        chat_id="*",
        content="",
        metadata={
            "_runtime_model_updated": True,
            "model": model,
            "model_preset": model_preset,
        },
    ))


def _parse_inbound_payload(raw: str) -> str | None:
    """Parse a client frame into text; return None for empty or unrecognized content."""
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict):
            for key in ("content", "text", "message"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None
        return None
    return text


# Accept UUIDs and short scoped keys like "unified:default". Keeps the capability
# namespace small enough to rule out path traversal / quote injection tricks.
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,64}$")


def _is_valid_chat_id(value: Any) -> bool:
    return isinstance(value, str) and _CHAT_ID_RE.match(value) is not None


def _parse_envelope(raw: str) -> dict[str, Any] | None:
    """Return a typed envelope dict if the frame is a new-style JSON envelope, else None.

    A frame qualifies when it parses as a JSON object with a string ``type`` field.
    Legacy frames (plain text, or ``{"content": ...}`` without ``type``) return None;
    callers should fall back to :func:`_parse_inbound_payload` for those.
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    t = data.get("type")
    if not isinstance(t, str):
        return None
    return data


# Per-message media limits. The server-side guard is a touch looser than the
# client's ``Worker`` normalization target (6 MB) — tolerate client slop, but
# still cap total ingress at ``_MAX_IMAGES_PER_MESSAGE * _MAX_IMAGE_BYTES``
# which fits comfortably inside ``max_message_bytes``.
_MAX_IMAGES_PER_MESSAGE = 4
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_VIDEOS_PER_MESSAGE = 1
_MAX_VIDEO_BYTES = 20 * 1024 * 1024

# Image MIME whitelist — matches the Composer's ``accept`` list. SVG is
# explicitly excluded to avoid the XSS surface inside embedded scripts.
_IMAGE_MIME_ALLOWED: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})

_VIDEO_MIME_ALLOWED: frozenset[str] = frozenset({
    "video/mp4",
    "video/webm",
    "video/quicktime",
})

_UPLOAD_MIME_ALLOWED: frozenset[str] = _IMAGE_MIME_ALLOWED | _VIDEO_MIME_ALLOWED

_DATA_URL_MIME_RE = re.compile(r"^data:([^;,]+)(?:;[^,]*)*;base64,", re.DOTALL)


def _extract_data_url_mime(url: str) -> str | None:
    """Return the MIME type of a ``data:<mime>;base64,...`` URL, else ``None``."""
    if not isinstance(url, str):
        return None
    m = _DATA_URL_MIME_RE.match(url)
    if not m:
        return None
    return m.group(1).strip().lower() or None


def _is_websocket_upgrade(request: WsRequest) -> bool:
    """Detect an actual WS upgrade; plain HTTP GETs to the same path should fall through."""
    upgrade = request.headers.get("Upgrade") or request.headers.get("upgrade")
    connection = request.headers.get("Connection") or request.headers.get("connection")
    if not upgrade or "websocket" not in upgrade.lower():
        return False
    if not connection or "upgrade" not in connection.lower():
        return False
    return True


class WebSocketChannel(BaseChannel):
    """Run a local WebSocket server; forward text/JSON messages to the message bus.

    【中文名称】WebSocket 渠道适配器

    【功能说明】
    这是 nanobot WebUI 与后端通信的核心通道。它在一个本地端口（默认 8765）
    上运行 WebSocket 服务器，WebUI 前端通过它实现：
    - 实时聊天：用户消息 → WS → MessageBus → AgentLoop → LLM → 流式返回
    - 多会话管理：一个连接可以订阅多个 chat_id，支持多标签页同时使用
    - 文件编辑进度：实时推送 write_file / edit_file 工具的增量变更
    - 长期任务状态：sustained goal 的 goal_state 同步
    - 运行时模型切换：runtime_model_updated 事件广播

    【关键架构】
    ┌─────────────┐     WebSocket      ┌──────────────────┐
    │  React SPA   │ ◄──────────────► │  WebSocketChannel │
    │  (WebUI)     │   JSON frames     │  (this class)    │
    └─────────────┘                    └──────┬───────────┘
                                              │ OutboundMessage
                                        ┌─────▼──────┐
                                        │ MessageBus  │
                                        └─────┬──────┘
                                              │
                                        ┌─────▼──────┐
                                        │ AgentLoop   │
                                        └────────────┘

    【入站消息类型（envelope dispatch）】
    - "message"：用户聊天消息（文本 + 图片/视频 base64）
    - "new_chat"：创建新会话
    - "attach"：客户端订阅已有的 chat_id
    - "fork_chat"：从已有会话分叉
    - "set_workspace_scope"：设置工作区访问范围
    - "transcribe_audio"：音频转写请求

    【出站消息类型】
    - "message"：Agent 回复文本
    - "delta" / "stream_end"：流式增量 token
    - "reasoning_delta" / "reasoning_end"：模型思考过程
    - "file_edit"：文件编辑进度事件
    - "turn_end"：回合结束信号（含 latency_ms）
    - "goal_state" / "goal_status"：长期任务状态同步
    - "session_updated"：会话范围变更通知
    - "runtime_model_updated"：运行时模型切换通知

    【连接鉴权】
    - 静态 token：通过 config.token 配置
    - 动态 token：通过 token_issue_path 端点签发（支持 Bearer 认证）
    - websocket_requires_token：是否强制要求 token
    """

    name = "websocket"
    display_name = "WebSocket"

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        *,
        gateway: GatewayServices,
    ):
        if isinstance(config, dict):
            config = WebSocketConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WebSocketConfig = config
        # chat_id -> connections subscribed to it (fan-out target).
        self._subs: dict[str, set[Any]] = {}
        # connection -> chat_ids it is subscribed to (O(1) cleanup on disconnect).
        self._conn_chats: dict[Any, set[str]] = {}
        # connection -> default chat_id for legacy frames that omit routing.
        self._conn_default: dict[Any, str] = {}
        self._stop_event: asyncio.Event | None = None
        self._server_task: asyncio.Task[None] | None = None

        self.gateway = gateway
        self._http_router = gateway.http
        self._tokens = gateway.tokens
        self._media = gateway.media
        self._transcripts = gateway.transcripts
        self._workspaces = gateway.workspaces

        self._stream_text_buffers: dict[tuple[str, str], list[str]] = {}

    # -- Subscription bookkeeping -------------------------------------------

    def _workspace_controls_available(self, connection: Any) -> bool:
        return self._http_router.workspace_controls_available(connection)

    def _attach(self, connection: Any, chat_id: str) -> None:
        """Idempotently subscribe *connection* to *chat_id*."""
        self._subs.setdefault(chat_id, set()).add(connection)
        self._conn_chats.setdefault(connection, set()).add(chat_id)

    def _cleanup_connection(self, connection: Any) -> None:
        """Remove *connection* from every subscription set; safe to call multiple times."""
        chat_ids = self._conn_chats.pop(connection, set())
        for cid in chat_ids:
            subs = self._subs.get(cid)
            if subs is None:
                continue
            subs.discard(connection)
            if not subs:
                self._subs.pop(cid, None)
        self._conn_default.pop(connection, None)

    async def _maybe_push_active_goal_state(self, chat_id: str) -> None:
        """Replay an active sustained goal from session metadata after *chat_id* is subscribed.

        Goal metadata lives on the session JSONL and survives gateway restarts, but
        connected clients normally see it via ``goal_state`` / ``turn_end`` frames.
        Pushing here makes refresh + reconnect restore the strip without a new model turn.
        """
        if self.gateway.session_manager is None:
            return
        row = self.gateway.session_manager.read_session_file(f"websocket:{chat_id}")
        meta = row.get("metadata", {}) if isinstance(row, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        blob = goal_state_ws_blob(meta)
        if not blob.get("active"):
            return
        await self.send_goal_state(chat_id, blob)

    async def _maybe_push_turn_run_wall_clock(self, chat_id: str) -> None:
        """Replay ``goal_status: running`` when a turn is still active (same-process refresh)."""
        t0 = websocket_turn_wall_started_at(chat_id)
        if t0 is None:
            return
        await self.send_goal_status(chat_id, "running", started_at=t0)

    async def _hydrate_after_subscribe(self, chat_id: str) -> None:
        """Replay goal/run strip state after subscribe (same-process refresh)."""
        await self._maybe_push_active_goal_state(chat_id)
        await self._maybe_push_turn_run_wall_clock(chat_id)

    async def _send_event(self, connection: Any, event: str, **fields: Any) -> None:
        """Send a control event (attached, error, ...) to a single connection."""
        payload: dict[str, Any] = {"event": event}
        payload.update(fields)
        raw = json.dumps(payload, ensure_ascii=False)
        try:
            await connection.send(raw)
        except ConnectionClosed:
            self._cleanup_connection(connection)
        except Exception as e:
            self.logger.warning("failed to send {} event: {}", event, e)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WebSocketConfig().model_dump(by_alias=True)

    def _expected_path(self) -> str:
        return _normalize_config_path(self.config.path)

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        cert = self.config.ssl_certfile.strip()
        key = self.config.ssl_keyfile.strip()
        if not cert and not key:
            return None
        if not cert or not key:
            raise ValueError(
                "ssl_certfile and ssl_keyfile must both be set for WSS, or both left empty"
            )
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=cert, keyfile=key)
        return ctx

    # -- HTTP dispatch ------------------------------------------------------

    async def _dispatch_http(self, connection: Any, request: WsRequest) -> Any:
        """Route an inbound HTTP request to the HTTP handler or WS upgrade."""
        got, query = _parse_request_path(request.path)

        # WebSocket upgrade — channel handles this itself
        expected_ws = self._expected_path()
        if got == expected_ws and _is_websocket_upgrade(request):
            client_id = _query_first(query, "client_id") or ""
            if len(client_id) > 128:
                client_id = client_id[:128]
            if not self.is_allowed(client_id):
                return connection.respond(403, "Forbidden")
            return self._authorize_websocket_handshake(connection, query)

        # Everything else goes to the HTTP handler
        return await self._http_router.dispatch(connection, request)

    def _authorize_websocket_handshake(self, connection: Any, query: dict[str, list[str]]) -> Any:
        supplied = _query_first(query, "token")
        static_token = self.config.token.strip()

        if static_token:
            if supplied and hmac.compare_digest(supplied, static_token):
                return None
            if supplied and self._tokens.take_issued_token_if_valid(supplied):
                return None
            return connection.respond(401, "Unauthorized")

        if self.config.websocket_requires_token:
            if supplied and self._tokens.take_issued_token_if_valid(supplied):
                return None
            return connection.respond(401, "Unauthorized")

        if supplied:
            self._tokens.take_issued_token_if_valid(supplied)
        return None

    # -- Server lifecycle and connection ingress ---------------------------

    async def start(self) -> None:
        """启动 WebSocket 服务器，进入监听循环。

        【中文名称】启动 WebSocket 服务器

        【功能说明】
        这是 WebSocket 渠道的完整启动流程：

        【启动流程 —— 5 个阶段】
        Phase 1 —— 构建 SSL 上下文：
          如果配置了 ssl_certfile + ssl_keyfile，创建 TLS 1.2+ 的 SSLContext，
          支持 wss:// 加密连接

        Phase 2 —— 注册 process_request 回调：
          所有 HTTP 请求（包括 WS 升级、token 签发、静态文件等）先进入
          _dispatch_http() 统一分发；
          - 是 WS 升级且路径匹配 → _authorize_websocket_handshake 做鉴权
          - 其他请求 → 转给 gateway HTTP router 处理

        Phase 3 —— 创建服务器：
          - Unix socket 模式：unix_serve（仅限本机进程间通信，安全隔离）
          - TCP 模式：serve（可配置 host:port，支持远程访问）
          两种模式都支持 max_message_bytes / ping_interval / ping_timeout

        Phase 4 —— 等待停止信号：
          await self._stop_event.wait()，直到 stop() 方法被调用

        Phase 5 —— 清理：
          关闭服务器，等待所有连接优雅退出；
          Unix socket 模式下删除 socket 文件

        【参数说明】
        无显式参数，所有配置从 self.config (WebSocketConfig) 读取。
        """
        from nanobot.utils.logging_bridge import redirect_lib_logging

        redirect_lib_logging("websockets", level="WARNING")
        ws_logger = websockets_server_logger()

        self._running = True
        self._stop_event = asyncio.Event()

        ssl_context = self._build_ssl_context()
        scheme = "wss" if ssl_context else "ws"

        async def process_request(
            connection: ServerConnection,
            request: WsRequest,
        ) -> Any:
            return await self._dispatch_http(connection, request)

        async def handler(connection: ServerConnection) -> None:
            await self._connection_loop(connection)

        self.logger.info(
            "WebSocket server listening on {}",
            (
                f"unix:{self.config.unix_socket_path}{self.config.path}"
                if self.config.unix_socket_path
                else f"{scheme}://{self.config.host}:{self.config.port}{self.config.path}"
            ),
        )
        if self.config.token_issue_path:
            self.logger.info(
                "WebSocket token issue route: {}",
                (
                    f"unix:{self.config.unix_socket_path}{_normalize_config_path(self.config.token_issue_path)}"
                    if self.config.unix_socket_path
                    else (
                        f"{scheme}://{self.config.host}:{self.config.port}"
                        f"{_normalize_config_path(self.config.token_issue_path)}"
                    )
                ),
            )

        async def runner() -> None:
            socket_path = self.config.unix_socket_path
            if socket_path:
                path_obj = Path(socket_path)
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                with suppress(FileNotFoundError):
                    path_obj.unlink()
                server = await unix_serve(
                    handler,
                    socket_path,
                    process_request=process_request,
                    max_size=self.config.max_message_bytes,
                    ping_interval=self.config.ping_interval_s,
                    ping_timeout=self.config.ping_timeout_s,
                    logger=ws_logger,
                )
                with suppress(OSError):
                    path_obj.chmod(0o600)
            else:
                server = await serve(
                    handler,
                    self.config.host,
                    self.config.port,
                    process_request=process_request,
                    max_size=self.config.max_message_bytes,
                    ping_interval=self.config.ping_interval_s,
                    ping_timeout=self.config.ping_timeout_s,
                    ssl=ssl_context,
                    logger=ws_logger,
                )
            try:
                assert self._stop_event is not None
                await self._stop_event.wait()
            finally:
                server.close()
                await server.wait_closed()
                if socket_path:
                    with suppress(FileNotFoundError):
                        Path(socket_path).unlink()

        self._server_task = asyncio.create_task(runner())
        await self._server_task

    async def _connection_loop(self, connection: Any) -> None:
        """单个 WebSocket 连接的主事件循环。

        【中文名称】连接主循环

        【功能说明】
        每个 WebSocket 客户端连接都会进入这个循环。它负责该连接的完整生命周期：

        【生命周期 —— 4 个阶段】
        Phase 1 —— 握手与身份识别：
          - 从 query string 中提取 client_id（未提供时生成匿名 ID）
          - 生成 default_chat_id（UUID4），作为该连接默认的会话标识

        Phase 2 —— 发送 ready 事件：
          - 向客户端推送 {"event": "ready", "chat_id": "...", "client_id": "..."}
          - 将 connection 注册到订阅表（_conn_default + _attach）
          - 回放 active goal state（如果当前有长期任务在运行）

        Phase 3 —— 消息循环：
          - 循环接收 WebSocket frame
          - 先尝试解析为新式 JSON envelope（带 type 字段，支持多会话路由）
          - 如果不是 envelope，回退为纯文本消息（走 default_chat_id）
          - 非 envelope 消息不需要配对（WebSocket 已在握手时鉴权）

        Phase 4 —— 清理：
          - 连接断开或异常时，从所有订阅表中移除该连接

        【参数说明】
        - connection: ServerConnection → websockets 库的连接对象
        """
        request = connection.request
        path_part = request.path if request else "/"
        _, query = _parse_request_path(path_part)
        client_id_raw = _query_first(query, "client_id")
        client_id = client_id_raw.strip() if client_id_raw else ""
        if not client_id:
            client_id = f"anon-{uuid.uuid4().hex[:12]}"
        elif len(client_id) > 128:
            self.logger.warning("client_id too long ({} chars), truncating", len(client_id))
            client_id = client_id[:128]

        default_chat_id = str(uuid.uuid4())

        try:
            await connection.send(
                json.dumps(
                    {
                        "event": "ready",
                        "chat_id": default_chat_id,
                        "client_id": client_id,
                    },
                    ensure_ascii=False,
                )
            )
            # Register only after ready is successfully sent to avoid out-of-order sends
            self._conn_default[connection] = default_chat_id
            self._attach(connection, default_chat_id)
            await self._hydrate_after_subscribe(default_chat_id)

            async for raw in connection:
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        self.logger.warning("ignoring non-utf8 binary frame")
                        continue

                envelope = _parse_envelope(raw)
                if envelope is not None:
                    await self._dispatch_envelope(connection, client_id, envelope)
                    continue

                content = _parse_inbound_payload(raw)
                if content is None:
                    continue
                # WebSocket already authenticates at handshake time (token),
                # so pairing is not applicable. Treat as non-DM to avoid
                # sending pairing codes to an already-authenticated client.
                await self._handle_message(
                    sender_id=client_id,
                    chat_id=default_chat_id,
                    content=content,
                    metadata={"remote": getattr(connection, "remote_address", None)},
                    is_dm=False,
                )
        except Exception as e:
            self.logger.debug("connection ended: {}", e)
        finally:
            self._cleanup_connection(connection)

    # -- Inbound WebSocket envelopes ---------------------------------------

    def _save_envelope_media(
        self,
        media: list[Any],
    ) -> tuple[list[str], str | None]:
        """Decode and persist ``media`` items from a ``message`` envelope.

        Returns ``(paths, None)`` on success or ``([], reason)`` on the first
        failure — the caller is expected to surface ``reason`` to the client
        and skip publishing so no half-formed message ever reaches the agent.
        On failure, any files already written to disk earlier in the same
        call are unlinked so partial ingress doesn't leak orphan files.
        ``reason`` is a short, stable token suitable for UI localization.

        Shape: ``list[{"data_url": str, "name"?: str | None}]``.
        """
        image_count = 0
        video_count = 0
        for item in media:
            mime = _extract_data_url_mime(item.get("data_url", "")) if isinstance(item, dict) else None
            if mime in _VIDEO_MIME_ALLOWED:
                video_count += 1
            elif mime in _IMAGE_MIME_ALLOWED:
                image_count += 1
        if image_count > _MAX_IMAGES_PER_MESSAGE:
            return [], "too_many_images"
        if video_count > _MAX_VIDEOS_PER_MESSAGE:
            return [], "too_many_videos"

        media_dir = get_media_dir("websocket")
        paths: list[str] = []

        def _abort(reason: str) -> tuple[list[str], str]:
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError as exc:
                    self.logger.warning(
                        "failed to unlink partial media {}: {}", p, exc
                    )
            return [], reason

        for item in media:
            if not isinstance(item, dict):
                return _abort("malformed")
            data_url = item.get("data_url")
            if not isinstance(data_url, str) or not data_url:
                return _abort("malformed")
            mime = _extract_data_url_mime(data_url)
            if mime is None:
                return _abort("decode")
            if mime not in _UPLOAD_MIME_ALLOWED:
                return _abort("mime")
            is_video = mime in _VIDEO_MIME_ALLOWED
            max_bytes = _MAX_VIDEO_BYTES if is_video else _MAX_IMAGE_BYTES
            try:
                saved = save_base64_data_url(
                    data_url, media_dir, max_bytes=max_bytes,
                )
            except FileSizeExceeded:
                return _abort("size")
            except Exception as exc:
                self.logger.warning("media decode failed: {}", exc)
                return _abort("decode")
            if saved is None:
                return _abort("decode")
            paths.append(saved)
        return paths, None

    async def _dispatch_envelope(
        self,
        connection: Any,
        client_id: str,
        envelope: dict[str, Any],
    ) -> None:
        """分发一条带 type 字段的 JSON envelope 到对应处理逻辑。

        【中文名称】Envelope 分发器

        【功能说明】
        WebSocket 的新式消息协议：所有结构化消息都带 type 字段，不同 type
        走不同的处理分支。相比纯文本消息，envelope 能承载更丰富的语义。

        【支持的 type 及处理逻辑】

        "new_chat" —— 创建新会话：
          1. 生成新的 UUID chat_id
          2. 检查并设置 workspace scope（文件系统访问权限）
          3. 发送 attached + session_updated 事件
          4. 回放 goal state

        "fork_chat" —— 分叉会话：
          委托 handle_webui_fork_chat 处理，从已有会话复制上下文

        "attach" —— 订阅已有会话：
          1. 校验 chat_id 格式
          2. 将 connection 订阅到该 chat_id
          3. 回放 goal state

        "set_workspace_scope" —— 设置工作区范围：
          在 agent 空闲时修改该会话能访问的文件系统路径

        "transcribe_audio" —— 音频转写：
          委托 webui_transcription_event 处理

        "message" —— 用户消息（最常用）：
          1. 校验 chat_id 和 content
          2. 解析 media 字段（base64 图片/视频），落盘到本地
          3. 提取 workspace scope、image_generation 参数、cli_apps、mcp_presets
          4. 写入 WebUI 转录（用于前端回放）
          5. 通过 self._handle_message 发送到 MessageBus

        【参数说明】
        - connection: ServerConnection → 客户端连接对象
        - client_id: str → 客户端标识符
        - envelope: dict → JSON 消息体，必须包含 type 字段
        """
        t = envelope.get("type")
        if t == "new_chat":
            new_id = str(uuid.uuid4())
            scope = await self._workspace_scope_or_error(
                connection,
                lambda: self._workspaces.scope_for_new_chat(
                    envelope,
                    controls_available=self._workspace_controls_available(connection),
                ),
            )
            if scope is None:
                return
            self._workspaces.persist_scope(new_id, scope)
            self._attach(connection, new_id)
            await self._send_event(connection, "attached", chat_id=new_id)
            await self._send_event(
                connection,
                "session_updated",
                chat_id=new_id,
                scope="metadata",
                workspace_scope=scope.payload(),
            )
            await self._hydrate_after_subscribe(new_id)
            return
        if t == "fork_chat":
            await handle_webui_fork_chat(self, connection, envelope)
            return
        if t == "attach":
            cid = envelope.get("chat_id")
            if not _is_valid_chat_id(cid):
                await self._send_event(connection, "error", detail="invalid chat_id")
                return
            self._attach(connection, cid)
            await self._send_event(connection, "attached", chat_id=cid)
            await self._hydrate_after_subscribe(cid)
            return
        if t == "set_workspace_scope":
            cid = envelope.get("chat_id")
            if not _is_valid_chat_id(cid):
                await self._send_event(connection, "error", detail="invalid chat_id")
                return
            scope = await self._workspace_scope_or_error(
                connection,
                lambda: self._workspaces.scope_for_set_request(
                    envelope,
                    chat_id=cid,
                    chat_running=websocket_turn_wall_started_at(cid) is not None,
                    controls_available=self._workspace_controls_available(connection),
                ),
                chat_id=cid,
            )
            if scope is None:
                return
            self._workspaces.persist_scope(cid, scope)
            await self._send_event(
                connection,
                "session_updated",
                chat_id=cid,
                scope="metadata",
                workspace_scope=scope.payload(),
            )
            return
        if t == "transcribe_audio":
            event, payload = await webui_transcription_event(envelope)
            await self._send_event(connection, event, **payload)
            return
        if t == "message":
            cid = envelope.get("chat_id")
            content = envelope.get("content")
            if not _is_valid_chat_id(cid):
                await self._send_event(connection, "error", detail="invalid chat_id")
                return
            if not isinstance(content, str):
                await self._send_event(connection, "error", detail="missing content")
                return

            raw_media = envelope.get("media")
            media_paths: list[str] = []
            if raw_media is not None:
                if not isinstance(raw_media, list):
                    await self._send_event(
                        connection, "error",
                        detail="image_rejected", reason="malformed",
                    )
                    return
                media_paths, reason = self._save_envelope_media(raw_media)
                if reason is not None:
                    await self._send_event(
                        connection, "error",
                        detail="image_rejected", reason=reason,
                    )
                    return

            # Allow image-only turns (content may be empty when media is attached).
            if not content.strip() and not media_paths:
                await self._send_event(connection, "error", detail="missing content")
                return
            scope = await self._workspace_scope_or_error(
                connection,
                lambda: self._workspaces.scope_for_message(
                    envelope,
                    chat_id=cid,
                    chat_running=websocket_turn_wall_started_at(cid) is not None,
                    controls_available=self._workspace_controls_available(connection),
                ),
                chat_id=cid,
            )
            if scope is None:
                return

            # Auto-attach on first use so clients can one-shot without a separate attach.
            self._attach(connection, cid)
            await self._hydrate_after_subscribe(cid)
            metadata: dict[str, Any] = {"remote": getattr(connection, "remote_address", None)}
            if envelope.get("webui") is True:
                metadata["webui"] = True
                metadata.update(self._transcripts.client_turn_metadata(envelope.get("turn_id")))
            cli_apps = normalize_cli_app_mentions(envelope.get("cli_apps"))
            if cli_apps:
                metadata["cli_apps"] = cli_apps
            mcp_presets = normalize_mcp_preset_mentions(envelope.get("mcp_presets"))
            if mcp_presets:
                metadata["mcp_presets"] = mcp_presets
            metadata[WORKSPACE_SCOPE_METADATA_KEY] = scope.metadata()
            self._workspaces.persist_scope(cid, scope)
            image_generation = envelope.get("image_generation")
            if isinstance(image_generation, dict) and image_generation.get("enabled") is True:
                aspect_ratio = image_generation.get("aspect_ratio")
                metadata["image_generation"] = {
                    "enabled": True,
                    "aspect_ratio": aspect_ratio if isinstance(aspect_ratio, str) else None,
                }
            if metadata.get("webui") is True and self.is_allowed(client_id):
                self._transcripts.append_user_message(
                    cid,
                    content,
                    metadata=metadata,
                    media_paths=media_paths or None,
                    cli_apps=cli_apps or None,
                    mcp_presets=mcp_presets or None,
                )
            await self._handle_message(
                sender_id=client_id,
                chat_id=cid,
                content=content,
                media=media_paths or None,
                metadata=metadata,
                is_dm=False,
            )
            return
        await self._send_event(connection, "error", detail=f"unknown type: {t!r}")

    async def _workspace_scope_or_error(
        self,
        connection: Any,
        resolver: Callable[[], Any],
        *,
        chat_id: str | None = None,
    ) -> Any | None:
        try:
            return resolver()
        except WorkspaceScopeError as exc:
            await self._send_event(
                connection,
                "error",
                detail="workspace_scope_rejected",
                reason=exc.message,
                **({"chat_id": chat_id} if chat_id else {}),
            )
            return None

    # -- Outbound WebSocket events -----------------------------------------

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._server_task:
            try:
                await self._server_task
            except Exception as e:
                self.logger.warning("server task error during shutdown: {}", e)
            self._server_task = None
        self._subs.clear()
        self._conn_chats.clear()
        self._conn_default.clear()
        self._tokens.clear()

    async def _safe_send_to(self, connection: Any, raw: str, *, label: str = "") -> None:
        """Send a raw frame to one connection, cleaning up on ConnectionClosed."""
        try:
            await connection.send(raw)
        except ConnectionClosed:
            self._cleanup_connection(connection)
            self.logger.warning("connection gone{}", label)
        except Exception:
            self.logger.exception("send failed{}", label)
            raise

    async def send(self, msg: OutboundMessage) -> None:
        """通过 WebSocket 向订阅的客户端发送一条消息。

        【中文名称】发送出站消息

        【功能说明】
        这是 ChannelManager 分发 OutboundMessage 到 WebSocket 客户端的入口。
        与 Telegram 的 send() 不同，WebSocket 的 send 需要处理更丰富的事件类型：

        【事件分发 —— 7 种类型】
        _runtime_model_updated → send_runtime_model_updated（模型切换）
        _goal_state_sync → send_goal_state（长期任务状态同步）
        _goal_status → send_goal_status（任务运行状态）
        _turn_end → send_turn_end（回合结束，含 latency + goal_state）
        _session_updated → send_session_updated（工作区范围变更）
        _file_edit_events → send_file_edit_events（文件编辑进度）
        普通消息 → 构造 JSON payload 并群发给所有订阅者

        【普通消息的 JSON payload 结构】
        {
          "event": "message",
          "chat_id": "...",
          "text": "...",           // 文本内容（media URLs 已重写）
          "media_urls": [...],     // 签名后的媒体文件 URL
          "tool_events": [...],    // 工具调用事件
          "agent_ui": {...},       // Agent UI 组件数据
          "kind": "tool_hint" | "progress",  // 消息类型标记
          "latency_ms": 1234       // 延迟
        }

        【参数说明】
        - msg: OutboundMessage → 从 MessageBus 分发的出站消息
        """
        if msg.metadata.get("_runtime_model_updated"):
            await self.send_runtime_model_updated(
                model_name=msg.metadata.get("model"),
                model_preset=msg.metadata.get("model_preset"),
            )
            return

        # Snapshot the subscriber set so ConnectionClosed cleanups mid-iteration are safe.
        conns = list(self._subs.get(msg.chat_id, ()))
        if not conns:
            if (
                msg.metadata.get("_progress")
                or msg.metadata.get("_file_edit_events")
                or msg.metadata.get("_turn_end")
                or msg.metadata.get("_session_updated")
                or msg.metadata.get("_goal_status")
                or msg.metadata.get("_goal_state_sync")
            ):
                self.logger.debug("no active subscribers for chat_id={}", msg.chat_id)
            else:
                self.logger.warning("no active subscribers for chat_id={}", msg.chat_id)
        if msg.metadata.get("_goal_state_sync"):
            if conns:
                blob = msg.metadata.get("goal_state")
                await self.send_goal_state(msg.chat_id, blob if isinstance(blob, dict) else {"active": False})
            return
        if msg.metadata.get("_goal_status"):
            if conns:
                status = msg.metadata.get("goal_status")
                if status in ("running", "idle"):
                    started_raw = msg.metadata.get("started_at", msg.metadata.get("goal_started_at"))
                    await self.send_goal_status(
                        msg.chat_id,
                        status,
                        started_at=float(started_raw) if isinstance(started_raw, int | float) else None,
                    )
            return
        # Signal that the agent has fully finished processing the current turn.
        if msg.metadata.get("_turn_end"):
            lat = msg.metadata.get("latency_ms")
            lat_i = int(lat) if isinstance(lat, (int, float)) else None
            gs = msg.metadata.get("goal_state")
            gs_blob = gs if isinstance(gs, dict) else None
            await self.send_turn_end(
                msg.chat_id,
                latency_ms=lat_i,
                goal_state=gs_blob,
                metadata=msg.metadata,
            )
            return
        if msg.metadata.get("_session_updated"):
            if conns:
                scope = msg.metadata.get("_session_update_scope")
                await self.send_session_updated(
                    msg.chat_id,
                    scope=scope if isinstance(scope, str) else None,
                )
            return
        if msg.metadata.get("_file_edit_events"):
            edits = msg.metadata.get("_file_edit_events")
            await self.send_file_edit_events(
                msg.chat_id,
                edits if isinstance(edits, list) else [],
                msg.metadata,
            )
            return
        text = msg.content
        wire_text = self._media.rewrite_local_markdown_images(text)
        payload: dict[str, Any] = {
            "event": "message",
            "chat_id": msg.chat_id,
            "text": wire_text,
        }
        if msg.media:
            payload["media"] = msg.media
            urls: list[dict[str, str]] = []
            for entry in msg.media:
                signed = self._media.sign_or_stage_media_path(Path(entry))
                if signed is not None:
                    urls.append(signed)
            if urls:
                payload["media_urls"] = urls
        if msg.reply_to:
            payload["reply_to"] = msg.reply_to
        lat = msg.metadata.get("latency_ms")
        if isinstance(lat, (int, float)):
            payload["latency_ms"] = int(lat)
        if msg.metadata.get("_tool_events"):
            payload["tool_events"] = msg.metadata["_tool_events"]
        agent_ui = msg.metadata.get(OUTBOUND_META_AGENT_UI)
        if agent_ui is not None:
            payload["agent_ui"] = agent_ui
        # Mark intermediate agent breadcrumbs (tool-call hints, generic
        # progress strings) so WS clients can render them as subordinate
        # trace rows rather than conversational replies.
        if msg.metadata.get("_tool_hint"):
            payload["kind"] = "tool_hint"
        elif msg.metadata.get("_progress"):
            payload["kind"] = "progress"
        phase = "activity" if payload.get("kind") in ("tool_hint", "progress") else "answer"
        self._transcripts.prepare_and_append(
            msg.chat_id,
            payload,
            metadata=msg.metadata,
            phase=phase,
            include_source=True,
            transcript_overrides={"text": text},
        )
        raw = json.dumps(payload, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" ")

    async def send_reasoning_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Push one chunk of model reasoning. Mirrors ``send_delta`` shape so
        clients receive a stream that opens, updates in place, and closes —
        rendered above the active assistant bubble with a shimmer header
        until the matching ``reasoning_end`` arrives.
        """
        conns = list(self._subs.get(chat_id, ()))
        if not delta:
            return
        meta = metadata or {}
        body: dict[str, Any] = {
            "event": "reasoning_delta",
            "chat_id": chat_id,
            "text": delta,
        }
        stream_id = meta.get("_stream_id")
        if stream_id is not None:
            body["stream_id"] = stream_id
        self._transcripts.prepare_and_append(
            chat_id,
            body,
            metadata=meta,
            phase="reasoning",
        )
        raw = json.dumps(body, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" reasoning ")

    async def send_reasoning_end(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Close the current reasoning stream segment for in-place renderers."""
        conns = list(self._subs.get(chat_id, ()))
        meta = metadata or {}
        body: dict[str, Any] = {
            "event": "reasoning_end",
            "chat_id": chat_id,
        }
        stream_id = meta.get("_stream_id")
        if stream_id is not None:
            body["stream_id"] = stream_id
        self._transcripts.prepare_and_append(
            chat_id,
            body,
            metadata=meta,
            phase="reasoning",
        )
        raw = json.dumps(body, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" reasoning_end ")

    async def send_file_edit_events(
        self,
        chat_id: str,
        edits: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conns = list(self._subs.get(chat_id, ()))
        payload: dict[str, Any] = {
            "event": "file_edit",
            "chat_id": chat_id,
            "edits": edits,
        }
        self._transcripts.prepare_and_append(
            chat_id,
            payload,
            metadata=metadata,
            phase="activity",
        )
        raw = json.dumps(payload, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" file_edit ")

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """向 WebSocket 客户端推送流式增量文本。

        【中文名称】流式增量发送

        【功能说明】
        与 Telegram 的 send_delta 对应，但 WebSocket 不需要编辑已有消息——
        客户端自行组装增量文本。工作方式：

        【流程】
        1. 非 _stream_end：发送 "delta" 事件，将 delta 追加到内存缓冲区
        2. _stream_end：发送 "stream_end" 事件，拼接缓冲区中的所有 delta，
           通过 _media.rewrite_local_markdown_images 重写图片 URL，
           把最终文本一起发给客户端

        【参数说明】
        - chat_id: str → 会话 ID
        - delta: str → 增量文本片段
        - metadata: dict → 包含 _stream_id（不同流标识）、_stream_end（流结束标记）
        """
        conns = list(self._subs.get(chat_id, ()))
        meta = metadata or {}
        stream_key = (chat_id, str(meta.get("_stream_id") or ""))
        if meta.get("_stream_end"):
            body: dict[str, Any] = {"event": "stream_end", "chat_id": chat_id}
            buffered = self._stream_text_buffers.pop(stream_key, [])
            if delta:
                buffered.append(delta)
            full_text = "".join(buffered)
            rewritten = self._media.rewrite_local_markdown_images(full_text)
            if delta or rewritten != full_text:
                body["text"] = rewritten
        else:
            body = {
                "event": "delta",
                "chat_id": chat_id,
                "text": delta,
            }
            self._stream_text_buffers.setdefault(stream_key, []).append(delta)
        if meta.get("_stream_id") is not None:
            body["stream_id"] = meta["_stream_id"]
        self._transcripts.prepare_and_append(
            chat_id,
            body,
            metadata=meta,
            phase="answer",
        )
        raw = json.dumps(body, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" stream ")

    async def send_turn_end(
        self,
        chat_id: str,
        latency_ms: int | None = None,
        *,
        goal_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Signal that the agent has fully finished processing the current turn."""
        conns = list(self._subs.get(chat_id, ()))
        body: dict[str, Any] = {"event": "turn_end", "chat_id": chat_id}
        if latency_ms is not None:
            body["latency_ms"] = int(latency_ms)
        if goal_state is not None:
            body["goal_state"] = goal_state
        self._transcripts.prepare_and_append(
            chat_id,
            body,
            metadata=metadata,
            phase="complete",
        )
        raw = json.dumps(body, ensure_ascii=False)
        if not conns:
            return
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" turn_end ")

    async def send_goal_state(self, chat_id: str, blob: dict[str, Any]) -> None:
        """Push persisted goal-state snapshot for *chat_id* (multi-chat isolation)."""
        conns = list(self._subs.get(chat_id, ()))
        if not conns:
            return
        body = {"event": "goal_state", "chat_id": chat_id, "goal_state": blob}
        raw = json.dumps(body, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" goal_state ")

    async def send_goal_status(
        self,
        chat_id: str,
        status: str,
        *,
        started_at: float | None = None,
    ) -> None:
        """Notify subscribed clients that a turn started or finished (wall-clock hint)."""
        conns = list(self._subs.get(chat_id, ()))
        if not conns:
            return
        body: dict[str, Any] = {
            "event": "goal_status",
            "chat_id": chat_id,
            "status": status,
        }
        if status == "running" and started_at is not None:
            body["started_at"] = started_at
        raw = json.dumps(body, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" goal_status ")

    async def send_session_updated(self, chat_id: str, *, scope: str | None = None) -> None:
        """Notify clients that session metadata changed outside the main turn."""
        conns = list(self._subs.get(chat_id, ()))
        if not conns:
            return
        body: dict[str, Any] = {"event": "session_updated", "chat_id": chat_id}
        if scope:
            body["scope"] = scope
        raw = json.dumps(body, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" session_updated ")

    async def send_runtime_model_updated(
        self,
        *,
        model_name: Any,
        model_preset: Any = None,
    ) -> None:
        """Broadcast runtime model changes to every open websocket connection."""
        conns = list(self._conn_chats)
        if not conns or not isinstance(model_name, str) or not model_name.strip():
            return
        body: dict[str, Any] = {
            "event": "runtime_model_updated",
            "model_name": model_name.strip(),
        }
        if isinstance(model_preset, str) and model_preset.strip():
            body["model_preset"] = model_preset.strip()
        raw = json.dumps(body, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" runtime_model_updated ")
