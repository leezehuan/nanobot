"""渠道管理器：统一协调所有聊天渠道。

如果说 ``MessageBus`` 负责“消息流转的骨架”，那 ``ChannelManager`` 负责的就是：

- 启动哪些渠道
- 出站消息该发给哪个渠道
- 是否需要重试发送
- 流式增量消息如何合并，减少 API 调用
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Config
from nanobot.utils.restart import consume_restart_notice_from_env, format_restart_completed_message

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager


def _default_webui_dist() -> Path | None:
    """返回打包后 WebUI dist 目录的绝对路径；不存在则返回 ``None``。"""
    try:
        import nanobot.web as web_pkg  # type: ignore[import-not-found]
    except ImportError:
        return None
    candidate = Path(web_pkg.__file__).resolve().parent / "dist"
    return candidate if candidate.is_dir() else None


# 发送失败后的重试等待时间，采用指数退避：1s -> 2s -> 4s
_SEND_RETRY_DELAYS = (1, 2, 4)

_BOOL_CAMEL_ALIASES: dict[str, str] = {
    "send_progress": "sendProgress",
    "send_tool_hints": "sendToolHints",
    "show_reasoning": "showReasoning",
}

class ChannelManager:
    """聊天渠道总协调器。

    它是“渠道层的总控台”，核心职责有三类：
    - 初始化已启用渠道
    - 启动/停止所有渠道
    - 从 ``bus.outbound`` 消费消息并路由到目标渠道
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        *,
        session_manager: "SessionManager | None" = None,
        cron_service: Any | None = None,
        webui_runtime_model_name: Callable[[], str | None] | None = None,
        webui_static_dist: bool = True,
        webui_runtime_surface: str = "browser",
        webui_runtime_capabilities: dict[str, Any] | None = None,
    ):
        self.config = config
        self.bus = bus
        self._session_manager = session_manager
        self._cron_service = cron_service
        self._webui_runtime_model_name = webui_runtime_model_name
        self._webui_static_dist = webui_static_dist
        self._webui_runtime_surface = webui_runtime_surface
        self._webui_runtime_capabilities = dict(webui_runtime_capabilities or {})
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._origin_reply_fingerprints: dict[tuple[str, str, str], str] = {}

        self._init_channels()

    def _init_channels(self) -> None:
        """初始化已启用渠道。

        发现来源既包括内置模块扫描，也包括 entry_points 插件。
        但真正实例化时只导入配置里启用的渠道，避免无谓依赖开销。
        """
        from nanobot.channels.registry import discover_channel_names, discover_enabled

        # 先收集“候选渠道名”，再只导入真正启用的那些。
        # 这样做可以避免未启用渠道的重依赖提前 import。
        names = discover_channel_names()
        candidate_names = set(names)
        extra = getattr(self.config.channels, "__pydantic_extra__", None) or {}
        candidate_names.update(extra.keys())

        enabled_names: set[str] = set()
        for name in candidate_names:
            section = getattr(self.config.channels, name, None)
            if section is None:
                continue
            if (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            ):
                enabled_names.add(name)

        for name, cls in discover_enabled(enabled_names, _names=names).items():
            section = getattr(self.config.channels, name, None)
            if section is None:
                continue
            try:
                kwargs: dict[str, Any] = {}
                if cls.name == "websocket":
                    # websocket 渠道比较特殊：它不只是“聊天渠道”，还会承载
                    # WebUI 网关能力，所以需要额外组装 gateway 服务对象。
                    from nanobot.channels.websocket import WebSocketConfig
                    from nanobot.webui.gateway_services import build_gateway_services

                    parsed = WebSocketConfig.model_validate(section)
                    static_path = _default_webui_dist() if self._webui_static_dist else None
                    workspace = Path(self.config.workspace_path)
                    gateway = build_gateway_services(
                        config=parsed,
                        bus=self.bus,
                        session_manager=self._session_manager,
                        static_dist_path=static_path,
                        workspace_path=workspace,
                        default_restrict_to_workspace=self.config.tools.restrict_to_workspace,
                        disabled_skills=set(self.config.agents.defaults.disabled_skills),
                        runtime_model_name=self._webui_runtime_model_name,
                        runtime_surface=self._webui_runtime_surface,
                        runtime_capabilities_overrides=self._webui_runtime_capabilities,
                        cron_service=self._cron_service,
                        logger=logger,
                    )
                    kwargs["gateway"] = gateway
                channel = cls(section, self.bus, **kwargs)
                channel.send_progress = self._resolve_bool_override(
                    section, "send_progress", self.config.channels.send_progress,
                )
                channel.send_tool_hints = self._resolve_bool_override(
                    section, "send_tool_hints", self.config.channels.send_tool_hints,
                )
                channel.show_reasoning = self._resolve_bool_override(
                    section, "show_reasoning", self.config.channels.show_reasoning,
                )
                self.channels[name] = channel
                logger.info("{} channel enabled", cls.display_name)
            except Exception as e:
                logger.warning("{} channel not available: {}", name, e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        """检查渠道的 allowFrom 配置，并给出 pairing 模式提示。"""
        for name, ch in self.channels.items():
            cfg = ch.config
            if isinstance(cfg, dict):
                if "allow_from" in cfg:
                    allow = cfg.get("allow_from")
                else:
                    allow = cfg.get("allowFrom")
            else:
                allow = getattr(cfg, "allow_from", None)
            if allow is None:
                # 如果没配置 allowFrom，就进入“仅配对码授权”模式。
                # 未授权用户不会被静默忽略，而是会收到 pairing code。
                logger.info(
                    '"{}" has no allowFrom; unapproved users will receive a pairing code',
                    name,
                )

    def _should_send_progress(self, channel_name: str, *, tool_hint: bool = False) -> bool:
        """判断某个渠道是否允许发送进度消息或工具提示。"""
        ch = self.channels.get(channel_name)
        if ch is None:
            logger.warning("Progress check for unknown channel: {}", channel_name)
            return False
        return ch.send_tool_hints if tool_hint else ch.send_progress

    def _resolve_bool_override(self, section: Any, key: str, default: bool) -> bool:
        """从渠道配置里读取布尔开关，读不到就回退默认值。

        同时兼容 snake_case 和 camelCase，方便直接读取原始 JSON/TOML 配置。
        """
        if isinstance(section, dict):
            value = section.get(key)
            if value is None:
                camel = _BOOL_CAMEL_ALIASES.get(key)
                if camel:
                    value = section.get(camel)
            return value if isinstance(value, bool) else default
        value = getattr(section, key, None)
        return value if isinstance(value, bool) else default

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """启动单个渠道，并把异常记录下来。"""
        try:
            await channel.start()
        except Exception:
            logger.exception("Failed to start channel {}", name)

    async def start_all(self) -> None:
        """启动所有渠道，以及统一的出站消息分发协程。"""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # 先启动统一出站分发器，再启动各个具体渠道。
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # 再启动各个具体渠道。
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        self._notify_restart_done_if_needed()

        # 渠道通常都是常驻任务，因此这里会一直等待。
        await asyncio.gather(*tasks, return_exceptions=True)

    def _notify_restart_done_if_needed(self) -> None:
        """如果环境变量里带着“重启通知标记”，则向对应聊天发送重启完成提示。"""
        notice = consume_restart_notice_from_env()
        if not notice:
            return
        target = self.channels.get(notice.channel)
        if not target:
            return
        asyncio.create_task(self._send_with_retry(
            target,
            OutboundMessage(
                channel=notice.channel,
                chat_id=notice.chat_id,
                content=format_restart_completed_message(notice.started_at_raw),
                metadata=dict(notice.metadata or {}),
            ),
        ))

    async def stop_all(self) -> None:
        """停止所有渠道以及出站分发器。"""
        logger.info("Stopping all channels...")

        # 先停统一出站分发器。
        if self._dispatch_task:
            self._dispatch_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatch_task

        # 再依次停止所有渠道实例。
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception:
                logger.exception("Error stopping {}", name)

    @staticmethod
    def _fingerprint_content(content: str) -> str:
        """为消息正文生成去空白后的稳定指纹，用于重复发送抑制。"""
        normalized = " ".join(content.split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def _should_suppress_outbound(self, msg: OutboundMessage) -> bool:
        """判断某条出站消息是否应因“与已发送内容重复”而被抑制。"""
        metadata = msg.metadata or {}
        if metadata.get("_progress"):
            return False
        fingerprint = self._fingerprint_content(msg.content)
        if not fingerprint:
            return False

        origin_message_id = metadata.get("origin_message_id")
        if isinstance(origin_message_id, str) and origin_message_id:
            key = (msg.channel, msg.chat_id, origin_message_id)
            if self._origin_reply_fingerprints.get(key) == fingerprint:
                return True
            self._origin_reply_fingerprints[key] = fingerprint

        message_id = metadata.get("message_id")
        if isinstance(message_id, str) and message_id:
            key = (msg.channel, msg.chat_id, message_id)
            self._origin_reply_fingerprints[key] = fingerprint

        return False

    async def _dispatch_outbound(self) -> None:
        """持续消费出站队列，并把消息路由到对应渠道。

        【中文名称】出站消息分发器

        【功能说明】
        这是 ChannelManager 的"心跳协程"。它一边不断从 MessageBus.outbound
        队列消费出站消息，一边根据消息类型和元数据标记做路由和去重。

        【消息分类处理（按 metadata 标记）】
        1. _reasoning_delta / _reasoning_end / _reasoning：
           走独立的 reasoning 发送通道（send_reasoning_delta / send_reasoning_end）
           只有目标渠道开启了 show_reasoning 才真正发送

        2. _progress / _tool_hint：
           根据渠道的 send_progress / send_tool_hints 开关决定是否发送

        3. _retry_wait：
           静默跳过（这只是重试等待心跳通知，不是用户可见消息）

        4. _stream_delta（非 _stream_end）：
           流式增量合并：用 _coalesce_stream_deltas() 把连续的流式 delta
           合并成一条大消息再发送，减少渠道 API 调用次数

        5. 普通消息：
           先做重复抑制检查（_should_suppress_outbound），
           再通过 _send_with_retry() （带指数退避重试）发送

        【去重机制】
        对同一 (channel, chat_id, origin_message_id) 组合，如果内容指纹相同，
        则被视为重复消息并抑制发送。这主要防止某些场景下同一条回复被多次生成。

        【重试策略】
        _send_with_retry() 使用指数退避（1s → 2s → 4s），最多重试次数
        由 channels.send_max_retries 配置（默认 3）。

        【参数说明】
        无 —— 这是常驻协程，通过 self.bus 拿到队列引用。

        【返回值】
        无 —— 这是一个常驻协程，进程退出时才会结束。
        """
        logger.info("Outbound dispatcher started")

        # 因为 asyncio.Queue 没有 push_front，所以在流式合并过程中如果多拿了
        # 一条不该当前处理的消息，只能先暂存到这个本地缓冲里。
        pending: list[OutboundMessage] = []

        while True:
            try:
                # 先处理本地缓冲，再去队列阻塞等待新消息。
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await asyncio.wait_for(
                        self.bus.consume_outbound(),
                        timeout=1.0
                    )

                if (
                    msg.metadata.get("_reasoning_delta")
                    or msg.metadata.get("_reasoning_end")
                    or msg.metadata.get("_reasoning")
                ):
                    # reasoning 有自己独立的一条展示通道：
                    # 只有目标渠道明确支持 show_reasoning，且实现了对应接口，
                    # 才会真正发送；否则就静默跳过。
                    channel = self.channels.get(msg.channel)
                    if channel is not None and channel.show_reasoning:
                        await self._send_with_retry(channel, msg)
                    continue

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self._should_send_progress(
                        msg.channel, tool_hint=True,
                    ):
                        continue
                    if not msg.metadata.get("_tool_hint") and not self._should_send_progress(
                        msg.channel, tool_hint=False,
                    ):
                        continue

                if msg.metadata.get("_retry_wait"):
                    continue

                if (
                    msg.metadata.get("_runtime_model_updated")
                    and msg.channel == "websocket"
                    and "websocket" not in self.channels
                ):
                    continue

                # 合并连续的流式增量片段，减少平台 API 调用次数，
                # 也减少“队列里积压很多小碎片消息”带来的延迟。
                if msg.metadata.get("_stream_delta") and not msg.metadata.get("_stream_end"):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)

                channel = self.channels.get(msg.channel)
                if channel:
                    # 去重只在“同一来源消息”范围内生效，
                    # 不会误伤来自不同 turn 但内容恰好相同的回复。
                    if (
                        not msg.metadata.get("_stream_delta")
                        and not msg.metadata.get("_stream_end")
                        and not msg.metadata.get("_streamed")
                    ):
                        if self._should_suppress_outbound(msg):
                            logger.info("Suppressing duplicate outbound message to {}:{}", msg.channel, msg.chat_id)
                            continue
                    await self._send_with_retry(channel, msg)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        """单次发送一条消息；这里不处理重试策略。"""
        if msg.metadata.get("_reasoning_end"):
            await channel.send_reasoning_end(msg.chat_id, msg.metadata)
        elif msg.metadata.get("_reasoning_delta"):
            await channel.send_reasoning_delta(msg.chat_id, msg.content, msg.metadata)
        elif msg.metadata.get("_reasoning"):
            # 向后兼容：老路径可能一次性发送完整 reasoning。
            # BaseChannel 会把它翻译成“一条 delta + 一条 end”，
            # 这样插件只实现流式原语即可。
            await channel.send_reasoning(msg)
        elif msg.metadata.get("_file_edit_events"):
            edits = msg.metadata.get("_file_edit_events")
            await channel.send_file_edit_events(
                msg.chat_id,
                edits if isinstance(edits, list) else [],
                msg.metadata,
            )
        elif msg.metadata.get("_stream_delta") or msg.metadata.get("_stream_end"):
            await channel.send_delta(msg.chat_id, msg.content, msg.metadata)
        elif not msg.metadata.get("_streamed"):
            await channel.send(msg)

    def _coalesce_stream_deltas(
        self, first_msg: OutboundMessage
    ) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """合并同一目标的连续流式增量消息。

        典型场景是：LLM 输出很快，但渠道发送较慢，队列里会积压很多碎片 delta。
        这时把它们批量拼起来再发，会明显减少平台请求次数。
        """
        target_key = (first_msg.channel, first_msg.chat_id)
        combined_content = first_msg.content
        final_metadata = dict(first_msg.metadata or {})
        non_matching: list[OutboundMessage] = []

        # 只合并“连续出现”的 delta。
        # 一旦遇到其他类型消息，就立即停止，并把边界消息交回 pending。
        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            # 判断下一条消息是否仍属于同一条流。
            same_target = (next_msg.channel, next_msg.chat_id) == target_key
            is_delta = next_msg.metadata and next_msg.metadata.get("_stream_delta")
            is_end = next_msg.metadata and next_msg.metadata.get("_stream_end")

            if same_target and is_delta and not final_metadata.get("_stream_end"):
                # 继续把内容拼接进当前批次。
                combined_content += next_msg.content
                # 如果已经看到流结束标记，就把结束状态带上并停止合并。
                if is_end:
                    final_metadata["_stream_end"] = True
                    # 这一条流已经结束，不再继续合并后续消息。
                    break
            else:
                # 第一条不匹配消息定义了当前合并边界。
                non_matching.append(next_msg)
                break

        merged = OutboundMessage(
            channel=first_msg.channel,
            chat_id=first_msg.chat_id,
            content=combined_content,
            metadata=final_metadata,
        )
        return merged, non_matching

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """带指数退避重试地发送消息。

        这里统一实现重试策略，渠道本身只要在失败时抛异常即可。
        注意 ``CancelledError`` 必须继续上抛，确保程序停机时能优雅退出。
        """
        max_attempts = max(self.config.channels.send_max_retries, 1)

        for attempt in range(max_attempts):
            try:
                await self._send_once(channel, msg)
                return  # Send succeeded
            except asyncio.CancelledError:
                raise  # Propagate cancellation for graceful shutdown
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.exception(
                        "Failed to send to {} after {} attempts",
                        msg.channel, max_attempts
                    )
                    return
                delay = _SEND_RETRY_DELAYS[min(attempt, len(_SEND_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Send to {} failed (attempt {}/{}): {}, retrying in {}s",
                    msg.channel, attempt + 1, max_attempts, type(e).__name__, delay
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise  # Propagate cancellation during sleep

    def get_channel(self, name: str) -> BaseChannel | None:
        """按名字获取渠道实例。"""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """返回所有渠道的启用与运行状态。"""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """返回已启用渠道名列表。"""
        return list(self.channels.keys())
