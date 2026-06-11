"""聊天渠道抽象基类。

nanobot 支持多个聊天平台，但 Agent 核心不应该关心 Telegram、Discord、
Slack 各自 SDK 的差异。所以这里定义一个统一渠道接口，让所有平台适配器
都按同一套协议工作。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.pairing import (
    PAIRING_CODE_META_KEY,
    format_pairing_reply,
    generate_code,
    is_approved,
)


class BaseChannel(ABC):
    """聊天渠道抽象基类。

    每个具体渠道（Telegram、Discord、WebSocket 等）都应该继承它，并实现：
    - 如何启动监听
    - 如何停止
    - 如何发送消息
    """

    name: str = "base"
    display_name: str = "Base"
    send_progress: bool = True
    send_tool_hints: bool = False
    show_reasoning: bool = True

    def __init__(self, config: Any, bus: MessageBus):
        """初始化渠道实例。"""
        self.config = config
        self.logger = logger.bind(channel=self.name)
        self.bus = bus
        self._running = False

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """把音频转成文本；失败时返回空字符串。

        这里是渠道层的一个通用辅助能力，目的是让不同渠道在收到语音时都能复用
        同一套转写逻辑，而不用各自重复实现。
        """
        try:
            from nanobot.audio.transcription import (
                resolve_transcription_config,
                transcribe_audio_file,
            )
            from nanobot.config.loader import load_config

            return await transcribe_audio_file(file_path, resolve_transcription_config(load_config()))
        except Exception:
            self.logger.exception("Audio transcription failed")
            return ""

    async def login(self, force: bool = False) -> bool:
        """执行渠道自己的交互式登录流程，例如扫码登录。

        默认实现直接返回成功；只有需要显式登录的渠道才需要重写。
        """
        return True

    @abstractmethod
    async def start(self) -> None:
        """启动渠道并开始监听消息。

        一个典型实现会做三件事：
        1. 连接平台
        2. 监听消息事件
        3. 在收到消息时调用 ``_handle_message()`` 投递到消息总线
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道并释放相关资源。

        典型清理动作包括：
        - 断开网络连接
        - 停止后台任务
        - 关闭 SDK client / session
        """
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """通过当前渠道发送一条完整消息。

        约定：如果发送失败，子类应该抛异常，而不是静默吞掉。
        这样 ChannelManager 才能在统一位置做重试策略。
        """
        pass

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """发送一段流式文本增量。

        如果某个渠道支持“边生成边显示”，就重写这个方法。
        流式协议约定：
        - ``_stream_delta``：表示这是一个增量片段
        - ``_stream_end``：表示当前流片段结束
        - 有状态实现应优先按 ``_stream_id`` 缓冲，而不是只按 ``chat_id``
        """
        pass

    async def send_reasoning_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """发送一段模型 reasoning / thinking 的流式内容。

        默认什么也不做。只有支持“弱强调展示推理痕迹”的渠道才需要重写。
        它的流式协议与 ``send_delta`` 类似，只是键名换成 reasoning 版本。
        """
        return

    async def send_reasoning_end(
        self, chat_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """标记当前 reasoning 流片段结束。"""
        return

    async def send_file_edit_events(
        self,
        chat_id: str,
        edits: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """发送结构化文件编辑事件。

        富交互渠道可以重写它，把“正在修改哪个文件、修改已结束/失败”等信息
        渲染成更友好的活动面板，而不是依赖空白文本消息。
        """
        return

    async def send_reasoning(self, msg: OutboundMessage) -> None:
        """发送完整 reasoning 块。

        默认实现会复用 ``send_reasoning_delta`` + ``send_reasoning_end``，
        这样插件只要实现流式接口，就能同时兼容“分段 reasoning”和“一次性 reasoning”。
        """
        if not msg.content:
            return
        meta = dict(msg.metadata or {})
        meta.setdefault("_reasoning_delta", True)
        await self.send_reasoning_delta(msg.chat_id, msg.content, meta)
        end_meta = dict(meta)
        end_meta.pop("_reasoning_delta", None)
        end_meta["_reasoning_end"] = True
        await self.send_reasoning_end(msg.chat_id, end_meta)

    @property
    def supports_streaming(self) -> bool:
        """判断当前渠道是否真正支持流式输出。

        条件必须同时满足：
        1. 配置里开启了 streaming
        2. 子类真的重写了 ``send_delta``
        """
        cfg = self.config
        streaming = cfg.get("streaming", False) if isinstance(cfg, dict) else getattr(cfg, "streaming", False)
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        """检查发送者是否有权限使用机器人。

        权限顺序是：
        1. ``*`` 全开放
        2. allowFrom 精确匹配
        3. 已完成 pairing 的用户
        4. 否则拒绝
        """
        if isinstance(self.config, dict):
            allow_list = self.config.get("allow_from") or self.config.get("allowFrom") or []
        else:
            allow_list = getattr(self.config, "allow_from", None) or []
        if "*" in allow_list:
            return True
        # allowFrom 里的条目被视为“不透明身份令牌”，这里只做精确匹配，不做模糊推断。
        if str(sender_id) in allow_list:
            return True
        if is_approved(self.name, str(sender_id)):
            return True
        return False

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        is_dm: bool = False,
    ) -> None:
        """处理渠道收到的一条消息。

        这是渠道层最关键的公共辅助方法。它负责：
        1. 做权限检查
        2. 对未授权私聊用户发送 pairing code
        3. 把合法消息包装成 ``InboundMessage`` 投递到总线
        """
        if not self.is_allowed(sender_id):
            if is_dm:
                code = generate_code(self.name, str(sender_id))
                await self.send(
                    OutboundMessage(
                        channel=self.name,
                        chat_id=str(chat_id),
                        content=format_pairing_reply(code),
                        metadata={PAIRING_CODE_META_KEY: code},
                    )
                )
                self.logger.info(
                    "Sent pairing code {} to sender {} in chat {}",
                    code, sender_id, chat_id,
                )
            else:
                self.logger.warning(
                    "Access denied for sender {}. "
                    "Add them to allowFrom list in config to grant access.",
                    sender_id,
                )
            return

        meta = metadata or {}
        if self.supports_streaming:
            # 告诉后续链路：这个渠道希望收到流式输出。
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )

        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """返回渠道默认配置，用于初始化配置文件时自动填充。"""
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        """判断渠道当前是否处于运行状态。"""
        return self._running
