"""消息发送工具：让 Agent 主动给用户或其他会话发消息。

它和“普通回复当前用户”不是一回事：

- 普通回复：主 Agent 直接输出最终答案，由主流程回到当前聊天
- ``message`` 工具：额外主动投递一条消息，常用于提醒、跨会话发送、附件分发

所以如果只是回答当前问题，通常不该调用这个工具。
"""

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.bus.events import OutboundMessage
from nanobot.config.paths import get_workspace_path


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema(
            "Message content for proactive or cross-channel delivery. "
            "Do not use this for a normal reply in the current chat."
        ),
        channel=StringSchema(
            "Optional target channel for cross-channel/proactive delivery. "
            "Do not set this to the current runtime channel for a normal reply."
        ),
        chat_id=StringSchema(
            "Optional target chat/user ID for cross-channel/proactive delivery. "
            "On WebSocket/WebUI turns: omit chat_id to use the server's conversation id "
            "(never pass client_id values like anon-…). "
            "Do not set this to the current runtime chat for a normal reply."
        ),
        media=ArraySchema(
            StringSchema(""),
            description=(
                "Optional list of existing file paths to attach. "
                "Use artifact paths returned by generate_image here when delivering generated images."
            ),
        ),
        buttons=ArraySchema(
            ArraySchema(StringSchema("Button label")),
            description="Optional: inline keyboard buttons as list of rows, each row is list of button labels.",
        ),
        required=["content"],
    )
)
class MessageTool(Tool, ContextAware):
    """主动消息发送工具。"""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        workspace: str | Path | None = None,
        restrict_to_workspace: bool = False,
    ):
        """init。
        
        初始化 MessageTool 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._send_callback = send_callback
        self._workspace = (
            Path(workspace).expanduser() if workspace is not None else get_workspace_path()
        )
        self._restrict_to_workspace = restrict_to_workspace
        self._default_channel: ContextVar[str] = ContextVar(
            "message_default_channel", default=default_channel
        )
        self._default_chat_id: ContextVar[str] = ContextVar(
            "message_default_chat_id", default=default_chat_id
        )
        self._default_message_id: ContextVar[str | None] = ContextVar(
            "message_default_message_id",
            default=default_message_id,
        )
        self._default_metadata: ContextVar[dict[str, Any]] = ContextVar(
            "message_default_metadata",
            default={},
        )
        self._sent_in_turn_var: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
        self._turn_delivered_media_var: ContextVar[tuple[str, ...]] = ContextVar(
            "message_turn_delivered_media",
            default=(),
        )
        self._record_channel_delivery_var: ContextVar[bool] = ContextVar(
            "message_record_channel_delivery",
            default=False,
        )
        self._suppress_delivery_var: ContextVar[bool] = ContextVar(
            "message_suppress_delivery",
            default=False,
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        """create。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        send_callback = ctx.bus.publish_outbound if ctx.bus else None
        return cls(
            send_callback=send_callback,
            workspace=ctx.workspace,
            restrict_to_workspace=ctx.config.restrict_to_workspace,
        )

    def set_context(self, ctx: RequestContext) -> None:
        """记录当前请求上下文，作为默认发送目标。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        self._default_channel.set(ctx.channel)
        self._default_chat_id.set(ctx.chat_id)
        self._default_message_id.set(ctx.message_id)
        self._default_metadata.set(dict(ctx.metadata or {}))

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """注入真正执行发送的回调函数。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        self._send_callback = callback

    def start_turn(self) -> None:
        """开始新 turn 时重置本轮发送痕迹。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        self._sent_in_turn = False
        self._turn_delivered_media_var.set(())

    def turn_delivered_media_paths(self) -> list[str]:
        """返回本轮通过该工具发到当前聊天的附件绝对路径。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return list(self._turn_delivered_media_var.get())

    def set_record_channel_delivery(self, active: bool):
        """开启/关闭“记录为主动渠道投递”的状态。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        return self._record_channel_delivery_var.set(active)

    def reset_record_channel_delivery(self, token) -> None:
        """恢复之前的主动投递记录状态。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        self._record_channel_delivery_var.reset(token)

    def set_suppress_delivery(self, active: bool):
        """让工具“确认发送但不真正投递”，供内部探活/自检使用。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        return self._suppress_delivery_var.set(active)

    def reset_suppress_delivery(self, token) -> None:
        """恢复之前的抑制发送状态。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        self._suppress_delivery_var.reset(token)

    @property
    def _sent_in_turn(self) -> bool:
        """sent in turn。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return self._sent_in_turn_var.get()

    @_sent_in_turn.setter
    def _sent_in_turn(self, value: bool) -> None:
        """sent in turn。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        self._sent_in_turn_var.set(value)

    @property
    def name(self) -> str:
        """name。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return "message"

    @property
    def description(self) -> str:
        """description。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return (
            "Proactively send a message to a user/channel, optionally with file attachments. "
            "Use this for reminders, cross-channel delivery, or explicit proactive sends. "
            "Do not use this for the normal reply in the current chat: answer naturally instead. "
            "If channel/chat_id would target the current runtime conversation, do not call this tool "
            "unless the user explicitly asked you to proactively send an existing file attachment. "
            "When generate_image creates images in the current chat, use the message tool "
            "with the artifact paths in the media parameter to deliver the images to the user. "
            "For proactive attachment delivery, use the 'media' parameter with file paths. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    def _resolve_media(self, media: list[str]) -> list[str]:
        """解析附件路径，并在需要时执行工作区边界检查。
        
        实现方法：先规范化名字或路径，再结合配置、预设和默认值得到最终可执行对象。"""
        resolved: list[str] = []
        access = current_tool_workspace(
            self._workspace,
            restrict_to_workspace=self._restrict_to_workspace,
        )
        workspace = access.project_path or self._workspace
        for p in media:
            if p.startswith(("http://", "https://")):
                # 远程 URL 直接透传，交给后续渠道层处理。
                resolved.append(p)
            elif not access.restrict_to_workspace:
                # 没有限制工作区时，相对路径默认基于当前 workspace 解释。
                path = Path(p).expanduser()
                resolved.append(p if path.is_absolute() else str(workspace / path))
            else:
                # 开启工作区约束时，必须保证附件仍在允许范围内。
                resolved.append(str(resolve_workspace_path(p, workspace, access.allowed_root)))
        return resolved

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """execute。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        from nanobot.utils.helpers import strip_think

        # 任何发给用户的内容都不应该包含内部 think 痕迹。
        content = strip_think(content)

        if buttons is not None:
            if not isinstance(buttons, list) or any(
                not isinstance(row, list) or any(not isinstance(label, str) for label in row)
                for row in buttons
            ):
                return "Error: buttons must be a list of list of strings"
        default_channel = self._default_channel.get()
        default_chat_id = self._default_chat_id.get()
        channel = channel or default_channel
        explicit_chat_id = chat_id
        if (
            default_channel == "websocket"
            and channel == "websocket"
            and explicit_chat_id is not None
            and str(explicit_chat_id).strip() != ""
            and str(explicit_chat_id).strip() != str(default_chat_id).strip()
        ):
            return (
                "Error: chat_id does not match the active WebSocket conversation. "
                "Omit chat_id (and usually channel) so delivery uses the current "
                "conversation id from context — WebSocket client_id strings "
                "(e.g. anon-…) are not chat ids."
            )
        chat_id = chat_id or default_chat_id
        # 只有目标还是“当前这一聊天”时，才允许继承默认 message_id。
        # 否则跨聊天投递若错误带上原 message_id，某些平台会把它当 reply 线索，
        # 最终把消息路由到错误会话。
        same_target = channel == default_channel and chat_id == default_chat_id
        if same_target:
            message_id = message_id or self._default_message_id.get()
        else:
            message_id = None

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        if media:
            try:
                media = self._resolve_media(media)
            except (OSError, PermissionError, ValueError) as e:
                return f"Error: media path is not allowed: {str(e)}"

        metadata = dict(self._default_metadata.get()) if same_target else {}
        if message_id:
            metadata["message_id"] = message_id
        if self._record_channel_delivery_var.get() or media:
            metadata["_record_channel_delivery"] = True

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            buttons=buttons or [],
            metadata=metadata,
        )

        if self._suppress_delivery_var.get():
            logger.debug("MessageTool: delivery suppressed during internal check")
            return f"Message acknowledged for {channel}:{chat_id} (not delivered)"

        try:
            await self._send_callback(msg)
            if channel == default_channel and chat_id == default_chat_id:
                # 只对当前聊天记录“本轮已主动发送过内容”，
                # 这样上层逻辑才能知道本轮是否额外投递过消息/附件。
                self._sent_in_turn = True
                if media:
                    prev = self._turn_delivered_media_var.get()
                    self._turn_delivered_media_var.set(prev + tuple(str(p) for p in media))
            media_info = f" with {len(media)} attachments" if media else ""
            button_info = f" with {sum(len(row) for row in buttons)} button(s)" if buttons else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{button_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
