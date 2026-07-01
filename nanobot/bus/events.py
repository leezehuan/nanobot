"""消息总线事件类型定义。

这个模块定义 nanobot 在“渠道层 <-> Agent 核心”之间传递的两类基础消息：

1. ``InboundMessage``：外部渠道收到用户输入后，投递给 Agent 的入站消息。
2. ``OutboundMessage``：Agent 处理完成后，准备发回渠道的出站消息。

可以把它理解成项目里最基础的“消息信封”定义。后续无论是 Telegram、
WebSocket，还是命令行直连，最终都会落到这里的统一结构上。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ``OutboundMessage.metadata`` 中可选的结构化 UI 负载键。
# 这里放的是“与具体渠道无关”的富文本/富交互数据，值需要能被 JSON 序列化。
# WebUI 之类的富客户端可以识别并渲染，其他渠道即使忽略也不会出错。
OUTBOUND_META_AGENT_UI = "_agent_ui"

# 仅供进程内渠道使用的内部入站元数据键。
# 它的作用不是表示“用户发来了一条消息”，而是让某些内部组件直接通知
# AgentLoop 更新运行时状态，而不经过普通用户会话。
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
RUNTIME_CONTROL_ACK = "_ack"
RUNTIME_CONTROL_MCP_RELOAD = "mcp_reload"


@dataclass
class InboundMessage:
    """来自聊天渠道的入站消息。

    【中文名称】入站消息

    【作用】
    这是渠道层发给 Agent 核心的统一消息模型。
    不同平台的原始事件格式差异很大，但进入 AgentLoop 之后，都会被整理成
    这个结构，后续逻辑就不必再关心“它原本来自哪个 SDK 的什么事件对象”。
    """

    channel: str  # channel = 渠道名，例如 telegram / discord / slack / websocket
    sender_id: str  # sender_id = 发送者唯一标识，通常是平台用户 ID
    chat_id: str  # chat_id = 会话/群聊/频道标识，用来区分消息落到哪个聊天空间
    content: str  # content = 用户发送的文本正文
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # media = 附件路径或媒体引用列表
    metadata: dict[str, Any] = field(default_factory=dict)  # metadata = 渠道私有附加信息
    session_key_override: str | None = None  # 可选：覆盖默认 session_key，用于线程级会话等场景

    @property
    def session_key(self) -> str:
        """返回会话唯一键。

        默认规则是 ``channel:chat_id``，也就是“同一渠道下同一个聊天空间共享一个会话”。
        如果某些平台支持更细粒度的线程/子会话，就可以通过
        ``session_key_override`` 显式覆盖。

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """准备发往聊天渠道的出站消息。

    【中文名称】出站消息

    【作用】
    这是 Agent 处理完成后要交回给渠道层的统一消息模型。
    ChannelManager 会根据 ``channel`` 把它路由到对应渠道实例，再由渠道实例
    调用平台 API 真实发送。

    ``metadata`` 常见用途包括：
    - 路由字段：例如 ``message_id``、``reply_to`` 等
    - 运行中标记：例如 ``_progress``、``_stream_delta``、``_reasoning_delta``
    - 富客户端专用负载：例如 ``OUTBOUND_META_AGENT_UI``

    普通渠道可以忽略自己不认识的元数据键，WebUI 之类的富客户端则可以利用这些
    键做更细的界面表现。
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)
