"""异步消息队列：用于解耦渠道层与 Agent 核心。

这个文件实现的是一个非常轻量的“双队列总线”：

- ``inbound``：渠道 -> Agent
- ``outbound``：Agent -> 渠道

好处是渠道适配器不需要直接调用 Agent 内部逻辑，Agent 也不需要知道外部平台
SDK 的细节。两边只通过统一消息结构和异步队列通信。
"""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """异步消息总线。

    【核心思想】
    把“接收消息”和“处理消息”拆成两个独立环节：

    1. 渠道收到消息后，只负责把 ``InboundMessage`` 放进 ``inbound`` 队列
    2. AgentLoop 从 ``inbound`` 队列消费消息并处理
    3. AgentLoop 处理完后，把 ``OutboundMessage`` 放进 ``outbound`` 队列
    4. ChannelManager 再从 ``outbound`` 队列取出并发送到对应平台

    这种设计是整个项目数据流的第一层骨架。
    """

    def __init__(self):
        # 入站队列：渠道层把收到的用户消息放到这里，等待 AgentLoop 消费。
        """init。
        
        初始化 MessageBus 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        # 出站队列：AgentLoop 把生成的回复放到这里，等待 ChannelManager 发送。
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """发布一条入站消息：渠道把消息投递给 Agent。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """消费下一条入站消息；如果队列为空会一直等待。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """发布一条出站消息：Agent 把回复交还给渠道层。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """消费下一条出站消息；如果队列为空会一直等待。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """当前等待处理的入站消息数量。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """当前等待发送的出站消息数量。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return self.outbound.qsize()
