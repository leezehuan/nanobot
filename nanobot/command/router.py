"""命令路由表：斜杠命令的轻量分发系统。

【中文名称】命令路由器

【功能说明】
这是 nanobot 斜杠命令（如 /stop、/new、/memory）的核心分发层。
它按三层优先级依次匹配，保证关键控制命令不受普通命令锁的影响。

【三层分发架构】
Tier 1: priority —— 精确匹配 + 不加锁执行（/stop、/restart 等紧急控制命令）
Tier 2: exact   —— 精确匹配 + 加锁执行（/status、/memory 等常规命令）
Tier 3: prefix  —— 最长前缀优先匹配（如 "/team " 会匹配所有 /team xxx 的命令）

【为什么需要 priority 层】
当 Agent 正在执行工具调用或陷入无限循环时，普通命令需要等待 dispatch lock 释放。
但 /stop 必须立即中断。priority 层绕开锁直接执行，保证用户随时可以打断 Agent。

【完整分发流程】
1. 调用方先通过 is_priority() 检查是否为 priority 命令
2. 如果是 priority 命令，直接用 dispatch_priority() 执行（无锁）
3. 如果不是，先通过 is_dispatchable_command() 检查是否能分发
4. 确认可分发的命令进入 dispatch()，按 exact → prefix 顺序匹配
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.session.manager import Session

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


@dataclass
class CommandContext:
    """命令处理器的执行上下文。

    【中文名称】命令上下文

    【字段说明】
    - msg: 入站消息的原始事件对象，包含发送者 ID、频道来源、消息体等
    - session: 当前会话对象（可能为 None，例如用户还未创建 session 时执行 /new）
    - key: 会话标识符 key，用于定位历史记录和上下文
    - raw: 用户输入的原始命令文本（未做大小写归一），例如 "  /Stop please "
    - args: prefix 匹配后剩余的参数字符串，由 dispatch() 在匹配成功后填充
    - loop: AgentLoop 实例的弱引用，供命令 handler 触发新的 agent 轮次
    """

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: Any = None


class CommandRouter:
    """命令分发路由器。

    【中文名称】命令路由器

    【数据结构】
    内部维护三张表：

    1. _priority（dict[str, Handler]）
       - 优先级命令表，精确 key 匹配
       - 匹配时不加 dispatch lock，保证 /stop 能够立即生效

    2. _exact（dict[str, Handler]）
       - 精确命令表，key 为规范化后的小写命令名
       - 在 dispatch lock 保护下执行

    3. _prefix（list[tuple[str, Handler]]）
       - 前缀命令表，按前缀长度降序排列
       - 匹配时取第一个命中项（最长前缀优先）
       - 剩余文本写入 ctx.args

    【注册命令的完整流程】
    1. 命令 handler 在模块加载时调用 priority() / exact() / prefix() 注册
    2. prefix 注册时会自动按前缀长度降序重排，保证最长匹配优先
    3. 运行时，AgentLoop 先查 priority，再查 exact/prefix

    【命令分发时序图（以 /stop 为例）】
    AgentLoop.run()
      → router.is_priority("/stop")          # 返回 True，不加锁
      → router.dispatch_priority(ctx)        # 直接执行 stop handler
      → handler 返回 OutboundMessage         # 发送"已停止"回复

    【命令分发时序图（以 /status 为例）】
    AgentLoop.run()
      → router.is_priority("/status")         # 返回 False
      → 加 dispatch lock
      → router.dispatch(ctx)                  # exact 匹配 → 执行 status handler
      → handler 返回 OutboundMessage          # 发送状态信息
      → 释放 dispatch lock
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []

    def priority(self, cmd: str, handler: Handler) -> None:
        """注册一个优先级命令（不加锁执行）。

        【参数说明】
        - cmd: 命令名（如 "stop"、"restart"），会做小写归一化后存为 key
        - handler: 异步处理函数，签名 (CommandContext) -> OutboundMessage | None

        【使用场景】
        只用于 /stop、/restart 这类需要绕过 dispatch lock 立即生效的控制命令。
        普通命令请用 exact() 或 prefix() 注册。
        """
        self._priority[cmd] = handler

    def exact(self, cmd: str, handler: Handler) -> None:
        """注册一个精确匹配命令（加锁执行）。

        【参数说明】
        - cmd: 命令名（如 "status"），会做小写归一化后存为 key
        - handler: 异步处理函数

        【使用场景】
        适用于 /status、/memory 这类完全匹配的常规命令。
        这些命令在 dispatch lock 保护下执行，避免与 Agent 正在处理的回合并发。
        """
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        """注册一个前缀匹配命令（加锁执行）。

        【参数说明】
        - pfx: 命令前缀（如 "/team "，注意末尾空格确保不误匹配 /teamwork）
        - handler: 异步处理函数

        【匹配规则】
        - 前缀表按长度降序排列，保证 "/team create" 不会误匹配 "/team " 的处理逻辑
        - 匹配成功后，前缀之后的剩余文本写入 ctx.args

        【使用场景】
        适用于带子命令的复合命令（如 /pairing approve xxx, /pairing deny xxx）。
        """
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def is_priority(self, text: str) -> bool:
        """判断输入文本是否为优先级命令。

        【参数说明】
        - text: 原始输入文本（如 "  /Stop please "），内部会做 strip + lower 归一化

        【返回值】
        - True: 该命令命中 priority 表，应使用 dispatch_priority() 不加锁执行
        - False: 不是 priority 命令
        """
        return text.strip().lower() in self._priority

    def is_dispatchable_command(self, text: str) -> bool:
        """判断输入文本是否匹配任何非 priority 命令（exact 或 prefix 层）。

        【参数说明】
        - text: 原始输入文本

        【返回值】
        - True: 保证 dispatch() 一定能匹配到一个 handler
        - False: 不是已知命令，dispatch() 也会返回 None

        【注意】
        这个函数**不会**检查 priority 层。调用方需要先调用 is_priority()，
        确认不是 priority 命令后，再调用本函数。
        """
        cmd = text.strip().lower()
        if cmd in self._exact:
            return True
        for pfx, _ in self._prefix:
            if cmd.startswith(pfx):
                return True
        return False

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """执行 priority 层命令分发。

        【中文名称】优先级命令分发

        【功能说明】
        在 dispatch lock 之外直接执行 priority 命令。
        这是 /stop、/restart 能够被立即响应而无需等待 Agent 释放锁的关键机制。

        【调用者】
        由 AgentLoop.run() 在检查到 priority 命令时直接调用。

        【参数说明】
        - ctx: 命令上下文，其中的 raw 字段会做小写归一化后查 priority 表

        【返回值】
        - 匹配成功：返回 handler 产生的 OutboundMessage
        - 未匹配（理论上不会发生，因为调用前已通过 is_priority() 检查）：返回 None
        """
        handler = self._priority.get(ctx.raw.lower())
        if handler:
            return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """执行常规命令分发（exact → prefix 两级匹配）。

        【中文名称】常规命令分发

        【功能说明】
        按 exact → prefix 顺序依次匹配命令 handler。
        这个方法在 dispatch lock 保护下调用，保证同一时刻只执行一个常规命令。

        【分发顺序】
        Phase 1: 查 exact 表 —— 直接用规范化后的命令名做 key 匹配
        Phase 2: 查 prefix 表 —— 遍历前缀列表，取第一个命中项
        Phase 3: 都未命中 —— 返回 None，表示不是已知命令

        【参数说明】
        - ctx: 命令上下文，包含 raw（原始输入）和 args（prefix 匹配后剩余参数）

        【副作用】
        - 当 prefix 匹配成功时，ctx.args 会被设置为前缀之后的剩余文本
          例如：输入 "/pairing approve ABCD-EFGH"，匹配前缀 "/pairing " 后，
          ctx.args = "approve ABCD-EFGH"

        【返回值】
        - 匹配成功：handler 产生的 OutboundMessage
        - 未匹配：None（由 AgentLoop 当作 LLM 对话继续处理）
        """
        cmd = ctx.raw.lower()

        # === Phase 1: 精确匹配 ===
        if handler := self._exact.get(cmd):
            return await handler(ctx)

        # === Phase 2: 前缀匹配（取最长匹配） ===
        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        # === Phase 3: 未命中 ===
        return None
