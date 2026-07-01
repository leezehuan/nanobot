"""子 Agent 启动工具：给主 Agent 一个“把任务外包出去”的入口。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.schema import NumberSchema, StringSchema, tool_parameters_schema
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


@tool_parameters(
    tool_parameters_schema(
        task=StringSchema("The task for the subagent to complete"),
        label=StringSchema("Optional short label for the task (for display)"),
        temperature=NumberSchema(
            description=(
                "Optional sampling temperature for the subagent "
                "(0.0 = deterministic, higher = more creative). "
                "Defaults to the provider's configured temperature."
            ),
            minimum=0.0,
            maximum=2.0,
        ),
        required=["task"],
    )
)
class SpawnTool(Tool, ContextAware):
    """后台子 Agent 启动工具。"""

    def __init__(self, manager: "SubagentManager"):
        """init。
        
        初始化 SpawnTool 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._manager = manager
        self._origin_channel: ContextVar[str] = ContextVar("spawn_origin_channel", default="cli")
        self._origin_chat_id: ContextVar[str] = ContextVar("spawn_origin_chat_id", default="direct")
        self._session_key: ContextVar[str] = ContextVar("spawn_session_key", default="cli:direct")
        self._origin_message_id: ContextVar[str | None] = ContextVar(
            "spawn_origin_message_id",
            default=None,
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        """create。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return cls(manager=ctx.subagent_manager)

    def set_context(self, ctx: RequestContext) -> None:
        """记录当前请求来源，方便子 Agent 完成后回传到原会话。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        self._origin_channel.set(ctx.channel)
        self._origin_chat_id.set(ctx.chat_id)
        self._session_key.set(ctx.session_key or f"{ctx.channel}:{ctx.chat_id}")
        self._origin_message_id.set(ctx.message_id)

    @property
    def name(self) -> str:
        """name。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return "spawn"

    @property
    def description(self) -> str:
        """description。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    async def execute(
        self,
        task: str,
        label: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """创建一个子 Agent 去执行给定任务。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        running = self._manager.get_running_count()
        limit = self._manager.max_concurrent_subagents
        if running >= limit:
            return (
                f"Cannot spawn subagent: concurrency limit reached "
                f"({running}/{limit} running). Wait for a running subagent "
                f"to complete before spawning a new one."
            )
        # 把当前工作区访问范围也传递给子 Agent，让它继承本轮安全边界。
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel.get(),
            origin_chat_id=self._origin_chat_id.get(),
            session_key=self._session_key.get(),
            origin_message_id=self._origin_message_id.get(),
            temperature=temperature,
            workspace_scope=current_workspace_scope(),
        )
