"""Agent 主循环：整个项目的核心处理引擎。

如果你只打算先读懂 nanobot 的“主干数据流”，这个文件是最值得优先看的。

它负责把一条用户消息完整走完下面这条链路：

1. 从 ``MessageBus`` 收到 ``InboundMessage``
2. 恢复或创建 ``Session``
3. 构建本轮 prompt 上下文
4. 调用 ``AgentRunner`` 执行“模型 <-> 工具”循环
5. 保存本轮新增历史
6. 产出 ``OutboundMessage`` 交还给渠道层

所以可以把 ``AgentLoop`` 理解成“回合编排器”，而不是单纯的“模型调用器”。
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from contextlib import AsyncExitStack, nullcontext, suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent import context as agent_context
from nanobot.agent import model_presets as preset_helpers
from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, CompositeHook
from nanobot.agent.memory import Consolidator
from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.self import MyTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import build_bus_progress_callback
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import (
    RuntimeEventBus,
    RuntimeEventPublisher,
    ensure_runtime_event_publisher,
)
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults, ModelPresetConfig
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import ProviderSnapshot
from nanobot.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    reset_workspace_scope,
)
from nanobot.session import turn_continuation
from nanobot.session.goal_state import (
    goal_state_runtime_lines,
    runner_wall_llm_timeout_s,
    sustained_goal_active,
)
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.document import extract_documents, reference_non_image_attachments
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.image_generation_intent import image_generation_prompt
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    SUSTAINED_GOAL_CONTINUE_PROMPT,
)

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ProviderConfig,
        ToolsConfig,
    )
    from nanobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"

class TurnState(Enum):
    """单个 turn 在 AgentLoop 状态机中的阶段枚举。"""
    RESTORE = auto()
    COMPACT = auto()
    COMMAND = auto()
    BUILD = auto()
    RUN = auto()
    SAVE = auto()
    RESPOND = auto()
    DONE = auto()


@dataclass
class StateTraceEntry:
    """状态机阶段追踪记录。

    主要用于调试和观测：
    - 当前停在哪个阶段
    - 这个阶段耗时多少毫秒
    - 触发了哪个状态转移事件
    """
    state: TurnState
    started_at: float
    duration_ms: float
    event: str
    error: str | None = None


@dataclass
class TurnContext:
    """单个 turn 的共享运行上下文。

    AgentLoop 的各个状态处理函数不会互相直接传零散参数，
    而是统一读写这一份上下文对象。
    这样一个 turn 在整个生命周期中的中间结果就有了稳定容器。
    """
    # 本轮收到的原始入站消息；媒体预处理后可能被替换为新对象。
    msg: InboundMessage
    # Session 的唯一键，用于读取/保存历史、运行时事件和记忆归档。
    session_key: str
    # 当前状态机所在阶段，每个状态处理函数执行后会更新到下一阶段。
    state: TurnState
    # 本轮 turn 的唯一标识，主要用于日志和状态追踪。
    turn_id: str
    # 当前会话对象，恢复阶段填充，后续阶段都基于它读写历史。
    session: Session | None = None

    # 从 Session 中取出的历史尾部，会参与本轮模型上下文构建。
    history: list[dict[str, Any]] = field(default_factory=list)
    # 真正发送给 AgentRunner 的初始 messages，包含 system prompt、历史和当前用户消息。
    initial_messages: list[dict[str, Any]] = field(default_factory=list)

    # AgentRunner 返回的最终文本内容，保存和回复阶段都会使用。
    final_content: str | None = None
    # 本轮模型实际调用过的工具名称列表，用于观测和统计。
    tools_used: list[str] = field(default_factory=list)
    # AgentRunner 执行后的完整消息序列，用于从中提取本轮新增历史。
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    # 本轮停止原因，例如正常结束、错误、工具错误或空回复等。
    stop_reason: str = ""
    # 本轮是否检测到上下文注入，用于决定最终回复和元数据处理。
    had_injections: bool = False

    # 用户消息是否已提前写入 Session，避免保存阶段重复持久化。
    user_persisted_early: bool = False
    # 保存本轮历史时需要跳过的消息数量，通常由提前持久化或续跑边界决定。
    save_skip: int = 0

    # 准备返回给渠道的出站消息；命令快捷返回或响应阶段会填充它。
    outbound: OutboundMessage | None = None
    # 是否抑制最终回复；为 True 时只保存状态，不向渠道发送消息。
    suppress_response: bool = False

    # 进度回调，用于把工具调用、阶段进展等事件推送给渠道或调用方。
    on_progress: Callable[..., Awaitable[None]] | None = None
    # 流式文本回调，用于边生成边发送增量内容。
    on_stream: Callable[[str], Awaitable[None]] | None = None
    # 流式输出结束回调，用于通知调用方本轮流式响应已完成。
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    # 重试等待回调，用于在限流或临时错误等待重试时提示调用方。
    on_retry_wait: Callable[[str], Awaitable[None]] | None = None

    # 长轮次或续跑期间接收后续用户输入的队列，传给 AgentRunner 消费。
    pending_queue: asyncio.Queue | None = None
    # 自动压缩阶段生成的待注入摘要，用于补充被压缩掉的上下文。
    pending_summary: str | None = None

    # 是否为临时 turn；临时 turn 不触发长期记忆整理等持久化副作用。
    ephemeral: bool = False
    # 本轮使用的工具注册表；为空时使用 AgentLoop 默认工具集合。
    tools: ToolRegistry | None = None

    # 本轮从进入 AgentLoop 起算的墙钟开始时间，用于整体延迟统计。
    turn_wall_started_at: float = field(default_factory=time.time)
    # 用户可见运行开始时间；内部续跑会沿用原始开始时间来计算可见延迟。
    visible_run_started_at: float | None = None
    # 本轮最终统计出的延迟毫秒数，会写入事件和出站消息元数据。
    turn_latency_ms: int | None = None

    # 状态机执行轨迹，记录每个阶段的耗时、事件和异常信息。
    trace: list[StateTraceEntry] = field(default_factory=list)


class AgentLoop:
    """Agent 主循环。

    【核心职责】
    1. 从总线取消息
    2. 组织上下文
    3. 进入 Runner 执行
    4. 保存和恢复会话
    5. 把最终结果发回渠道

    从架构上说，它是“产品层”和“模型工具执行层”的中间桥梁。
    """

    @property
    def current_iteration(self) -> int:
        """current iteration。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return self._current_iteration

    @property
    def tool_names(self) -> list[str]:
        """tool names。
        
        返回当前注册表里的工具名快照。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self.tools.tool_names

    def llm_runtime(self) -> LLMRuntime:
        """返回当前 loop 正在使用的 provider/model 组合。
        
        实现方法：先刷新 provider snapshot，确保运行时配置热更新已同步；再把当前 provider 和 model 封装成 LLMRuntime，供状态查询或工具读取。"""
        self._refresh_provider_snapshot()
        return LLMRuntime(self.provider, self.model)

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    # 事件驱动状态机转移表。
    # 每个状态处理函数返回一个事件字符串，再由这里决定下一状态。
    _TRANSITIONS: dict[tuple[TurnState, str], TurnState] = {
        (TurnState.RESTORE, "ok"): TurnState.COMPACT,
        (TurnState.COMPACT, "ok"): TurnState.COMMAND,
        (TurnState.COMMAND, "dispatch"): TurnState.BUILD,
        (TurnState.COMMAND, "shortcut"): TurnState.DONE,
        (TurnState.BUILD, "ok"): TurnState.RUN,
        (TurnState.RUN, "ok"): TurnState.SAVE,
        (TurnState.SAVE, "ok"): TurnState.RESPOND,
        (TurnState.RESPOND, "ok"): TurnState.DONE,
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        max_concurrent_subagents: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        tool_hint_max_length: int | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        consolidation_ratio: float = 0.5,
        max_messages: int = 120,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
        image_generation_provider_config: ProviderConfig | None = None,
        image_generation_provider_configs: dict[str, ProviderConfig] | None = None,
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None = None,
        provider_signature: tuple[object, ...] | None = None,
        model_presets: dict[str, ModelPresetConfig] | None = None,
        model_preset: str | None = None,
        preset_snapshot_loader: preset_helpers.PresetSnapshotLoader | None = None,
        runtime_events: RuntimeEventBus | None = None,
        runtime_model_publisher: Callable[[str, str | None], None] | None = None,
    ):
        """init。
        
        初始化 AgentLoop 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        from nanobot.config.schema import ToolsConfig

        # 工具配置和 Agent 默认值是后续运行时限制的基础：
        # 调用方显式传入的参数优先，否则回退到项目默认设置。
        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()

        # 消息总线负责连接渠道和 AgentLoop；运行时事件总线负责把模型切换、
        # 工具进度等内部状态变化发布给 WebUI、桌面端或其他观察者。
        self.bus = bus
        self.runtime_events = runtime_events or RuntimeEventBus()
        self.runtime_event_publisher = RuntimeEventPublisher(self.runtime_events)
        self.channels_config = channels_config

        # Provider / model 是本轮 LLM 调用的核心运行时对象。
        # snapshot loader 允许配置热更新时为“下一轮”切换 provider 或模型。
        self.provider = provider
        self._provider_snapshot_loader = provider_snapshot_loader
        self._preset_snapshot_loader = preset_snapshot_loader
        self._runtime_model_publisher = runtime_model_publisher
        self._provider_signature = provider_signature
        self._default_selection_signature = preset_helpers.default_selection_signature(provider_signature)
        self.workspace = workspace
        self.model = model or provider.get_default_model()

        # 以下限制决定单轮对话的工具循环次数、上下文容量和工具结果截断策略。
        # 它们既影响成本，也影响模型可见的历史和工具输出规模。
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.tool_hint_max_length = (
            tool_hint_max_length if tool_hint_max_length is not None
            else defaults.tool_hint_max_length
        )

        # 拆出常用的工具子配置，方便文件执行、网页访问等工具快速读取。
        self.tools_config = _tc
        self.web_config = _tc.web
        self.exec_config = _tc.exec

        # 图片生成可以单独指定 provider；旧的单 provider 参数会兼容映射到 openrouter。
        self._image_generation_provider_configs = dict(image_generation_provider_configs or {})
        if (
            image_generation_provider_config is not None
            and "openrouter" not in self._image_generation_provider_configs
        ):
            self._image_generation_provider_configs["openrouter"] = image_generation_provider_config

        # Cron、工作区限制和工作区作用域解析器共同决定工具能在哪些路径和调度环境中运行。
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace_scopes = WorkspaceScopeResolver(
            default_workspace=workspace,
            default_restrict_to_workspace=restrict_to_workspace,
        )

        # 记录 AgentLoop 级别的轻量运行状态。
        # _start_time 可用于统计进程内本轮 loop 存活时间；_last_usage 保存最近一次
        # LLM 调用返回的 token/用量信息；_extra_hooks 是调用方额外注入的事件扩展点。
        # 这些 hook 不改变主流程，只在非临时回合里旁路接收进度、流式输出、工具调用等事件。
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        # ContextBuilder 负责把系统提示词、技能说明、工作区信息、历史摘要和当前消息
        # 拼成最终发给模型的 messages；timezone 和 disabled_skills 会影响运行时上下文内容。
        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        # SessionManager 管理每个会话的历史、metadata、checkpoint 和持久化存储。
        # 测试或嵌入式调用可以传入自定义 session_manager；否则使用工作区默认存储。
        self.sessions = session_manager or SessionManager(workspace)
        # ToolRegistry 是整个 AgentLoop 共享的一张工具表，后面 _register_default_tools()
        # 会把文件、命令、网页、MCP、消息发送等工具注册进来，供 AgentRunner 调用。
        self.tools = ToolRegistry()
        # 每个逻辑会话各自拥有一份文件读写跟踪状态。
        # 之所以不能直接挂在工具实例上，是因为 ToolRegistry 会被整个 AgentLoop 共享；
        # 因此工具需要通过 contextvars 动态解析“当前这次调用属于哪个 session”。
        self._file_state_store = FileStateStore()
        # AgentRunner 是真正执行“模型 -> 工具调用 -> 模型继续回答”循环的组件；
        # AgentLoop 负责外层编排，Runner 负责单轮内部的 LLM/tool 交互。
        self.runner = AgentRunner(provider)
        # SubagentManager 管理由主 Agent 派生出来的子 Agent。
        # 子 Agent 复用当前 provider/model/workspace/tool 配置，但有自己的并发限制和迭代限制；
        # llm_wall_timeout_for_session 会按 session 动态计算模型调用的墙钟超时时间。
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            tools_config=_tc,
            max_tool_result_chars=self.max_tool_result_chars,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_iterations=self.max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            llm_wall_timeout_for_session=lambda sk: runner_wall_llm_timeout_s(self.sessions, sk),
        )
        # 会话和历史回放策略。
        # unified_session 开启后，多渠道消息会合并到同一个逻辑会话；_max_messages 限制
        # 每轮回放给模型的历史条数，非法或非正值回退到默认 120。
        self._unified_session = unified_session
        self._max_messages = max_messages if max_messages > 0 else 120
        # 主循环和 MCP 连接状态。_mcp_stacks 保存每个 MCP 连接的异步清理栈，
        # _mcp_connected/_mcp_connecting 用于避免重复连接或并发连接。
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        # 异步任务管理。
        # _active_tasks 按 session_key 记录正在处理的主任务，方便 /stop 精确取消；
        # _background_tasks 保存自动压缩等后台协程；_session_locks 保证同一 session 内串行处理，
        # 避免多个请求同时写入同一份历史或 checkpoint。
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 每个 session 一条“回合中途注入队列”。
        # 如果某个 session 当前已有活跃任务，新消息不会新开并发任务，
        # 而是先进这个队列，等当前回合在合适时机把它们并入上下文。
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # ``NANOBOT_MAX_CONCURRENT_REQUESTS``:
        # 小于等于 0 表示不设上限；默认值是 3。
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        # 全局并发闸门限制同时处理的入站请求数量。
        # 这里和上面的 _session_locks 分工不同：lock 保证“同一 session 串行”，
        # semaphore 保证“整个 AgentLoop 同时运行的会话任务不要过多”。
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # Consolidator 负责在历史过长或上下文预算紧张时，把旧消息压缩成摘要。
        # 它需要访问长期记忆存储、当前 provider/model、SessionManager、上下文构建函数
        # 以及工具定义，用来估算 token、重建必要上下文并生成可回放的会话摘要。
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            consolidation_ratio=consolidation_ratio,
            unified_session=unified_session,
        )
        # AutoCompact 在主循环空闲或处理消息前检查会话 TTL，
        # 对长时间未活跃或超过阈值的 session 触发自动压缩，减少后续上下文压力。
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        # 模型预设允许把 provider/model/context_window 等组合命名保存，
        # 运行时可以通过 set_model_preset() 切换；_active_preset 记录当前生效的预设名。
        self.model_presets: dict[str, ModelPresetConfig] = model_presets or {}
        self._active_preset: str | None = None
        if model_preset:
            # 初始化阶段应用默认预设，但不发布运行时变更事件，避免启动时误报“模型切换”。
            self.set_model_preset(model_preset, publish_update=False)
        # 注册默认工具集合。必须在 consolidator 创建后执行，因为部分工具需要访问
        # 当前 loop 的运行时状态，而 consolidator 也会引用工具定义来构建压缩上下文。
        self._register_default_tools()
        # _runtime_vars 是给 MyTool 等运行时工具读写的轻量变量区；
        # _current_iteration 记录当前 AgentRunner 工具循环迭代次数，供状态查询或 UI 展示使用。
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        # 命令路由器负责匹配和派发内置斜杠命令，例如 /new、/stop 等。
        # 普通命令会进入消息处理状态机，高优先级命令可在主循环中提前拦截执行。
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    @classmethod
    def from_config(
        cls,
        config: Any,
        bus: MessageBus | None = None,
        **extra: Any,
    ) -> AgentLoop:
        """按项目标准配置创建一个 ``AgentLoop``。

        这里负责把 ``config`` 中常见的字段统一展开成构造参数。
        额外的 ``extra`` 会继续透传给 ``AgentLoop.__init__``，
        方便调用方在标准配置之外再覆盖或补充参数
        （例如 ``cron_service``、``session_manager``）。

        实现方法：读取 config 中的默认 Agent 设置、provider 配置、模型预设和工具配置，创建缺省 MessageBus，并把这些值统一传入 AgentLoop 构造函数。
        """
        from nanobot.providers.factory import make_provider

        if bus is None:
            bus = MessageBus()
        defaults = config.agents.defaults
        provider = extra.pop("provider", None) or make_provider(config)
        resolved = config.resolve_preset()
        model = extra.pop("model", None) or resolved.model
        context_window_tokens = extra.pop("context_window_tokens", None) or resolved.context_window_tokens
        provider_snapshot_loader = extra.pop("provider_snapshot_loader", None)
        preset_snapshot_loader = extra.pop("preset_snapshot_loader", None) or preset_helpers.make_preset_snapshot_loader(
            config,
            provider_snapshot_loader,
        )
        return cls(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=model,
            max_iterations=defaults.max_tool_iterations,
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            context_window_tokens=context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            tool_hint_max_length=defaults.tool_hint_max_length,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            consolidation_ratio=defaults.consolidation_ratio,
            max_messages=defaults.max_messages,
            tools_config=config.tools,
            model_presets=preset_helpers.configured_model_presets(config),
            model_preset=defaults.model_preset,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
            **extra,
        )

    def _sync_subagent_runtime_limits(self) -> None:
        """把 subagent 的运行时限制同步到当前主循环设置。
        
        实现方法：直接把 AgentLoop 当前的 max_iterations 写入 SubagentManager，保证后续子 Agent 使用与主 Agent
        一致的工具循环上限。"""
        self.subagents.max_iterations = self.max_iterations

    def _apply_provider_snapshot(
        self,
        snapshot: ProviderSnapshot,
        *,
        publish_update: bool = True,
        model_preset: str | None = None,
    ) -> None:
        """为后续回合热切换 model/provider，而不打断当前活跃回合。
        
        实现方法：从 snapshot 取出 provider、model 和上下文窗口，依次更新 AgentLoop、AgentRunner、SubagentManager 与
        Consolidator；最后记录签名并按需发布运行时模型变更事件，所以正在执行的回合继续使用已捕获的对象，后续回合才使用新配置。"""
        provider = snapshot.provider
        model = snapshot.model
        context_window_tokens = snapshot.context_window_tokens
        old_model = self.model
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.runner.provider = provider
        self.subagents.set_provider(provider, model)
        self.consolidator.set_provider(provider, model, context_window_tokens)
        self._provider_signature = snapshot.signature
        if publish_update and self._runtime_model_publisher is not None:
            self._runtime_model_publisher(
                self.model,
                model_preset if model_preset is not None else self.model_preset,
            )
        if publish_update:
            self._runtime_events().runtime_model_changed(
                self.model,
                model_preset if model_preset is not None else self.model_preset,
            )
        logger.info("Runtime model switched for next turn: {} -> {}", old_model, model)

    def _refresh_provider_snapshot(self) -> None:
        """refresh provider snapshot。

        实现方法：调用 snapshot loader 读取最新 provider 配置，比较签名判断是否变化；若当前使用模型预设则先重建预设快照，最后只在签名变化时应用新
        snapshot。"""
        # 没有配置快照加载器时，说明运行期 provider 配置不可热刷新，直接沿用当前对象。
        if self._provider_snapshot_loader is None:
            return
        try:
            # 先读取最新的基础 provider snapshot，后续再按模型预设决定是否包装/覆盖。
            snapshot = self._provider_snapshot_loader()
        except Exception:
            # 刷新失败不能影响当前回合，保留旧 provider，并记录异常供排查。
            logger.exception("Failed to refresh provider config")
            return

        # default_selection 用来判断“默认模型选择”是否改变；它变化时，当前预设需要失效。
        default_selection = preset_helpers.default_selection_signature(snapshot.signature)
        if self._active_preset and self._default_selection_signature in (None, default_selection):
            # 默认选择未变：继续保持当前模型预设，但基于最新基础 snapshot 重新构建。
            self._default_selection_signature = default_selection
            try:
                snapshot = self._build_model_preset_snapshot(self._active_preset)
            except Exception:
                # 预设重建失败同样不能破坏已有运行态，避免切到一个不完整配置。
                logger.exception("Failed to refresh active model preset")
                return
        else:
            # 默认选择发生变化或没有活动预设：清空预设状态，使用基础 snapshot。
            self._active_preset = None
            self._default_selection_signature = default_selection

        # 签名相同表示 provider/model/context window 等关键配置没有变化，无需重复应用。
        if snapshot.signature == self._provider_signature:
            return

        # 真正应用前用最终 snapshot 的签名刷新默认选择标记，保证预设 snapshot 与基础 snapshot 都一致。
        self._default_selection_signature = preset_helpers.default_selection_signature(snapshot.signature)
        self._apply_provider_snapshot(snapshot)

    @property
    def model_preset(self) -> str | None:
        """model preset。
        
        读写当前生效的模型预设名；设置时会转交给预设切换逻辑。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._active_preset

    @model_preset.setter
    def model_preset(self, name: str | None) -> None:
        """model preset。
        
        读写当前生效的模型预设名；设置时会转交给预设切换逻辑。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        self.set_model_preset(name)

    def _build_model_preset_snapshot(self, name: str) -> ProviderSnapshot:
        """build model preset snapshot。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
        return preset_helpers.build_runtime_preset_snapshot(
            name=name,
            presets=self.model_presets,
            provider=self.provider,
            loader=self._preset_snapshot_loader,
        )

    def set_model_preset(self, name: str | None, *, publish_update: bool = True) -> None:
        """按名字解析模型预设，并同步更新所有相关运行时对象。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        name = preset_helpers.normalize_preset_name(name, self.model_presets)
        snapshot = self._build_model_preset_snapshot(name)
        self._apply_provider_snapshot(snapshot, publish_update=publish_update, model_preset=name)
        self._active_preset = name

    def _register_default_tools(self) -> None:
        """通过工具加载器注册默认工具集合。
        
        实现方法：把对象写入内部映射表，并清理依赖该映射生成的缓存。"""
        from nanobot.agent.tools.context import ToolContext
        from nanobot.agent.tools.loader import ToolLoader

        ctx = ToolContext(
            config=self.tools_config,
            workspace=str(self.workspace),
            bus=self.bus,
            subagent_manager=self.subagents,
            cron_service=self.cron_service,
            sessions=self.sessions,
            provider_snapshot_loader=self._provider_snapshot_loader,
            image_generation_provider_configs=self._image_generation_provider_configs,
            timezone=self.context.timezone or "UTC",
            workspace_sandbox=self.workspace_scopes.sandbox_status,
            runtime_events=self.runtime_events,
        )
        loader = ToolLoader()
        registered = loader.load(ctx, self.tools)

        # ``MyTool`` 需要直接拿到运行时状态对象，
        # 所以不能完全依赖通用 loader，这里手动补注册。
        if self.tools_config.my.enable:
            self.tools.register(
                MyTool(runtime_state=self, modify_allowed=self.tools_config.my.allow_set)
            )
            registered.append("my")

        logger.info("Registered {} tools: {}", len(registered), registered)

    async def _connect_mcp(self) -> None:
        """连接配置中声明的 MCP 服务器。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        await agent_context.connect_mcp(self, self.tools)

    def _set_tool_context(
        self, channel: str, chat_id: str,
        message_id: str | None = None, metadata: dict | None = None,
        session_key: str | None = None,
    ) -> None:
        """给所有需要路由信息的工具刷新请求上下文。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        from nanobot.agent.tools.context import ContextAware

        if session_key is not None:
            effective_key = session_key
        elif self._unified_session:
            effective_key = UNIFIED_SESSION_KEY
        else:
            effective_key = f"{channel}:{chat_id}"

        request_ctx = RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=effective_key,
            metadata=dict(metadata or {}),
        )

        for name in self.tools.tool_names:
            tool = self.tools.get(name)
            if tool and isinstance(tool, ContextAware):
                tool.set_context(request_ctx)

    @staticmethod
    def _runtime_chat_id(msg: InboundMessage) -> str:
        """返回本轮在运行时元数据里暴露给模型的 chat_id。
        
        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
        return str(msg.metadata.get("context_chat_id") or msg.chat_id)

    async def _build_bus_progress_callback(
        self, msg: InboundMessage
    ) -> Callable[..., Awaitable[None]]:
        """构造一个会把进度事件发布到消息总线的回调。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
        return build_bus_progress_callback(self.bus, msg)

    async def _build_retry_wait_callback(
        self, msg: InboundMessage
    ) -> Callable[[str], Awaitable[None]]:
        """构造一个会把“重试等待”事件发布到消息总线的回调。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""

        async def _on_retry_wait(content: str) -> None:
            """on retry wait。
            
            实现方法：根据错误类型和 retry-after 提示计算等待时间，等待期间发送心跳进度，再重新执行请求。"""
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        return _on_retry_wait

    def _runtime_events(self) -> RuntimeEventPublisher:
        """runtime events。
        
        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
        return ensure_runtime_event_publisher(self)

    def _persist_user_message_early(
        self,
        msg: InboundMessage,
        session: Session,
        **kwargs: Any,
    ) -> bool:
        """在 turn 正式开始前，先把触发它的 user message 落盘。

        这样即使后面模型调用或工具执行过程中崩溃，至少用户输入不会丢。
        返回值表示这条消息是否真的被持久化了。

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        if not turn_continuation.should_persist_user_message(msg.metadata):
            return False
        media_paths = [p for p in (msg.media or []) if isinstance(p, str) and p]
        has_text = isinstance(msg.content, str) and msg.content.strip()
        if has_text or media_paths:
            extra: dict[str, Any] = ({"media": list(media_paths)} if media_paths else {}) | agent_context.session_extra(msg.metadata)
            extra.update(kwargs)
            text = msg.content if isinstance(msg.content, str) else ""
            session.add_message("user", text, **extra)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            return True
        return False

    def _build_initial_messages(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict[str, Any]],
        pending_summary: str | None,
        include_memory_recent_history: bool = True,
    ) -> list[dict[str, Any]]:
        """构建本轮 LLM 调用的初始消息列表。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
        scope = self.workspace_scopes.for_message(msg, session.metadata)
        return self.context.build_messages(
            history=history,
            current_message=image_generation_prompt(msg.content, msg.metadata),
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=self._runtime_chat_id(msg),
            sender_id=msg.sender_id,
            session_summary=pending_summary,
            session_metadata=session.metadata,
            workspace=scope.project_path,
            runtime_state=self,
            inbound_message=msg,
            include_memory_recent_history=include_memory_recent_history,
            session_key=session.key,
            unified_session=self._unified_session,
        )

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """直接在 loop 中派发内置命令，并把结果投递到消息总线。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        result = await dispatch_fn(ctx)
        if result:
            await self.bus.publish_outbound(result)
        else:
            logger.warning("Command '{}' matched but dispatch returned None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """取消某个 session_key 下的所有主任务和子 Agent。
        
        实现方法：找到目标 session 或任务对应的 asyncio task，发出取消并等待清理结果。"""
        tasks = self._active_tasks.pop(key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        sub_cancelled = await self.subagents.cancel_by_session(key)
        return cancelled + sub_cancelled

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """计算这条消息真正用于任务路由的 session_key。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    def _replay_token_budget(self) -> int:
        """根据上下文窗口推导“历史回放”可用 token 预算。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if self.context_window_tokens <= 0:
            return 0
        max_output = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        try:
            reserved_output = int(max_output)
        except (TypeError, ValueError):
            reserved_output = 4096
        budget = self.context_window_tokens - max(1, reserved_output) - 1024
        return budget if budget > 0 else max(128, self.context_window_tokens // 2)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        pending_queue: asyncio.Queue | None = None,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """运行内部 AgentRunner，并把结果整理成 AgentLoop 需要的统一格式。

        返回值依次是：
        ``(final_content, tools_used, messages, stop_reason, had_injections)``

        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。
        """
        self._sync_subagent_runtime_limits()

        loop_hook = AgentProgressHook(
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            metadata=metadata,
            session_key=session_key,
            tool_hint_max_length=self.tool_hint_max_length,
            set_tool_context=self._set_tool_context,
            on_iteration=lambda iteration: setattr(self, "_current_iteration", iteration),
        )
        hook: AgentHook = loop_hook
        if not ephemeral and self._extra_hooks:
            hook = CompositeHook([loop_hook] + self._extra_hooks)

        async def _checkpoint(payload: dict[str, Any]) -> None:
            """checkpoint。
            
            实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """从 pending queue 中提取中途追加的后续消息。

            典型场景：
            - 用户在同一 session 的本轮处理尚未结束时，又发来新消息
            - 子 Agent 在后台跑完，把结果回注到当前 turn

            实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
            """
            if pending_queue is None:
                return []

            def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                """to user message。
                
                实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = self._prepare_message_media(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                return {"role": "user", "content": user_content}

            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    items.append(_to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # 如果暂时没有可注入消息，但本回合新生成的 sub-agent 还在跑，
            # 就短暂阻塞等待。这样后续完成消息还能按顺序注入当前回合，
            # 而不是被拆成独立的新入站消息。
            if (not items
                    and session is not None
                    and self.subagents.get_running_count_by_session(session.key) > 0):
                try:
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items
                items.append(_to_user_message(msg))
                while len(items) < limit:
                    try:
                        items.append(_to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        active_session_key = session.key if session else session_key
        effective_scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        request_ctx = RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=active_session_key,
            metadata=dict(metadata or {}),
        )
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        request_token = bind_request_context(request_ctx)
        workspace_token = bind_workspace_scope(effective_scope)
        # 构造“继续执行长期目标”的补充提示，
        # 直接把当前目标写进消息里，避免前面 Runtime Context 被裁剪后模型看不到目标。
        _goal_lines = goal_state_runtime_lines(session.metadata if session is not None else None)
        _goal_continue = (
            "You have an active sustained goal:\n\n"
            + "\n".join(_goal_lines)
            + "\n\nPlease continue working toward the objective using your tools, "
            "or call complete_goal if the work is truly finished."
        ) if _goal_lines else SUSTAINED_GOAL_CONTINUE_PROMPT
        session_metadata = session.metadata if session is not None else None
        try:
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=tools or self.tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                hook=hook,
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,
                workspace=effective_scope.project_path,
                session_key=session.key if session else None,
                context_window_tokens=self.context_window_tokens,
                context_block_limit=self.context_block_limit,
                provider_retry_mode=self.provider_retry_mode,
                progress_callback=on_progress,
                stream_progress_deltas=on_stream is not None,
                retry_wait_callback=on_retry_wait,
                checkpoint_callback=_checkpoint,
                injection_callback=_drain_pending,
                # 长期目标回合合法地可能跑很久，所以不一定受普通 LLM 总超时限制；
                # 但流式 provider 仍会用空闲超时来防止真正卡死。
                llm_timeout_s=runner_wall_llm_timeout_s(
                    self.sessions,
                    session.key if session is not None else session_key,
                    metadata=session_metadata,
                    message_metadata=metadata,
                ),
                goal_active_predicate=lambda: sustained_goal_active(session.metadata) if session is not None else False,
                goal_continue_message=_goal_continue,
                finalize_on_max_iterations=turn_continuation.should_finalize_on_max_iterations(
                    pending_queue_available=pending_queue is not None and session is not None,
                    session_metadata=session_metadata,
                    message_metadata=metadata,
                ),
            ))
        finally:
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
            reset_file_states(file_state_token)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=metadata,
            )
            # 把最终完整内容再走一次流式通道，
            # 这样飞书等流式渠道才能把最终卡片内容补全，而不是停在空壳状态。
            if on_stream and on_stream_end and should_stream:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """运行 Agent 主循环，并把消息分发成任务以保持对 ``/stop`` 的响应性。

        【中文名称】运行 Agent 主循环

        【功能说明】
        这是整个 AgentLoop 的最外层入口，也是 nanobot 启动后的"主心跳"。
        它会无限循环地从 MessageBus 的 inbound 队列消费消息，然后创建 asyncio.Task
        去异步处理。之所以不直接在这里同步处理，是为了让循环始终保持响应，
        不会被某个慢回合阻塞住（例如期间可以接收 /stop 命令）。

        【完整生命周期】
        1. 连接 MCP 服务器 → 让外部工具能够在后续回合中被调用
        2. 进入 while self._running 无限循环
        3. 每次循环从 bus.inbound 取一条消息（1 秒超时）
        4. 超时期间检查是否有闲置会话需要自动压缩（AutoCompact）
        5. 收到消息后，先判断是否是运行时控制消息 → 是则直接处理并跳过
        6. 判断是否是高优先级命令 → 是则内联 dispatch 并跳过
        7. 如果该 session 已有活跃任务，把新消息放入 pending_queue（中途注入）
        8. 否则创建 asyncio.Task 异步执行 _dispatch()
        9. 记录活跃任务，方便 /stop 指令找到对应的 Task 并取消

        【任务调度】
        - 同一 session_key 内的消息通过 asyncio.Lock 串行化处理
        - 不同 session 之间通过 Semaphore（默认 3）限制并发数
        - 被 /stop 取消的任务会尽量恢复 checkpoint 保护已完成的工具结果

        【参数说明】
        无参数 —— 这是协程入口，所有配置都在 __init__ 时注入。

        【返回值】
        无返回值 —— 这是一个常驻协程，进程退出时才会结束。

        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。
        """
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                self.auto_compact.check_expired(
                    self._schedule_background,
                    active_session_keys=self._pending_queues.keys(),
                )
                continue
            except asyncio.CancelledError:
                # 真正的任务取消一定要继续向上抛，程序才能正确停机。
                # 这里只忽略某些集成层偶尔泄漏出来的“伪取消”信号。
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            effective_key = self._effective_session_key(msg)
            if await agent_context.handle_runtime_control(self, msg, self.tools):
                continue
            if self.commands.is_priority(raw):
                await self._dispatch_command_inline(
                    msg, effective_key, raw,
                    self.commands.dispatch_priority,
                )
                continue
            # 如果这个 session 已经有活跃的处理中任务，
            # 就把新消息送进它的 pending queue，走“中途注入”流程，
            # 避免同一会话并发开两个互相竞争的回合。
            if effective_key in self._pending_queues:
                # 非优先级命令不能被塞进注入队列，
                # 否则像普通消息一样排队会破坏命令语义，所以要直接分发。
                if self.commands.is_dispatchable_command(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch,
                    )
                    continue
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for session {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for session {}",
                        effective_key,
                    )
                    continue
            # 在真正分发前先算出生效 session key。
            # 这样 unified_session 开启时，``/stop`` 才能准确找到对应任务。
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """处理一条消息：同 session 串行，不同 session 可并发。

        【中文名称】消息分发处理器

        【功能说明】
        这是 AgentLoop 中每个异步 Task 的入口函数。它负责：
        - 用 asyncio.Lock 保证同一 session 内的消息串行处理
        - 用 asyncio.Semaphore 限制总并发 Task 数
        - 为当前回合注册 pending_queue（中途消息注入队列）
        - 管理流式输出回调（on_stream / on_stream_end）
        - 处理回合完成/异常后的清理和事件发布

        【并发控制两层】
        1. lock = asyncio.Lock（session 级别）→ 同一会话内严格串行
        2. gate = asyncio.Semaphore（全局级别）→ 控制同时运行的 Task 总数
        只有同时拿到 lock 和 gate 的 Task 才能真正开始处理回合。

        【pending_queue 机制】
        持有锁的 Task 会创建一个 asyncio.Queue(maxsize=20) 作为本回合的
        中途注入队列。后续到达的同一 session 消息不进新建 Task，
        而是 put 到这个队列里，由当前回合在合适时机（工具执行后、回答前）消费。

        【异常处理】
        - asyncio.CancelledError：恢复 checkpoint 保护已完成的工具结果
        - 普通 Exception：发布 "Sorry, I encountered an error." 回渠道

        【流式输出回调】
        如果消息 metadata 中含 _wants_stream，则构造 on_stream / on_stream_end
        回调，把 LLM 的增量输出分段发送到消息总线。
        Stream ID 格式为 {session_key}:{timestamp_ns}:{segment_index}，
        这样渠道端可以根据 stream_id 区分不同的流段落。

        【参数说明】
        - msg: InboundMessage → 要处理的入站消息

        【返回值】
        无返回值 —— 结果通过 bus.publish_outbound() 发布到消息总线

        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
        """
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        pending: asyncio.Queue | None = None
        try:
            async with lock, gate:
                # 只有真正持有这个 session 锁的任务，
                # 才能登记并持有当前有效的“中途注入队列”。
                pending = asyncio.Queue(maxsize=20)
                self._pending_queues[session_key] = pending
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # 把同一轮回答拆成多个独立流片段，便于渠道端做分段展示。
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            """current stream id。
                            
                            实现方法：逐块读取上游事件，把文本增量、工具调用增量和完成信号分别转发给调用方。"""
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            """on stream。
                            
                            实现方法：逐块读取上游事件，把文本增量、工具调用增量和完成信号分别转发给调用方。"""
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            """on stream end。
                            
                            实现方法：逐块读取上游事件，把文本增量、工具调用增量和完成信号分别转发给调用方。"""
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    completed_channel = msg.channel
                    completed_chat_id = msg.chat_id
                    if response is not None:
                        await self.bus.publish_outbound(response)
                        completed_channel = response.channel
                        completed_chat_id = response.chat_id
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                    continuing = turn_continuation.internal_continuation_pending(msg.metadata)
                    if not continuing:
                        await self._runtime_events().turn_completed(
                            channel=completed_channel,
                            chat_id=completed_chat_id,
                            session_key=session_key,
                            metadata=msg.metadata,
                        )
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    # 尽量保住被 /stop 打断前已经产生的上下文：
                    # 包括工具结果、assistant 已生成消息等。
                    # 这些 checkpoint 在工具执行时已经写进 session metadata；
                    # 这里把它们真正落进 session history，下一轮对话就还能看见。
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        if self._restore_runtime_checkpoint(session):
                            self._clear_pending_user_turn(session)
                            self.sessions.save(session)
                            logger.info(
                                "Restored partial context for cancelled session {}",
                                key,
                            )
                    except Exception:
                        logger.debug(
                            "Could not restore checkpoint for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
                    if not turn_continuation.internal_continuation_pending(msg.metadata):
                        await self._runtime_events().turn_completed(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            session_key=session_key,
                            metadata=msg.metadata,
                        )
                finally:
                    # 把 pending queue 里还没处理到的消息重新发布回总线，
                    # 让它们作为新的入站消息重新参与后续处理，而不是被静默丢掉。
                    # 同时只能清理由“当前任务自己拥有”的队列，
                    # 避免后来的等待任务错误接管清理权。
                    queue = None
                    if self._pending_queues.get(session_key) is pending:
                        queue = self._pending_queues.pop(session_key, None)
                    else:
                        queue = pending
                    if queue is not None:
                        leftover = 0
                        while True:
                            try:
                                item = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            await self.bus.publish_inbound(item)
                            leftover += 1
                        if leftover:
                            logger.info(
                                "Re-published {} leftover message(s) to bus for session {}",
                                leftover, session_key,
                            )
                    if not turn_continuation.internal_continuation_pending(msg.metadata):
                        await self._runtime_events().run_status_changed(
                            msg, session_key, "idle"
                        )
                        self._runtime_events().clear_turn(session_key)
        finally:
            if pending is None:
                await self._runtime_events().run_status_changed(
                    msg, session_key, "idle"
                )
                self._runtime_events().clear_turn(session_key)

    async def close_mcp(self) -> None:
        """先清空后台归档任务，再关闭 MCP 连接。
        
        实现方法：按已登记的异步清理栈释放连接、会话或后台资源，并重置连接状态。"""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """把一个协程登记为可追踪后台任务，并在停机时统一等待它结束。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """停止 Agent 主循环。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_system_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """处理系统级入站消息，例如 subagent 的公告或结果回传。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        channel, chat_id = (
            msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        )
        logger.info("Processing system message from {}", msg.sender_id)
        key = msg.session_key_override or f"{channel}:{chat_id}"
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)
        if pending:
            logger.info("Memory compact triggered for session {}", key)

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            replay_max_messages=self._max_messages,
        )
        is_subagent = msg.sender_id == "subagent"
        if is_subagent and self._persist_subagent_followup(session, msg):
            logger.debug("Subagent result persisted for session {}", key)
            self.sessions.save(session)
        self._set_tool_context(
            channel, chat_id, msg.metadata.get("message_id"),
            msg.metadata, session_key=key,
        )
        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
        }
        history = session.get_history(**_hist_kwargs)
        current_role = "assistant" if is_subagent else "user"
        workspace_scope = self.workspace_scopes.for_message(msg, session.metadata)

        messages = self.context.build_messages(
            history=history,
            current_message="" if is_subagent else msg.content,
            channel=channel,
            chat_id=chat_id,
            current_role=current_role,
            sender_id=msg.sender_id,
            session_summary=pending,
            session_metadata=session.metadata,
            workspace=workspace_scope.project_path,
            runtime_state=self,
            inbound_message=msg,
            skip_runtime_lines=is_subagent,
            session_key=key,
            unified_session=self._unified_session,
        )
        t_wall = time.time()
        final_content, _, all_msgs, stop_reason, _ = await self._run_agent_loop(
            messages, session=session, channel=channel, chat_id=chat_id,
            message_id=msg.metadata.get("message_id"),
            metadata=msg.metadata,
            session_key=key,
            pending_queue=pending_queue,
        )
        wall_done = time.time()
        latency_ms = max(0, int((wall_done - t_wall) * 1000))
        self._save_turn(session, all_msgs, 1 + len(history), turn_latency_ms=latency_ms)
        self._runtime_events().record_turn_latency(key, latency_ms)
        session.enforce_file_cap(
            on_archive=partial(self.context.memory.raw_archive, session_key=key)
        )
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(
            self.consolidator.maybe_consolidate_by_tokens(
                session,
                replay_max_messages=self._max_messages,
            )
        )
        content = final_content or "Background task completed."
        outbound_metadata: dict[str, Any] = {}
        if channel == "slack" and key.startswith("slack:") and key.count(":") >= 2:
            outbound_metadata["slack"] = {"thread_ts": key.split(":", 2)[2]}
        if origin_message_id := msg.metadata.get("origin_message_id"):
            outbound_metadata["origin_message_id"] = origin_message_id
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            metadata=outbound_metadata,
        )

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
    ) -> OutboundMessage | None:
        """按状态机完整处理一条入站消息，逐步走完 7 个阶段。

        【中文名称】按状态机处理消息

        【功能说明】
        这是 AgentLoop 处理单条用户消息的"总编排函数"。它不直接调用 LLM，
        而是通过 TurnContext 状态机逐个阶段推进：恢复 → 压缩 → 命令 →
        构建 → 运行 → 保存 → 响应。每个阶段由 _state_* 方法实现。

        【完整的 7 个处理阶段】
        Phase 1: RESTORE（恢复） → 加载 session、恢复 checkpoint、发出事件
        Phase 2: COMPACT（压缩） → 检查是否需要自动压缩闲置会话
        Phase 3: COMMAND（命令） → 尝试匹配内置斜杠命令（/new, /stop 等）
        Phase 4: BUILD（构建） → 拼接 system prompt + history → 构建 messages
        Phase 5: RUN（运行）    → 调用 AgentRunner 执行"模型↔工具"循环
        Phase 6: SAVE（保存）   → 将本轮新增消息写入 Session 并持久化
        Phase 7: RESPOND（响应）→ 组装 OutboundMessage 返回给调用方

        【状态转移表（self._TRANSITIONS）】
        RESTORE + "ok"     → COMPACT
        COMPACT + "ok"     → COMMAND
        COMMAND + "dispatch" → BUILD（普通消息）
        COMMAND + "shortcut" → DONE（快捷命令，跳过 LLM）
        BUILD   + "ok"     → RUN
        RUN     + "ok"     → SAVE
        SAVE    + "ok"     → RESPOND
        RESPOND + "ok"     → DONE

        【TurnContext 的作用】
        各个状态处理函数不会互相直接传参，而是通过 TurnContext 这个共享上下文对象
        来读写中间结果（session、history、final_content 等）。

        【参数说明】
        - msg: InboundMessage → 要处理的入站消息
        - session_key: str | None → 会话唯一键（为 None 时用 msg.session_key）
        - on_progress: callback → 工具执行进度回调
        - on_stream: callback → 流式增量输出回调
        - on_stream_end: callback → 流式输出结束回调
        - pending_queue: Queue | None → 中途消息注入队列
        - ephemeral: bool → 是否为临时轮次（不保存历史）

        【返回值】
        - OutboundMessage | None: 组装好的出站消息；被 MessageTool 消费或命令抑制时返回 None

        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
        """
        self._refresh_provider_snapshot()

        if msg.channel == "system":
            return await self._process_system_message(
                msg,
                session_key=session_key,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                pending_queue=pending_queue,
            )

        key = session_key or msg.session_key
        t0 = time.time()
        ctx = TurnContext(
            msg=msg,
            session=None,
            session_key=key,
            state=TurnState.RESTORE,
            turn_id=f"{key}:{time.time_ns()}",
            turn_wall_started_at=t0,
            visible_run_started_at=turn_continuation.internal_continuation_run_started_at(
                msg.metadata,
            ),
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            pending_queue=pending_queue,
            ephemeral=ephemeral,
            tools=tools,
        )

        # 主状态机循环：每轮执行一个状态处理函数，再根据事件名跳到下一状态。
        while ctx.state is not TurnState.DONE:
            handler_name = f"_state_{ctx.state.name.lower()}"
            handler = getattr(self, handler_name, None)
            if handler is None:
                raise RuntimeError(f"Missing state handler for {ctx.state}")

            t0 = time.perf_counter()
            try:
                event = await handler(ctx)
            except Exception:
                duration = (time.perf_counter() - t0) * 1000
                ctx.trace.append(
                    StateTraceEntry(
                        state=ctx.state,
                        started_at=t0,
                        duration_ms=duration,
                        event="",
                        error="exception",
                    )
                )
                raise

            duration = (time.perf_counter() - t0) * 1000
            ctx.trace.append(
                StateTraceEntry(
                    state=ctx.state,
                    started_at=t0,
                    duration_ms=duration,
                    event=event,
                )
            )
            logger.debug(
                "[turn {}] State {} took {:.1f}ms -> event {}",
                ctx.turn_id,
                ctx.state.name,
                duration,
                event,
            )

            next_state = self._TRANSITIONS.get((ctx.state, event))
            if next_state is None:
                raise RuntimeError(
                    f"[turn {ctx.turn_id}] No transition from {ctx.state} "
                    f"on event {event!r}"
                )
            ctx.state = next_state

        logger.debug(
            "[turn {}] Turn completed after {} states",
            ctx.turn_id,
            len(ctx.trace),
        )
        return ctx.outbound

    def _assemble_outbound(
        self,
        msg: InboundMessage,
        final_content: str,
        all_msgs: list[dict[str, Any]],
        stop_reason: str,
        had_injections: bool,
        on_stream: Callable[[str], Awaitable[None]] | None,
        *,
        turn_latency_ms: int | None = None,
    ) -> OutboundMessage | None:
        """把 turn 结果组装成最终出站消息。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        # 如果本轮已经通过 MessageTool 主动发过消息，
        # 某些场景下就不再补发一条默认最终回复。
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason not in {"error", "tool_error"}:
            meta["_streamed"] = True
        if turn_latency_ms is not None:
            meta["latency_ms"] = int(turn_latency_ms)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    async def _state_restore(self, ctx: TurnContext) -> TurnState:
        """Phase 1: 恢复阶段。

        这里负责：
        - 预处理媒体附件
        - 获取/恢复 session
        - 恢复未完成 turn 的 checkpoint
        - 发出 turn 开始事件

        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
        """
        msg = ctx.msg

        if msg.media:
            new_content, image_only = self._prepare_message_media(msg.content, msg.media)
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_only)
            msg = ctx.msg

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 正常情况下 session 已由外层预备好，
        # 这里再兜底一次，防止独立调用状态处理函数时缺失。
        if ctx.session is None:
            ctx.session = self.sessions.get_or_create(ctx.session_key)
        await self._runtime_events().session_turn_started(msg, ctx.session_key)
        self.workspace_scopes.persist_message_scope(ctx.session, msg)

        if self._restore_runtime_checkpoint(ctx.session):
            self.sessions.save(ctx.session)
        if self._restore_pending_user_turn(ctx.session):
            self.sessions.save(ctx.session)

        return "ok"

    def _prepare_message_media(self, content: str, media: list[str]) -> tuple[str, list[str]]:
        """prepare message media。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if self._should_extract_document_text():
            return extract_documents(content, media)
        return reference_non_image_attachments(content, media)

    def _should_extract_document_text(self) -> bool:
        """should extract document text。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        if self.channels_config is None:
            return True
        return self.channels_config.extract_document_text

    async def _state_compact(self, ctx: TurnContext) -> str:
        """Phase 2: 自动压缩准备阶段。
        
        实现方法：在不改变当前任务关键上下文的前提下，压缩过长历史或工具结果，减少后续 provider 请求的 token 压力。"""
        ctx.session, pending = self.auto_compact.prepare_session(ctx.session, ctx.session_key)
        ctx.pending_summary = pending
        return "ok"

    async def _state_command(self, ctx: TurnContext) -> str:
        """Phase 3: 内置命令分发阶段。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        raw = ctx.msg.content.strip()
        cmd_ctx = CommandContext(
            msg=ctx.msg, session=ctx.session, key=ctx.session_key, raw=raw, loop=self
        )
        result = await self.commands.dispatch(cmd_ctx)
        if result is not None:
            ctx.outbound = result
            # 快捷命令不会进入 BUILD/SAVE，所以这里必须自己完成持久化。
            # 同时把这些消息打上 _command 标记，避免再次进入 LLM 历史回放。
            if raw.lower() != "/new":
                ctx.user_persisted_early = self._persist_user_message_early(
                    ctx.msg, ctx.session, _command=True
                )
                ctx.session.add_message(
                    "assistant", result.content, _command=True
                )
                self.sessions.save(ctx.session)
                self._clear_pending_user_turn(ctx.session)
            return "shortcut"
        return "dispatch"

    async def _state_build(self, ctx: TurnContext) -> str:
        """Phase 4: 上下文构建阶段。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
        if not ctx.ephemeral:
            await self.consolidator.maybe_consolidate_by_tokens(
                ctx.session,
                replay_max_messages=self._max_messages,
            )
        self._set_tool_context(
            ctx.msg.channel,
            ctx.msg.chat_id,
            ctx.msg.metadata.get("message_id"),
            ctx.msg.metadata,
            session_key=ctx.session_key,
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # 从 Session 中提取将要回放给模型的历史尾部。
        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
        }
        ctx.history = ctx.session.get_history(**_hist_kwargs)
        self._runtime_events().record_turn_runtime(
            ctx.session_key,
            self.llm_runtime(),
        )

        # 把 system prompt、历史、当前用户消息真正拼成最终 messages。
        ctx.initial_messages = self._build_initial_messages(
            ctx.msg,
            ctx.session,
            ctx.history,
            ctx.pending_summary,
            include_memory_recent_history=not ctx.ephemeral,
        )
        ctx.user_persisted_early = self._persist_user_message_early(
            ctx.msg, ctx.session
        )

        if ctx.on_progress is None:
            ctx.on_progress = await self._build_bus_progress_callback(ctx.msg)
        if ctx.on_retry_wait is None:
            ctx.on_retry_wait = await self._build_retry_wait_callback(ctx.msg)

        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        """Phase 5: 调用 AgentRunner 执行模型/工具循环。
        
        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
        await self._runtime_events().run_status_changed(
            ctx.msg,
            ctx.session_key,
            "running",
            started_at=ctx.visible_run_started_at,
        )
        result = await self._run_agent_loop(
            ctx.initial_messages,
            on_progress=ctx.on_progress,
            on_stream=ctx.on_stream,
            on_stream_end=ctx.on_stream_end,
            on_retry_wait=ctx.on_retry_wait,
            session=ctx.session,
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            metadata=ctx.msg.metadata,
            session_key=ctx.session_key,
            pending_queue=ctx.pending_queue,
            ephemeral=ctx.ephemeral,
            tools=ctx.tools,
        )
        final_content, tools_used, all_msgs, stop_reason, had_injections = result
        ctx.final_content = final_content
        ctx.tools_used = tools_used
        ctx.all_messages = all_msgs
        ctx.stop_reason = stop_reason
        ctx.had_injections = had_injections
        await turn_continuation.maybe_continue_turn(ctx)
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        """Phase 6: 保存本轮新增历史。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        turn_continuation.prepare_save_boundary(ctx)

        if (
            (ctx.final_content is None or not ctx.final_content.strip())
            and not ctx.suppress_response
        ):
            ctx.final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        latency_started_at = (
            ctx.visible_run_started_at
            if turn_continuation.internal_continuation_inbound(ctx.msg.metadata)
            and ctx.visible_run_started_at is not None
            else ctx.turn_wall_started_at
        )
        ctx.turn_latency_ms = max(0, int((time.time() - latency_started_at) * 1000))
        # 注意这里只保存“本轮新增部分”，不是把整个 messages 全量重写回 Session。
        self._save_turn(
            ctx.session, ctx.all_messages, ctx.save_skip,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        self._runtime_events().record_turn_latency(
            ctx.session_key,
            ctx.turn_latency_ms,
        )
        if not ctx.ephemeral:
            ctx.session.enforce_file_cap(
                on_archive=partial(self.context.memory.raw_archive, session_key=ctx.session_key)
            )
            self._schedule_background(
                self.consolidator.maybe_consolidate_by_tokens(
                    ctx.session,
                    replay_max_messages=self._max_messages,
                )
            )
        self._clear_pending_user_turn(ctx.session)
        self._clear_runtime_checkpoint(ctx.session)
        self.sessions.save(ctx.session)
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        """Phase 7: 组装出站消息。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        if ctx.suppress_response:
            ctx.outbound = None
            return "ok"
        ctx.outbound = self._assemble_outbound(
            ctx.msg,
            ctx.final_content,
            ctx.all_messages,
            ctx.stop_reason,
            ctx.had_injections,
            ctx.on_stream,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        if ctx.ephemeral and ctx.outbound is not None:
            ctx.outbound.metadata["_stop_reason"] = ctx.stop_reason
        return "ok"

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """在写入 Session 前，清理不适合长期保存的多模态块。
        
        实现方法：先复制或规范化输入，再移除 provider 或工具无法接受的字段，并保留可安全回放的信息。"""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        *,
        turn_latency_ms: int | None = None,
    ) -> None:
        """把本轮新增消息写进 Session，并在必要时截断超大 tool 结果。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        from datetime import datetime

        last_assistant_idx: int | None = None
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and ContextBuilder._RUNTIME_CONTEXT_TAG in content:
                    # 去掉拼在 user 消息尾部的 runtime-context 块。
                    tag_pos = content.find(ContextBuilder._RUNTIME_CONTEXT_TAG)
                    before = content[:tag_pos].rstrip("\n ")
                    if before:
                        entry["content"] = before
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            if role == "assistant":
                last_assistant_idx = len(session.messages) - 1
        if turn_latency_ms is not None and last_assistant_idx is not None:
            session.messages[last_assistant_idx]["latency_ms"] = int(turn_latency_ms)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """在 prompt 构建前先持久化子 Agent 回传结果，增强可恢复性。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """把当前进行中的 turn 状态暂存到 session.metadata。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        """mark pending user turn。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        """clear pending user turn。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        """clear runtime checkpoint。
        
        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        """checkpoint message key。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """把未完成 turn 的 checkpoint 恢复成可见历史。
        
        实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """补完一种特殊崩溃场景：只存下了 user message，还没生成回复。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
    ) -> OutboundMessage | None:
        """直接处理一条消息并返回出站结果。

        这条入口主要给 CLI / SDK / 内部调用使用，不需要先经过渠道分发。

        实现方法：把直接调用参数包装成 InboundMessage 和回调上下文，复用同一套状态机处理流程，最后把 OutboundMessage 的内容返回给调用方。
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id,
            content=content, media=media or [],
        )
        # 直接调用路径也复用同一把 dispatch 锁，
        # 保证它和来自消息总线的普通回合同样遵守串行规则。
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        try:
            async with lock:
                kwargs: dict[str, Any] = {
                    "session_key": session_key,
                    "on_progress": on_progress,
                    "on_stream": on_stream,
                    "on_stream_end": on_stream_end,
                    "ephemeral": ephemeral,
                }
                if tools is not None:
                    kwargs["tools"] = tools
                return await self._process_message(
                    msg,
                    **kwargs,
                )
        finally:
            await self._runtime_events().run_status_changed(msg, session_key, "idle")
            self._runtime_events().clear_turn(session_key)
