"""工具运行时上下文。

工具虽然是被统一注册的，但每一次调用发生时，仍然需要知道：
- 当前来自哪个 channel/chat
- 当前 session_key 是什么
- 这次运行挂了哪些管理器和服务

这些上下文就通过本模块传递。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

_CURRENT_REQUEST_CONTEXT: ContextVar["RequestContext | None"] = ContextVar(
    "nanobot_tool_request_context",
    default=None,
)


@dataclass(frozen=True)
class RequestContext:
    """单次请求级上下文，会在处理消息时注入工具。"""
    channel: str
    chat_id: str
    message_id: str | None = None
    session_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextAware(Protocol):
    """约定：支持运行时上下文注入的工具应实现 ``set_context``。"""
    def set_context(self, ctx: RequestContext) -> None:
        """set context。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        ...


def bind_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    """把当前请求上下文绑定到 ``contextvars``。
    
    实现方法：根据当前运行模式选择同步或流式 provider 调用，并把回调、工具定义和超时参数传入。"""
    return _CURRENT_REQUEST_CONTEXT.set(ctx)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    """恢复之前的请求上下文绑定状态。
    
    实现方法：根据当前运行模式选择同步或流式 provider 调用，并把回调、工具定义和超时参数传入。"""
    _CURRENT_REQUEST_CONTEXT.reset(token)


def current_request_context() -> RequestContext | None:
    """获取当前异步任务绑定的请求上下文。
    
    实现方法：根据当前运行模式选择同步或流式 provider 调用，并把回调、工具定义和超时参数传入。"""
    return _CURRENT_REQUEST_CONTEXT.get()


def current_request_session_key() -> str | None:
    """快捷获取当前请求对应的 session_key。
    
    实现方法：根据当前运行模式选择同步或流式 provider 调用，并把回调、工具定义和超时参数传入。"""
    ctx = current_request_context()
    return ctx.session_key if ctx else None


@dataclass
class ToolContext:
    """构造工具实例时使用的广义上下文。

    和 ``RequestContext`` 不同，这里更偏“运行时依赖注入容器”，
    例如 config、workspace、bus、session manager、cron service 等。
    """
    config: Any
    workspace: str
    bus: Any | None = None
    subagent_manager: Any | None = None
    cron_service: Any | None = None
    sessions: Any | None = None
    file_state_store: Any = field(default=None)
    provider_snapshot_loader: Callable[[], Any] | None = None
    image_generation_provider_configs: dict[str, Any] | None = None
    timezone: str = "UTC"
    workspace_sandbox: Any | None = None
    runtime_events: Any | None = None
