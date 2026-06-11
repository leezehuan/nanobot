"""RuntimeState 协议：定义 MyTool 可读取/修改的运行时状态接口。

这里不是一个真正的实现类，而是一个“最小契约”：

- MyTool 至少要求运行时对象提供哪些属性
- 这些属性大概代表什么运行时状态

在实际运行中，这个协议通常由 ``AgentLoop`` 满足。
"""

from typing import Any, Protocol


class RuntimeState(Protocol):
    """MyTool 对运行时状态提供者要求的最小契约。

    注意：

    - 这只是“最低必需字段”列表
    - MyTool 还会通过 ``getattr`` / ``setattr`` 动态访问更多属性
    - 那些点路径访问是否合法，不靠这个协议静态保证，而是在运行时校验
    """

    @property
    def model(self) -> str: ...

    @property
    def max_iterations(self) -> int: ...

    @property
    def current_iteration(self) -> int: ...

    @property
    def tool_names(self) -> list[str]: ...

    @property
    def workspace(self) -> str: ...

    @property
    def provider_retry_mode(self) -> str: ...

    @property
    def max_tool_result_chars(self) -> int: ...

    @property
    def context_window_tokens(self) -> int: ...

    @property
    def web_config(self) -> Any: ...

    @property
    def exec_config(self) -> Any: ...

    @property
    def workspace_sandbox(self) -> Any: ...

    @property
    def subagents(self) -> Any: ...

    @property
    def _runtime_vars(self) -> dict[str, Any]: ...

    @property
    def _last_usage(self) -> Any: ...

    def _sync_subagent_runtime_limits(self) -> None: ...

    @property
    def model_preset(self) -> str | None: ...

    _active_preset: str | None
