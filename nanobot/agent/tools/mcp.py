"""MCP 客户端：把外部 MCP 服务器接入为 nanobot 原生工具。

这是理解 nanobot “怎么连接第三方工具生态”的关键文件之一。

它主要负责：

1. 连接配置里的 MCP server
2. 枚举 server 暴露的 tools / resources / prompts
3. 包装成 nanobot 统一的 ``Tool`` 接口
4. 在断线或配置热更新时，负责重连与重注册
"""

import asyncio
import os
import re
import shutil
import urllib.parse
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from typing import Any, Mapping
from weakref import WeakKeyDictionary

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_ACK,
    RUNTIME_CONTROL_MCP_RELOAD,
    InboundMessage,
)
from nanobot.security.network import validate_url_target

# 这些异常通常表示“瞬时连接故障”，适合自动重试一次。
# 常见场景是 MCP server 刚重启，或两次调用之间网络短暂中断。
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset((
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "ConnectionAbortedError",
    "ConnectionError",
))

_WINDOWS_SHELL_LAUNCHERS: frozenset[str] = frozenset(("npx", "npm", "pnpm", "yarn", "bunx"))

# 各家模型 API 对工具名字符集都有要求。
# 因此这里会把非法字符替换为下划线，并折叠连续下划线。
_SANITIZE_RE = re.compile(r"_+")
_RELOAD_LOCKS: WeakKeyDictionary[Any, asyncio.Lock] = WeakKeyDictionary()
_ReconnectCallback = Callable[[str, str, Tool], Awaitable[Tool | None]]


def _sanitize_name(name: str) -> str:
    """把 MCP 派生名称清洗成模型 API 可接受的工具名。
    
    实现方法：先复制或规范化输入，再移除 provider 或工具无法接受的字段，并保留可安全回放的信息。"""
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name))


def _is_transient(exc: BaseException) -> bool:
    """判断异常是否像“可重试的瞬时连接错误”。
    
    实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _is_session_terminated(exc: BaseException) -> bool:
    """判断异常是否意味着 MCP client session 已经失效。
    
    实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
    messages = [str(exc)]
    error = getattr(exc, "error", None)
    if error is not None:
        messages.append(str(getattr(error, "message", "")))
    return any(
        marker in message.lower()
        for marker in ("session terminated", "connection closed")
        for message in messages
    )


async def _probe_http_url(url: str, timeout: float = 3.0) -> bool:
    """先做一次轻量 TCP 探测，判断 HTTP MCP 服务是否可达。

    这样可以避免在端口根本没开时直接进入更重的 transport 初始化流程，
    减少 anyio 清理异常把错误抛到事件循环外的风险。

    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if not port:
        port = 443 if parsed.scheme == "https" else 80
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        with suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def _validate_mcp_request_url(request: httpx.Request) -> None:
    """校验每一个发往 MCP HTTP 服务的请求 URL。
    
    实现方法：先做格式与安全边界检查，再把失败原因转换成调用方可读的错误信息。"""
    ok, error = validate_url_target(str(request.url))
    if not ok:
        raise httpx.RequestError(
            f"Blocked unsafe MCP URL {request.url} ({error})",
            request=request,
        )


def _windows_command_basename(command: str) -> str:
    """提取 Windows 命令/路径的 basename，并转成小写。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()


def _normalize_windows_stdio_command(
    command: str,
    args: list[str] | None,
    env: dict[str, str] | None,
) -> tuple[str, list[str], dict[str, str] | None]:
    """在 Windows 上包装某些 shell 启动器，保证 stdio 型 MCP server 稳定启动。
    
    实现方法：把输入统一成内部约定格式，处理大小写、空值、别名或 provider 差异。"""
    normalized_args = list(args or [])
    if os.name != "nt":
        return command, normalized_args, env

    basename = _windows_command_basename(command)
    if basename in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return command, normalized_args, env

    if basename.endswith((".exe", ".com")):
        return command, normalized_args, env

    resolved = shutil.which(command, path=(env or {}).get("PATH")) or command
    resolved_basename = _windows_command_basename(resolved)
    should_wrap = (
        basename in _WINDOWS_SHELL_LAUNCHERS
        or basename.endswith((".cmd", ".bat"))
        or resolved_basename.endswith((".cmd", ".bat"))
    )
    if not should_wrap:
        return command, normalized_args, env

    comspec = (env or {}).get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
    return comspec, ["/d", "/c", command, *normalized_args], env


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    """从可空联合类型里提取唯一的“非 null 分支”。
    
    实现方法：从对象、字典或响应块中按候选路径提取目标值，提取失败时返回空值而不是中断主流程。"""
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """把 MCP schema 规范化成更适合 OpenAI 工具定义的形式。
    
    实现方法：把输入统一成内部约定格式，处理大小写、空值、别名或 provider 差异。"""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop) if isinstance(prop, dict) else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


class _MCPWrapperBase(Tool):
    """所有 MCP 包装工具的公共基类。

    这里最重要的职责不是“执行工具”，而是记住自己属于哪个 server，
    并在 session 断掉时协助触发定向重连。
    """

    _plugin_discoverable = False

    def _set_mcp_connection(self, session: Any, server_name: str) -> None:
        """set mcp connection。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        self._session = session
        self._server_name = server_name
        self._reconnect: _ReconnectCallback | None = None

    def set_reconnect_handler(self, reconnect: _ReconnectCallback) -> None:
        """set reconnect handler。
        
        实现方法：把新值写入实例状态，并同步更新依赖该状态的子组件或上下文变量。"""
        self._reconnect = reconnect

    async def _refresh_session_after_termination(
        self,
        exc: BaseException,
        already_refreshed: bool,
        capability_kind: str,
    ) -> bool:
        """refresh session after termination。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        if already_refreshed or not _is_session_terminated(exc) or self._reconnect is None:
            return False
        logger.warning(
            "MCP {} '{}' session terminated; reconnecting server '{}' before retry",
            capability_kind,
            self._name,
            self._server_name,
        )
        refreshed_tool = await self._reconnect(self._server_name, self._name, self)
        refreshed_session = getattr(refreshed_tool, "_session", None)
        if refreshed_session is None:
            logger.warning(
                "MCP {} '{}' could not refresh session for server '{}'",
                capability_kind,
                self._name,
                self._server_name,
            )
            return False
        self._session = refreshed_session
        return True


class MCPToolWrapper(_MCPWrapperBase):
    """把单个 MCP tool 包装成 nanobot 的 ``Tool``。"""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        """init。
        
        初始化 MCPToolWrapper 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._set_mcp_connection(session, server_name)
        self._original_name = tool_def.name
        self._name = _sanitize_name(f"mcp_{server_name}_{tool_def.name}")
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        """name。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._name

    @property
    def description(self) -> str:
        """description。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """parameters。
        
        返回本工具的参数 JSON Schema，供模型按结构生成调用参数。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        """execute。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        from mcp import types

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._original_name, arguments=kwargs),
                    timeout=self._tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP tool '{}' timed out after {}s", self._name, self._tool_timeout
                )
                return f"(MCP tool call timed out after {self._tool_timeout}s)"
            except asyncio.CancelledError:
                # MCP SDK 底层 anyio cancel scope 在超时/失败时，偶尔会把
                # CancelledError 冒出来。只有真的是外部取消时才继续向上抛。
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
                return "(MCP tool call was cancelled)"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "tool",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP tool '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)  # Brief backoff before retry
                        continue
                    # 第二次瞬时失败还没恢复，就放弃重试并返回明确错误。
                    logger.exception(
                        "MCP tool '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP tool call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP tool '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP tool call failed: {type(exc).__name__})"
            else:
                # 成功后，把 MCP 返回的内容块统一转换成字符串结果。
                parts = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP tool call failed)"  # Unreachable, but satisfies type checkers


class MCPResourceWrapper(_MCPWrapperBase):
    """把 MCP resource URI 包装成只读 nanobot 工具。"""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, resource_def, resource_timeout: int = 30):
        """init。
        
        初始化 MCPResourceWrapper 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._set_mcp_connection(session, server_name)
        self._uri = resource_def.uri
        self._name = _sanitize_name(f"mcp_{server_name}_resource_{resource_def.name}")
        desc = resource_def.description or resource_def.name
        self._description = f"[MCP Resource] {desc}\nURI: {self._uri}"
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._resource_timeout = resource_timeout

    @property
    def name(self) -> str:
        """name。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._name

    @property
    def description(self) -> str:
        """description。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """parameters。
        
        返回本工具的参数 JSON Schema，供模型按结构生成调用参数。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._parameters

    @property
    def read_only(self) -> bool:
        """read only。
        
        声明工具是否只读，供调度器判断并发和安全策略。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return True

    async def execute(self, **kwargs: Any) -> str:
        """execute。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        from mcp import types

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.read_resource(self._uri),
                    timeout=self._resource_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP resource '{}' timed out after {}s", self._name, self._resource_timeout
                )
                return f"(MCP resource read timed out after {self._resource_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP resource '{}' was cancelled by server/SDK", self._name)
                return "(MCP resource read was cancelled)"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "resource",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP resource '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP resource '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP resource read failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP resource '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP resource read failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for block in result.contents:
                    if isinstance(block, types.TextResourceContents):
                        parts.append(block.text)
                    elif isinstance(block, types.BlobResourceContents):
                        parts.append(f"[Binary resource: {len(block.blob)} bytes]")
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP resource read failed)"  # Unreachable


class MCPPromptWrapper(_MCPWrapperBase):
    """把 MCP prompt 包装成只读 nanobot 工具。"""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, prompt_def, prompt_timeout: int = 30):
        """init。
        
        初始化 MCPPromptWrapper 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._set_mcp_connection(session, server_name)
        self._prompt_name = prompt_def.name
        self._name = _sanitize_name(f"mcp_{server_name}_prompt_{prompt_def.name}")
        desc = prompt_def.description or prompt_def.name
        self._description = (
            f"[MCP Prompt] {desc}\n"
            "Returns a filled prompt template that can be used as a workflow guide."
        )
        self._prompt_timeout = prompt_timeout

        # 用 prompt 的参数声明来构造工具参数 schema，
        # 这样模型可以像调普通工具一样填写参数。
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in prompt_def.arguments or []:
            prop: dict[str, Any] = {"type": "string"}
            if getattr(arg, "description", None):
                prop["description"] = arg.description
            properties[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def name(self) -> str:
        """name。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._name

    @property
    def description(self) -> str:
        """description。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """parameters。
        
        返回本工具的参数 JSON Schema，供模型按结构生成调用参数。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self._parameters

    @property
    def read_only(self) -> bool:
        """read only。
        
        声明工具是否只读，供调度器判断并发和安全策略。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return True

    async def execute(self, **kwargs: Any) -> str:
        """execute。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        from mcp import types
        from mcp.shared.exceptions import McpError

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.get_prompt(self._prompt_name, arguments=kwargs),
                    timeout=self._prompt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP prompt '{}' timed out after {}s", self._name, self._prompt_timeout
                )
                return f"(MCP prompt call timed out after {self._prompt_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP prompt '{}' was cancelled by server/SDK", self._name)
                return "(MCP prompt call was cancelled)"
            except McpError as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                logger.exception(
                    "MCP prompt '{}' failed: code={} message={}",
                    self._name,
                    exc.error.code,
                    exc.error.message,
                )
                return f"(MCP prompt call failed: {exc.error.message} [code {exc.error.code}])"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP prompt '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP prompt '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP prompt call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP prompt '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP prompt call failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for message in result.messages:
                    content = message.content
                    if isinstance(content, types.TextContent):
                        parts.append(content.text)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, types.TextContent):
                                parts.append(block.text)
                            else:
                                parts.append(str(block))
                    else:
                        parts.append(str(content))
                return "\n".join(parts) or "(no output)"

        return "(MCP prompt call failed)"  # Unreachable


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry
) -> dict[str, AsyncExitStack]:
    """连接配置中的 MCP server，并注册其 tools/resources/prompts。

    返回 ``server_name -> AsyncExitStack`` 的映射。
    每个 server 独占一个 exit stack，避免多个 MCP 连接共享同一清理栈时互相影响。

    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    async def connect_single_server(name: str, cfg) -> tuple[str, AsyncExitStack | None]:
        """connect single server。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        server_stack = AsyncExitStack()
        await server_stack.__aenter__()

        try:
            transport_type = cfg.type
            if not transport_type:
                if cfg.command:
                    transport_type = "stdio"
                elif cfg.url:
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    logger.warning("MCP server '{}': no command or url configured, skipping", name)
                    await server_stack.aclose()
                    return name, None

            if transport_type in {"sse", "streamableHttp"}:
                ok, error = validate_url_target(cfg.url)
                if not ok:
                    logger.warning(
                        "MCP server '{}': blocked unsafe URL {} ({})",
                        name,
                        cfg.url,
                        error,
                    )
                    await server_stack.aclose()
                    return name, None

            if transport_type == "stdio":
                command, args, env = _normalize_windows_stdio_command(
                    cfg.command,
                    cfg.args,
                    cfg.env or None,
                )
                params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env,
                    cwd=cfg.cwd or None,
                )
                read, write = await server_stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                if not await _probe_http_url(cfg.url):
                    logger.warning("MCP server '{}': {} unreachable, skipping", name, cfg.url)
                    await server_stack.aclose()
                    return name, None

                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    """httpx client factory。
                    
                    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
                    merged_headers = {
                        "Accept": "application/json, text/event-stream",
                        **(cfg.headers or {}),
                        **(headers or {}),
                    }
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        event_hooks={"request": [_validate_mcp_request_url]},
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await server_stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                if not await _probe_http_url(cfg.url):
                    logger.warning("MCP server '{}': {} unreachable, skipping", name, cfg.url)
                    await server_stack.aclose()
                    return name, None

                http_client = await server_stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        event_hooks={"request": [_validate_mcp_request_url]},
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await server_stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                await server_stack.aclose()
                return name, None

            session = await server_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            enabled_tools = set(cfg.enabled_tools)
            allow_all_tools = "*" in enabled_tools
            registered_count = 0
            matched_enabled_tools: set[str] = set()
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            available_wrapped_names = [_sanitize_name(f"mcp_{name}_{tool_def.name}") for tool_def in tools.tools]
            for tool_def in tools.tools:
                wrapped_name = _sanitize_name(f"mcp_{name}_{tool_def.name}")
                if (
                    not allow_all_tools
                    and tool_def.name not in enabled_tools
                    and wrapped_name not in enabled_tools
                ):
                    logger.debug(
                        "MCP: skipping tool '{}' from server '{}' (not in enabledTools)",
                        wrapped_name,
                        name,
                    )
                    continue
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)
                registered_count += 1
                if enabled_tools:
                    if tool_def.name in enabled_tools:
                        matched_enabled_tools.add(tool_def.name)
                    if wrapped_name in enabled_tools:
                        matched_enabled_tools.add(wrapped_name)

            if enabled_tools and not allow_all_tools:
                unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
                if unmatched_enabled_tools:
                    logger.warning(
                        "MCP server '{}': enabledTools entries not found: {}. Available raw names: {}. "
                        "Available wrapped names: {}",
                        name,
                        ", ".join(unmatched_enabled_tools),
                        ", ".join(available_raw_names) or "(none)",
                        ", ".join(available_wrapped_names) or "(none)",
                    )

            try:
                resources_result = await session.list_resources()
                for resource in resources_result.resources:
                    wrapper = MCPResourceWrapper(
                        session, name, resource, resource_timeout=cfg.tool_timeout
                    )
                    registry.register(wrapper)
                    registered_count += 1
                    logger.debug(
                        "MCP: registered resource '{}' from server '{}'", wrapper.name, name
                    )
            except Exception as e:
                logger.debug("MCP server '{}': resources not supported or failed: {}", name, e)

            try:
                prompts_result = await session.list_prompts()
                for prompt in prompts_result.prompts:
                    wrapper = MCPPromptWrapper(
                        session, name, prompt, prompt_timeout=cfg.tool_timeout
                    )
                    registry.register(wrapper)
                    registered_count += 1
                    logger.debug("MCP: registered prompt '{}' from server '{}'", wrapper.name, name)
            except Exception as e:
                logger.debug("MCP server '{}': prompts not supported or failed: {}", name, e)

            logger.info(
                "MCP server '{}': connected, {} capabilities registered", name, registered_count
            )
            return name, server_stack

        except Exception as e:
            hint = ""
            text = str(e).lower()
            if any(
                marker in text
                for marker in (
                    "parse error",
                    "invalid json",
                    "unexpected token",
                    "jsonrpc",
                    "content-length",
                )
            ):
                hint = (
                    " Hint: this looks like stdio protocol pollution. Make sure the MCP server writes "
                    "only JSON-RPC to stdout and sends logs/debug output to stderr instead."
                )
            logger.exception("MCP server '{}': failed to connect: {}", name, hint)
            with suppress(Exception):
                await server_stack.aclose()
            return name, None

    server_stacks: dict[str, AsyncExitStack] = {}

    for name, cfg in mcp_servers.items():
        try:
            result = await connect_single_server(name, cfg)
        except Exception as e:
            logger.exception("MCP server '{}' connection failed: {}", name, e)
            continue
        if result is not None and result[1] is not None:
            server_stacks[result[0]] = result[1]

    return server_stacks


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """提取需要随 session 持久化保存的 MCP preset 附加信息。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    mcp_presets = metadata.get("mcp_presets") if isinstance(metadata, Mapping) else None
    return {"mcp_presets": mcp_presets} if isinstance(mcp_presets, list) and mcp_presets else {}


def runtime_lines(
    message: Any,
    *,
    available_server_names: set[str] | None = None,
    configured_server_names: set[str] | None = None,
    connected_server_names: set[str] | None = None,
    skip: bool = False,
) -> list[str]:
    """生成当前 turn 暴露给模型看的 MCP preset 注释文本。
    
    实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
    if skip:
        return []
    if configured_server_names is None:
        configured_server_names = available_server_names
    if connected_server_names is None:
        connected_server_names = available_server_names
    metadata = message.metadata if isinstance(getattr(message, "metadata", None), Mapping) else None
    structured = metadata.get("mcp_presets") if isinstance(metadata, Mapping) else None
    if not isinstance(structured, list):
        return []

    lines: list[str] = []
    for item in structured[:8]:
        if not isinstance(item, Mapping):
            continue
        raw_name = str(item.get("name") or "").strip().lower()
        if not raw_name:
            continue
        display = str(item.get("display_name") or raw_name).strip() or raw_name
        transport = str(item.get("transport") or "mcp").strip() or "mcp"
        prefix = f"mcp_{raw_name}_"
        if configured_server_names is not None and raw_name not in configured_server_names:
            lines.append(
                "MCP Preset Attachment: "
                f"@{raw_name} ({display}; transport={transport}) is configured in WebUI Settings, "
                "but this gateway has not loaded the latest MCP settings yet. "
                f"Tools with prefix `{prefix}` may not be available yet; if they are missing, "
                "tell the user to restart nanobot."
            )
            continue
        if connected_server_names is not None and raw_name not in connected_server_names:
            lines.append(
                "MCP Preset Attachment: "
                f"@{raw_name} ({display}; transport={transport}) is configured, "
                "but its MCP connection is not currently live. "
                f"Tools with prefix `{prefix}` may be unavailable; tell the user to open Settings, "
                "run the preset test, and restart nanobot only if hot reload is unavailable."
            )
            continue
        lines.append(
            "MCP Preset Attachment: "
            f"@{raw_name} ({display}; transport={transport}; tool_prefix={prefix}). "
            f"Prefer available tools whose names start with `{prefix}` for this request; "
            "do not substitute shell commands for this MCP integration unless the user asks."
        )
    return lines


async def connect_missing_servers(state: Any, registry: ToolRegistry) -> None:
    """把已配置但当前未连上的 MCP server 补连上。
    
    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
    missing_servers = {
        name: cfg for name, cfg in state._mcp_servers.items() if name not in state._mcp_stacks
    }
    if state._mcp_connecting or not missing_servers:
        return
    state._mcp_connecting = True
    try:
        connected = await connect_mcp_servers(missing_servers, registry)
        state._mcp_stacks.update(connected)
        _attach_reconnect_handlers(state, registry, connected)
        state._mcp_connected = bool(state._mcp_stacks)
        if connected:
            logger.info("MCP connected servers: {}", sorted(connected))
        else:
            logger.warning("No MCP servers connected successfully (will retry next message)")
    except asyncio.CancelledError:
        logger.warning("MCP connection cancelled (will retry next message)")
        state._mcp_connected = bool(state._mcp_stacks)
    except BaseException as e:
        logger.warning("Failed to connect MCP servers (will retry next message): {}", e)
        state._mcp_connected = bool(state._mcp_stacks)
    finally:
        state._mcp_connecting = False


async def reload_servers(state: Any, registry: ToolRegistry) -> dict[str, Any]:
    """根据最新配置文件，对在线 MCP 连接做热重载对账。
    
    实现方法：从配置、内置目录或入口点发现候选项，过滤不可用项后注册到运行时。"""
    async with _reload_lock(state):
        try:
            from nanobot.config.loader import load_config, resolve_config_env_vars

            config = resolve_config_env_vars(load_config())
            next_servers = dict(config.tools.mcp_servers)
        except Exception as exc:
            logger.warning("MCP hot reload could not read config: {}", exc)
            return {
                "ok": False,
                "message": "Could not reload MCP config. Restart nanobot to pick up changes.",
                "requires_restart": True,
                "error": str(exc),
            }

        current_servers = dict(state._mcp_servers)
        current_names = set(current_servers)
        next_names = set(next_servers)
        removed = sorted(current_names - next_names)
        added = sorted(next_names - current_names)
        changed = sorted(
            name
            for name in current_names & next_names
            if _server_signature(current_servers[name]) != _server_signature(next_servers[name])
        )

        tools_removed = 0
        for name in [*removed, *changed]:
            tools_removed += _unregister_server_tools(state, registry, name)
            await _close_server(state, name)

        state._mcp_servers = next_servers
        retry_missing = sorted(
            name
            for name in next_names
            if name not in state._mcp_stacks and name not in set(added) | set(changed)
        )
        to_connect_names = sorted(set(added) | set(changed) | set(retry_missing))
        to_connect = {name: next_servers[name] for name in to_connect_names}
        connected: dict[str, AsyncExitStack] = {}
        if to_connect:
            connected = await connect_mcp_servers(to_connect, registry)
            state._mcp_stacks.update(connected)
            _attach_reconnect_handlers(state, registry, connected)

        state._mcp_connected = bool(state._mcp_stacks)
        failed = sorted(set(to_connect) - set(connected))
        unchanged = not removed and not added and not changed and not retry_missing
        ok = not failed
        if failed:
            message = "MCP config reloaded, but some servers did not connect: " + ", ".join(failed)
        elif unchanged:
            message = "MCP config is already live."
        elif retry_missing and not added and not changed and not removed:
            message = "MCP connections refreshed without restarting nanobot."
        else:
            message = "MCP config reloaded without restarting nanobot."

        logger.info(
            "MCP hot reload: added={} changed={} removed={} retried={} connected={} failed={} tools_removed={}",
            added,
            changed,
            removed,
            retry_missing,
            sorted(connected),
            failed,
            tools_removed,
        )
        return {
            "ok": ok,
            "message": message,
            "added": added,
            "changed": changed,
            "removed": removed,
            "retried": retry_missing,
            "connected": sorted(state._mcp_stacks),
            "configured": sorted(state._mcp_servers),
            "failed": failed,
            "tools_removed": tools_removed,
            "requires_restart": False,
        }


async def request_mcp_reload(bus: Any, *, timeout: float = 15.0) -> dict[str, Any]:
    """请求运行中的 AgentLoop 重新对账 MCP 连接。
    
    实现方法：从配置、内置目录或入口点发现候选项，过滤不可用项后注册到运行时。"""
    loop = asyncio.get_running_loop()
    ack: asyncio.Future[dict[str, Any]] = loop.create_future()
    await bus.publish_inbound(
        InboundMessage(
            channel="system",
            sender_id="webui-settings",
            chat_id="runtime",
            content=RUNTIME_CONTROL_MCP_RELOAD,
            metadata={
                INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_MCP_RELOAD,
                RUNTIME_CONTROL_ACK: ack,
            },
        )
    )
    try:
        result = await asyncio.wait_for(ack, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "message": "MCP hot reload timed out. Restart nanobot to pick up changes.",
            "requires_restart": True,
        }
    return result if isinstance(result, dict) else {
        "ok": False,
        "message": "MCP hot reload returned an unexpected response.",
        "requires_restart": True,
    }


async def handle_runtime_control(state: Any, msg: InboundMessage, registry: ToolRegistry) -> bool:
    """handle runtime control。
    
    实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
    metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
    control = metadata.get(INBOUND_META_RUNTIME_CONTROL)
    if control != RUNTIME_CONTROL_MCP_RELOAD:
        return False

    ack = metadata.get(RUNTIME_CONTROL_ACK)
    try:
        result = await reload_servers(state, registry)
    except Exception as exc:
        logger.exception("MCP hot reload failed")
        result = {
            "ok": False,
            "message": "MCP hot reload failed. Restart nanobot to pick up changes.",
            "requires_restart": True,
            "error": str(exc),
        }
    if isinstance(ack, asyncio.Future) and not ack.done():
        ack.set_result(result)
    return True


def _reload_lock(state: Any) -> asyncio.Lock:
    """reload lock。
    
    实现方法：从配置、内置目录或入口点发现候选项，过滤不可用项后注册到运行时。"""
    try:
        return _RELOAD_LOCKS[state]
    except KeyError:
        lock = asyncio.Lock()
        _RELOAD_LOCKS[state] = lock
        return lock


def _attach_reconnect_handlers(
    state: Any,
    registry: ToolRegistry,
    server_names: Mapping[str, Any] | set[str] | list[str] | tuple[str, ...],
) -> None:
    """attach reconnect handlers。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    async def reconnect(server_name: str, tool_name: str, stale_tool: Tool) -> Tool | None:
        """reconnect。
        
        实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
        return await _refresh_terminated_server(
            state,
            registry,
            server_name,
            tool_name,
            stale_tool,
        )

    for server_name in server_names:
        prefix = _tool_prefix(server_name)
        for tool_name in list(registry.tool_names):
            if not tool_name.startswith(prefix):
                continue
            tool = registry.get(tool_name)
            if isinstance(tool, _MCPWrapperBase):
                tool.set_reconnect_handler(reconnect)


async def _refresh_terminated_server(
    state: Any,
    registry: ToolRegistry,
    server_name: str,
    tool_name: str,
    stale_tool: Tool,
) -> Tool | None:
    """refresh terminated server。
    
    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
    async with _reload_lock(state):
        cfg = state._mcp_servers.get(server_name)
        if cfg is None:
            logger.warning(
                "MCP server '{}' session terminated but is no longer configured",
                server_name,
            )
            return None

        current_tool = registry.get(tool_name)
        if (
            current_tool is not None
            and current_tool is not stale_tool
            and server_name in state._mcp_stacks
        ):
            return current_tool

        logger.warning("MCP server '{}' session terminated; refreshing connection", server_name)
        _unregister_server_tools(state, registry, server_name)
        await _close_server(state, server_name)

        connected = await connect_mcp_servers({server_name: cfg}, registry)
        state._mcp_stacks.update(connected)
        _attach_reconnect_handlers(state, registry, connected)
        state._mcp_connected = bool(state._mcp_stacks)
        if server_name not in connected:
            logger.warning("MCP server '{}' reconnect failed after session termination", server_name)
            return None
        return registry.get(tool_name)


def _server_signature(cfg: Any) -> Any:
    """server signature。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    return cfg


def _tool_prefix(server_name: str) -> str:
    """tool prefix。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return _sanitize_name(f"mcp_{server_name}_")


def _unregister_server_tools(state: Any, registry: ToolRegistry, server_name: str) -> int:
    """unregister server tools。
    
    实现方法：把对象写入内部映射表，并清理依赖该映射生成的缓存。"""
    prefix = _tool_prefix(server_name)
    removed = 0
    for tool_name in list(registry.tool_names):
        if tool_name.startswith(prefix):
            registry.unregister(tool_name)
            removed += 1
    return removed


async def _close_server(state: Any, server_name: str) -> None:
    """close server。
    
    实现方法：按已登记的异步清理栈释放连接、会话或后台资源，并重置连接状态。"""
    stack = state._mcp_stacks.pop(server_name, None)
    if stack is None:
        return
    try:
        await stack.aclose()
    except (RuntimeError, BaseExceptionGroup):
        logger.debug("MCP server '{}' cleanup error (can be ignored)", server_name)
