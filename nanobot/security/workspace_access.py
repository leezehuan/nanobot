"""工作区访问范围与沙箱状态解析工具。

【中文名称】工作区访问作用域

这个模块主要回答两个问题：

1. 当前这一轮 Agent 能访问哪个项目目录？
2. 这种限制是“系统沙箱强制的”，还是“nanobot 应用层自己约束的”？

它既服务 WebUI 的“Full Access / Restricted”切换，也服务工具层做
路径、回环访问等安全判断。
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

WorkspaceAccessMode = Literal["restricted", "full"]
WORKSPACE_SCOPE_METADATA_KEY = "workspace_scope"
_ACCESS_MODES = {"restricted", "full"}

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled", ""}
_PROVIDER_LABELS = {
    "none": "None",
    "unknown": "Unknown system sandbox",
    "macos_app_sandbox": "macOS App Sandbox",
    "bwrap": "Bubblewrap",
}

_CURRENT_WORKSPACE_SCOPE: ContextVar["WorkspaceScope | None"] = ContextVar(
    "nanobot_workspace_scope",
    default=None,
)


class WorkspaceScopeError(ValueError):
    """当 WebUI 请求的工作区作用域非法时抛出的错误。"""

    status = 400

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class WorkspaceSandboxStatus:
    """当前宿主环境下的工作区沙箱状态快照。

    这份信息既能给前端显示，也能给工具层判断“当前限制到底有多硬”。
    """

    restrict_to_workspace: bool
    workspace_root: str
    level: str
    enforced: bool
    provider: str
    provider_label: str
    summary: str

    def as_dict(self) -> dict[str, object]:
        """转换为适合 API / WebSocket 传输的普通字典。"""
        return {
            "restrict_to_workspace": self.restrict_to_workspace,
            "workspace_root": self.workspace_root,
            "level": self.level,
            "enforced": self.enforced,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class WorkspaceScope:
    """某一轮 Agent 实际生效的工作区作用域。"""

    project_path: Path
    access_mode: WorkspaceAccessMode
    restrict_to_workspace: bool
    sandbox_status: WorkspaceSandboxStatus
    source_channel: str | None = None

    @property
    def project_name(self) -> str:
        """返回项目目录的人类可读名称。"""
        return self.project_path.name or str(self.project_path)

    def metadata(self) -> dict[str, str]:
        """生成适合持久化到 session metadata 的精简作用域信息。"""
        return {
            "project_path": str(self.project_path),
            "access_mode": self.access_mode,
        }

    def payload(self) -> dict[str, Any]:
        """生成适合返回给前端的完整作用域载荷。"""
        return {
            **self.metadata(),
            "project_name": self.project_name,
            "restrict_to_workspace": self.restrict_to_workspace,
            "sandbox_status": self.sandbox_status.as_dict(),
        }


@dataclass(frozen=True)
class ToolWorkspace:
    """某一次工具调用解析后的工作区策略。"""

    project_path: Path | None
    restrict_to_workspace: bool
    scope: WorkspaceScope | None = None

    @property
    def allowed_root(self) -> Path | None:
        """返回当前工具真正受限的根目录；如果未受限则为 `None`。"""
        if self.restrict_to_workspace and self.project_path is not None:
            return self.project_path
        return None


@dataclass(frozen=True)
class WorkspaceScopeResolver:
    """在 Agent 回合边界解析最终生效的工作区作用域。"""

    default_workspace: str | Path
    default_restrict_to_workspace: bool
    scoped_channel: str = "websocket"

    @property
    def sandbox_status(self) -> WorkspaceSandboxStatus:
        """读取默认工作区对应的沙箱状态。"""
        return self.default().sandbox_status

    def default(self) -> WorkspaceScope:
        """返回默认作用域。未指定特殊请求时，就使用它。"""
        return default_workspace_scope(
            self.default_workspace,
            self.default_restrict_to_workspace,
        )

    def for_message(
        self,
        msg: Any,
        session_metadata: Any,
    ) -> WorkspaceScope:
        """根据一条入站消息和会话元数据解析作用域。"""
        return self.for_turn(
            channel=getattr(msg, "channel", None),
            message_metadata=getattr(msg, "metadata", None),
            session_metadata=session_metadata,
        )

    def for_turn(
        self,
        *,
        channel: str | None,
        message_metadata: Any,
        session_metadata: Any,
    ) -> WorkspaceScope:
        """在一轮处理开始时解析有效作用域。

        只有指定频道（默认是 WebSocket/WebUI）允许覆盖默认作用域；
        其它频道统一走默认值，避免外部渠道任意提升访问能力。
        """
        if channel != self.scoped_channel:
            return self.default()
        return resolve_effective_workspace_scope(
            message_metadata=message_metadata,
            session_metadata=session_metadata,
            default_workspace=self.default_workspace,
            default_restrict_to_workspace=self.default_restrict_to_workspace,
            source_channel=channel,
        )

    def persist_message_scope(self, session: Any, msg: Any) -> None:
        """把消息里声明的作用域写回会话，供后续轮次继承。"""
        if getattr(msg, "channel", None) != self.scoped_channel:
            return
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            return
        raw = metadata.get(WORKSPACE_SCOPE_METADATA_KEY)
        if isinstance(raw, dict):
            session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = dict(raw)


def workspace_sandbox_status(
    *,
    restrict_to_workspace: bool,
    workspace: str | Path,
    environ: dict[str, str] | None = None,
) -> WorkspaceSandboxStatus:
    """推断当前宿主环境如何实施工作区限制。"""

    workspace_root = str(Path(workspace).expanduser().resolve(strict=False))
    provider = _env_system_provider(environ)
    if not restrict_to_workspace:
        return WorkspaceSandboxStatus(
            restrict_to_workspace=False,
            workspace_root=workspace_root,
            level="off",
            enforced=False,
            provider="none",
            provider_label=_provider_label("none"),
            summary="Workspace restriction is disabled.",
        )

    if provider:
        label = _provider_label(provider)
        return WorkspaceSandboxStatus(
            restrict_to_workspace=True,
            workspace_root=workspace_root,
            level="system",
            enforced=True,
            provider=provider,
            provider_label=label,
            summary=f"Workspace restriction is system-enforced by {label}.",
        )

    return WorkspaceSandboxStatus(
        restrict_to_workspace=True,
        workspace_root=workspace_root,
        level="application",
        enforced=False,
        provider="none",
        provider_label=_provider_label("none"),
        summary="Workspace restriction uses nanobot application-level guards.",
    )


def default_access_mode(restrict_to_workspace: bool) -> WorkspaceAccessMode:
    """把布尔限制开关映射成对外展示的访问模式字符串。"""
    return "restricted" if restrict_to_workspace else "full"


def build_workspace_scope(
    project_path: str | Path,
    access_mode: str,
    *,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """从路径和访问模式直接构造 `WorkspaceScope`。"""
    mode = _normalize_access_mode(access_mode)
    root = Path(project_path).expanduser().resolve(strict=False)
    restrict = mode == "restricted"
    return WorkspaceScope(
        project_path=root,
        access_mode=mode,
        restrict_to_workspace=restrict,
        sandbox_status=workspace_sandbox_status(
            restrict_to_workspace=restrict,
            workspace=root,
        ),
        source_channel=source_channel,
    )


def default_workspace_scope(
    workspace: str | Path,
    restrict_to_workspace: bool,
    *,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """基于默认配置构造默认作用域。"""
    return build_workspace_scope(
        workspace,
        default_access_mode(restrict_to_workspace),
        source_channel=source_channel,
    )


def validate_workspace_scope_payload(
    raw: Any,
    *,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """校验客户端提交的工作区作用域请求。

    这里会严格检查：

    - payload 必须是对象
    - `project_path` 必须是绝对路径
    - 目标目录必须存在
    - `access_mode` 必须是支持的值
    """
    if raw is None:
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )
    if not isinstance(raw, dict):
        raise WorkspaceScopeError("workspace_scope must be an object")

    raw_path = raw.get("project_path") or raw.get("path")
    if raw_path is None or raw_path == "":
        raw_path = str(Path(default_workspace).expanduser().resolve(strict=False))
    if not isinstance(raw_path, str):
        raise WorkspaceScopeError("project_path must be a string")
    if "\0" in raw_path:
        raise WorkspaceScopeError("project_path contains invalid characters")

    project = Path(raw_path).expanduser()
    if not project.is_absolute():
        raise WorkspaceScopeError("project_path must be absolute")
    project = project.resolve(strict=False)
    if not project.is_dir():
        raise WorkspaceScopeError("project_path must be an existing directory")

    raw_mode = raw.get("access_mode")
    if raw_mode is None:
        raw_mode = default_access_mode(default_restrict_to_workspace)
    if not isinstance(raw_mode, str):
        raise WorkspaceScopeError("access_mode must be a string")
    return build_workspace_scope(project, raw_mode, source_channel=source_channel)


def workspace_scope_from_metadata(
    metadata: Any,
    *,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """从持久化 metadata 中恢复作用域，失败时安全回退到默认值。"""
    if not isinstance(metadata, dict):
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )
    try:
        return validate_workspace_scope_payload(
            metadata.get(WORKSPACE_SCOPE_METADATA_KEY),
            default_workspace=default_workspace,
            default_restrict_to_workspace=default_restrict_to_workspace,
            source_channel=source_channel,
        )
    except WorkspaceScopeError:
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )


def resolve_effective_workspace_scope(
    *,
    message_metadata: Any,
    session_metadata: Any,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """综合消息级与会话级信息，解析最终生效的作用域。"""
    if isinstance(message_metadata, dict) and WORKSPACE_SCOPE_METADATA_KEY in message_metadata:
        return workspace_scope_from_metadata(
            message_metadata,
            default_workspace=default_workspace,
            default_restrict_to_workspace=default_restrict_to_workspace,
            source_channel=source_channel,
        )
    return workspace_scope_from_metadata(
        session_metadata,
        default_workspace=default_workspace,
        default_restrict_to_workspace=default_restrict_to_workspace,
        source_channel=source_channel,
    )


def bind_workspace_scope(scope: WorkspaceScope) -> Token[WorkspaceScope | None]:
    """把某轮回合的作用域绑定到当前上下文。"""
    return _CURRENT_WORKSPACE_SCOPE.set(scope)


def reset_workspace_scope(token: Token[WorkspaceScope | None]) -> None:
    """在回合结束后恢复之前的作用域上下文。"""
    _CURRENT_WORKSPACE_SCOPE.reset(token)


def current_workspace_scope() -> WorkspaceScope | None:
    """读取当前上下文中已绑定的工作区作用域。"""
    return _CURRENT_WORKSPACE_SCOPE.get()


def current_tool_workspace(
    default_workspace: str | Path | None,
    *,
    restrict_to_workspace: bool = False,
    sandbox_restricts_workspace: bool = False,
) -> ToolWorkspace:
    """返回当前工具调用实际应遵守的工作区访问策略。"""

    scope = current_workspace_scope()
    project_path = (
        scope.project_path
        if scope is not None
        else Path(default_workspace).expanduser() if default_workspace is not None else None
    )
    restrict = (
        scope.restrict_to_workspace
        if scope is not None
        else bool(restrict_to_workspace)
    ) or sandbox_restricts_workspace
    return ToolWorkspace(
        project_path=project_path,
        restrict_to_workspace=restrict,
        scope=scope,
    )


def current_scope_allows_loopback(*, enabled: bool) -> bool:
    """判断当前回合是否允许访问本机回环地址。"""

    scope = current_workspace_scope()
    return bool(
        enabled
        and scope is not None
        and scope.source_channel == "websocket"
        and scope.access_mode == "full"
        and not scope.restrict_to_workspace
    )


def _env_system_provider(environ: dict[str, str] | None = None) -> str | None:
    """从环境变量推断是否存在系统级工作区沙箱。"""
    env = environ if environ is not None else os.environ
    explicit_provider = env.get("NANOBOT_WORKSPACE_SANDBOX_PROVIDER")
    enforced = env.get("NANOBOT_WORKSPACE_SANDBOX_ENFORCED")
    compatibility = env.get("NANOBOT_SANDBOX_ENFORCED")

    marker = enforced if enforced is not None else compatibility
    if marker is None:
        return None

    normalized_marker = marker.strip().lower()
    if normalized_marker in _FALSE_VALUES:
        return None
    if normalized_marker in _TRUE_VALUES:
        return _normalize_provider(explicit_provider)
    return _normalize_provider(marker)


def _normalize_provider(value: str | None) -> str:
    """把 provider 名称规范化成内部统一格式。"""
    if not value:
        return "unknown"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "unknown"


def _provider_label(provider: str) -> str:
    """把内部 provider 标识转成前端更适合展示的标签。"""
    if provider in _PROVIDER_LABELS:
        return _PROVIDER_LABELS[provider]
    return provider.replace("_", " ").title()


def _normalize_access_mode(value: str) -> WorkspaceAccessMode:
    """把用户输入的访问模式标准化成受支持的内部值。"""
    mode = value.strip().lower().replace("_", "-")
    if mode == "restrict":
        mode = "restricted"
    if mode == "full-access":
        mode = "full"
    if mode not in _ACCESS_MODES:
        raise WorkspaceScopeError("access_mode must be restricted or full")
    return mode  # type: ignore[return-value]
