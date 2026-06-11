"""嵌入式 WebUI 网关的服务装配器。

【中文名称】网关服务拼装层

这个模块本身不处理 HTTP 请求，也不直接处理 WebSocket 消息。
它做的是“依赖注入”工作：把 WebUI 网关运行所需的一组服务对象组装好，
再一次性返回给上层。

这样做的好处是：

- 网关启动代码更清爽；
- 各个服务的职责边界更清楚；
- 测试时也更容易替换单个依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger as default_logger

from nanobot.webui.gateway_tokens import GatewayTokenStore
from nanobot.webui.media_gateway import WebUIMediaGateway
from nanobot.webui.transcript import WebUITranscriptRecorder
from nanobot.webui.workspaces import WebUIWorkspaceController
from nanobot.webui.ws_http import GatewayHTTPHandler


@dataclass(frozen=True)
class GatewayServices:
    """WebUI 网关需要共享的一组明确依赖。"""

    http: GatewayHTTPHandler
    tokens: GatewayTokenStore
    media: WebUIMediaGateway
    transcripts: WebUITranscriptRecorder
    workspaces: WebUIWorkspaceController
    session_manager: Any | None
    cron_service: Any | None


def build_gateway_services(
    *,
    config: Any,
    bus: Any,
    session_manager: Any | None,
    static_dist_path: Path | None,
    workspace_path: Path,
    default_restrict_to_workspace: bool,
    runtime_model_name: Any | None,
    runtime_surface: str,
    runtime_capabilities_overrides: dict[str, Any] | None,
    disabled_skills: set[str] | None = None,
    cron_service: Any | None = None,
    logger: Any = default_logger,
) -> GatewayServices:
    """构建一整套 WebUI 网关运行时服务。

    可以把这个函数理解成“后端网关的装配工厂”：

    - token 服务负责鉴权票据
    - media 服务负责媒体签名和改写
    - transcript 服务负责会话文本记录
    - workspaces 服务负责工作区访问策略
    - http 服务负责真正的 HTTP / WS 路由处理
    """
    tokens = GatewayTokenStore()
    media = WebUIMediaGateway(
        workspace_path=workspace_path,
        logger=logger,
    )
    transcripts = WebUITranscriptRecorder(log=logger)
    workspaces = WebUIWorkspaceController(
        session_manager=session_manager,
        default_workspace=workspace_path,
        default_restrict_to_workspace=default_restrict_to_workspace,
    )
    http = GatewayHTTPHandler(
        config=config,
        session_manager=session_manager,
        static_dist_path=static_dist_path,
        runtime_model_name=runtime_model_name,
        runtime_surface=runtime_surface,
        runtime_capabilities_overrides=runtime_capabilities_overrides,
        bus=bus,
        tokens=tokens,
        media=media,
        workspaces=workspaces,
        skills_workspace_path=workspace_path,
        disabled_skills=disabled_skills,
        cron_service=cron_service,
        log=logger,
    )
    return GatewayServices(
        http=http,
        tokens=tokens,
        media=media,
        transcripts=transcripts,
        workspaces=workspaces,
        session_manager=session_manager,
        cron_service=cron_service,
    )
