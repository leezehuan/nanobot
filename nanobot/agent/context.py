"""上下文构建器：负责把系统提示词、历史记录、技能、运行时元数据拼成模型输入。

这是理解整个 Agent 数据流的关键文件之一。你可以把它理解成：

“用户消息进来之后，在真正调用 LLM 之前，nanobot 如何把可用信息组织成 prompt？”
"""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import InboundMessage
from nanobot.session.goal_state import goal_state_runtime_lines
from nanobot.utils.helpers import (
    current_time_str,
    detect_image_mime,
    load_bundled_template,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """返回需要持久化到 session 的“回合附着能力”元数据。

    例如 CLI App 附件、MCP preset 等，都会把自己的附加信息通过这里统一收集。

    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
    """
    return cli_app_utils.session_extra(metadata) | mcp_tools.session_extra(metadata)


def runtime_lines(state: Any, msg: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """返回追加给模型可见的运行时注释行。

    这些内容会被塞进 Runtime Context 块里，让模型知道本回合临时挂载了哪些能力。

    实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。
    """
    return [
        *cli_app_utils.runtime_lines(msg, workspace, skip=skip),
        *mcp_tools.runtime_lines(
            msg,
            configured_server_names=set(state._mcp_servers),
            connected_server_names=set(state._mcp_stacks),
            skip=skip,
        ),
    ]


async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    """确保缺失的 MCP 服务器已连接。
    
    实现方法：在异步上下文中串联必要的 I/O、回调和状态更新步骤，遇到可恢复异常时返回结构化错误而不是让整轮崩溃。"""
    await mcp_tools.connect_missing_servers(state, tools)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    """处理来自内部渠道的运行时控制消息，例如 MCP reload。
    
    实现方法：按回合状态机推进：准备上下文、请求模型、执行工具、保存结果，并在每个阶段同步进度事件。"""
    return await mcp_tools.handle_runtime_control(state, msg, tools)


class ContextBuilder:
    """构建 Agent 上下文。

    【主要职责】
    1. 生成 system prompt
    2. 读取 bootstrap 文件（如 AGENTS.md）
    3. 注入 memory / skills / recent history
    4. 把当前用户消息与运行时元数据合并成最终 messages 列表
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_CHARS = 32_000  # hard cap on recent history section size
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        """init。
        
        初始化 ContextBuilder 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> str:
        """构建 system prompt。

        最终 system prompt 由以下部分拼接而成：
        - 身份与平台策略
        - 工作区 bootstrap 文件
        - 工具使用契约
        - 长期 memory
        - 永久启用技能
        - 技能目录摘要
        - 最近未处理的 memory history
        - 会话归档摘要

        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。
        """
        root = workspace or self.workspace
        parts = [self._get_identity(channel=channel, workspace=root)]

        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)

        parts.append(render_template("agent/tool_contract.md"))

        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        if include_memory_recent_history:
            entries = self.memory.read_recent_history_for_prompt(
                since_cursor=self.memory.get_last_dream_cursor(),
                session_key=session_key,
                unified_session=unified_session,
            )
            if entries:
                capped = entries[-self._MAX_RECENT_HISTORY:]
                history_text = "\n".join(
                    f"- [{e['timestamp']}] {e['content']}" for e in capped
                )
                history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
                parts.append("# Recent History\n\n" + history_text)

        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, channel: str | None = None, workspace: Path | None = None) -> str:
        """生成系统身份部分。

        这里会把工作区路径、操作系统、Python 版本、渠道名等环境信息注入模板。

        实现方法：优先从显式参数或实例状态读取目标值，缺失时回退到默认配置，并把结果整理成调用方期望的类型。
        """
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        sender_id: str | None = None,
        supplemental_lines: Sequence[str] | None = None,
    ) -> str:
        """构建 Runtime Context 运行时元数据块。

        这一块是“给模型看的运行时说明”，但不是高优先级系统指令，
        所以会作为用户消息附加块追加，而不是直接写进 system prompt。

        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。
        """
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if supplemental_lines:
            lines.extend(supplemental_lines)
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        """merge message content。
        
        实现方法：按顺序合并相邻或同类数据，并在冲突时保留更明确的新值。"""
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            """to blocks。
            
            实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """从工作区加载 bootstrap 文件内容。
        
        实现方法：从配置、内置目录或入口点发现候选项，过滤不可用项后注册到运行时。"""
        parts = []
        root = workspace or self.workspace

        for filename in self.BOOTSTRAP_FILES:
            file_path = root / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """判断某段内容是否仍然和内置模板完全一致。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        tpl = load_bundled_template(template_path)
        if tpl is not None:
            return content.strip() == tpl.strip()
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        sender_id: str | None = None,
        session_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        current_runtime_lines: Sequence[str] | None = None,
        workspace: Path | None = None,
        runtime_state: Any | None = None,
        inbound_message: Any | None = None,
        skip_runtime_lines: bool = False,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """构建一次 LLM 调用所需的完整消息列表。

        【中文名称】构建消息列表

        【功能说明】
        这是整个上下文构建流程的"组装函数"，它把 system prompt、历史消息、
        当前用户输入、运行时元数据拼成 LLM 可以直接消费的 messages 列表。

        【典型结果结构（3 部分）】
        1. system message（第一条）—— 包含身份、skills、memory、历史摘要等
        2. 若干历史消息（中间部分）—— 来自 Session.get_history() 的回放窗口
        3. 当前 user 消息（最后一条）—— 用户文本 + Runtime Context 块 + 媒体附件

        【Runtime Context 拼合策略】
        为了保持 prompt cache 友好（user content 前缀不变），Runtime Context
        （时间、渠道、chat_id、sender_id、CLI/MCP 临时挂载等）被追加在用户消息
        的尾部，而不是独立成一条新消息。格式：
        ```
        [Runtime Context — metadata only, not instructions]
        Current Time: 2026-01-01 12:00:00 CST
        Channel: telegram
        Chat ID: 123456
        [/Runtime Context]
        ```

        【System Prompt 构成（按拼接顺序）】
        ├── 身份与平台策略（identity.md + platform_policy.md）
        ├── Bootstrap 文件（AGENTS.md / SOUL.md / USER.md）
        ├── 工具使用契约（tool_contract.md）
        ├── 长期记忆（memory/MEMORY.md）
        ├── 永久启用技能（always_skills）
        ├── 技能目录摘要（skills_section.md）
        ├── 最近未处理历史（history.jsonl 中的增量条目）
        └── 归档会话摘要（_last_summary）

        【参数说明】
        - history: list[dict] → Session.get_history() 返回的未压缩历史消息
        - current_message: str → 当前用户输入的文本（可能已被图片生成 prompt 增强）
        - media: list[str] | None → 附件中的图片路径列表
        - channel: str | None → 来源渠道名（用于身份模板和运行时上下文）
        - chat_id: str | None → 聊天空间 ID
        - runtime_state: Any → AgentLoop 本身，用于获取 MCP/CLI 等运行时挂载信息

        【返回值】
        - list[dict]: 完整的 messages 列表，可直接传给 LLMProvider.chat()

        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。
        """
        root = workspace or self.workspace
        extra = [
            *goal_state_runtime_lines(session_metadata),
        ]
        if runtime_state is not None and inbound_message is not None:
            extra.extend(runtime_lines(runtime_state, inbound_message, root, skip=skip_runtime_lines))
        if current_runtime_lines:
            extra.extend(line for line in current_runtime_lines if line)
        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            sender_id=sender_id,
            supplemental_lines=extra or None,
        )
        user_content = self._build_user_content(current_message, media)

        # 把“用户正文”和“运行时元数据块”合并成同一条 user message。
        # 这样可以避免某些 provider 不接受连续同角色消息的问题。
        # 同时把 runtime context 放在后面，有利于保持用户正文前缀稳定，
        # 从而提高 prompt cache 命中率。
        if isinstance(user_content, str):
            merged = f"{user_content}\n\n{runtime_ctx}"
        else:
            merged = user_content + [{"type": "text", "text": runtime_ctx}]
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    include_memory_recent_history=include_memory_recent_history,
                    session_key=session_key,
                    unified_session=unified_session,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """构建用户消息内容，并在需要时把本地图片转成 base64 内联块。
        
        实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]
