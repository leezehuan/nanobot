"""会话管理：负责对话历史的内存缓存与 JSONL 持久化。

这是 nanobot 的另一块核心骨架。它回答的是：

- 每个聊天会话的历史消息放在哪里？
- 如何从磁盘恢复它？
- 如何保证写入尽量安全、原子？
- 历史过长时如何截断、归档、压缩？
"""

import json
import os
import re
import shutil
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    find_legal_message_start,
    image_placeholder_text,
    safe_filename,
    strip_think,
)
from nanobot.utils.subagent_channel_display import scrub_subagent_announce_body

FILE_MAX_MESSAGES = 2000
_MESSAGE_TIME_PREFIX_RE = re.compile(r"^\[Message Time: [^\]]+\]\n?")
_LOCAL_IMAGE_BREADCRUMB_RE = re.compile(r"^\[image: (?:/|~)[^\]]+\]\s*$")
_TOOL_CALL_ECHO_RE = re.compile(r'^\s*(?:generate_image|message)\([^)]*\)\s*$')
_SESSION_PREVIEW_MAX_CHARS = 120
_SESSION_LIST_PREVIEW_MAX_RECORDS = 200
_SESSION_LIST_PREVIEW_MAX_CHARS = 1_000_000
_FORK_VOLATILE_METADATA_KEYS = {
    "goal_state",
    "pending_user_turn",
    "runtime_checkpoint",
    "thread_goal",
    "title",
    "title_user_edited",
}


def _sanitize_assistant_replay_text(content: str) -> str:
    """清理 assistant 历史回放文本中的内部痕迹。

    某些内部标记如果被原样喂回模型，就会变成“示范样本”，导致模型在新回复里也
    学着输出这些标记。因此这里会在历史回放前先做一次去污染。
    """
    content = _MESSAGE_TIME_PREFIX_RE.sub("", content, count=1)
    lines = [
        line
        for line in content.splitlines()
        if not _LOCAL_IMAGE_BREADCRUMB_RE.match(line)
        and not _TOOL_CALL_ECHO_RE.match(line)
    ]
    return "\n".join(lines).strip()


def _text_preview(content: Any) -> str:
    """生成用于会话列表展示的短预览文本。"""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = " ".join(parts)
    else:
        return ""
    text = _sanitize_assistant_replay_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _SESSION_PREVIEW_MAX_CHARS:
        text = text[: _SESSION_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
    return text


def _message_preview_text(message: dict[str, Any]) -> str:
    """为会话列表生成消息预览；对子 Agent 注入内容做额外缩略。"""
    content: Any = message.get("content")
    if message.get("injected_event") == "subagent_result" and isinstance(content, str):
        content = scrub_subagent_announce_body(content)
    return _text_preview(content)


def _metadata_title(metadata: Any) -> str:
    """从 session metadata 中取标题，并在必要时去掉 think 痕迹。"""
    if not isinstance(metadata, dict):
        return ""
    title = metadata.get("title")
    if not isinstance(title, str):
        return ""
    if metadata.get("title_user_edited") is True:
        return title
    return strip_think(title)


@dataclass
class Session:
    """单个对话会话对象。

    一个 Session 可以理解为“某个聊天线程/会话的完整历史容器”，其中保存：
    - messages：消息序列
    - metadata：附加状态，如 goal_state、标题、checkpoint 等
    - last_consolidated：已经被压缩归档过的历史边界
    """

    key: str  # key = 会话唯一键，通常是 channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # 已经被压缩/归档进 memory 文件的消息数量

    def __post_init__(self) -> None:
        # 如果 last_consolidated 超出范围，说明元数据可能损坏。
        # 不修正的话会导致历史被“错误隐藏”，因此这里主动回退到 0。
        if (
            isinstance(self.last_consolidated, bool)
            or not isinstance(self.last_consolidated, int)
            or not 0 <= self.last_consolidated <= len(self.messages)
        ):
            self.last_consolidated = 0

    @staticmethod
    def _annotate_message_time(message: dict[str, Any], content: Any) -> Any:
        """给模型暴露消息时间戳，帮助相对时间推理。

        这里非常克制：只给 user turn 打时间戳，不给 assistant turn 打。
        否则模型会把 ``[Message Time: ...]`` 当成示范格式学走，最终泄漏到用户回复里。
        """
        timestamp = message.get("timestamp")
        if not timestamp or not isinstance(content, str):
            return content
        role = message.get("role")
        if role != "user":
            return content
        return f"[Message Time: {timestamp}]\n{content}"

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """向当前会话追加一条消息。"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(
        self,
        max_messages: int = 120,
        *,
        max_tokens: int = 0,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """返回喂给 LLM 的未压缩历史消息。

        【中文名称】获取回放历史

        【功能说明】
        这是 Session 最重要的方法之一，决定"每次调用 LLM 时，模型能看到哪些上下文"。
        它只返回 ``last_consolidated`` 之后未压缩的尾部消息，并经过多层裁剪确保
        不超出上下文窗口预算。

        调用链路：
        AgentLoop._state_build() → session.get_history() → 构建成 messages → 发给 LLM

        【5 个处理步骤】
        Step 1: 切片 → 只取 last_consolidated 之后的"未压缩尾部"
        Step 2: 消息条数裁剪 → 按 max_messages 保留最近 N 条
        Step 3: Token 预算裁剪 → 按 max_tokens 从尾部反推保留
        Step 4: 边界修正 → 确保窗口不从孤儿 tool result 或 assistant 半回合开始
        Step 5: 补回面包屑 → 为图片/CLI/MCP 附件补回文字占位，让模型知道它们存在过

        【参数说明】
        - max_messages: int → 最多回放几条消息（0 表示用默认 120）
        - max_tokens: int → token 预算上限（0 表示不限制）
        - include_timestamps: bool → 是否给 user 消息打时间戳前缀

        【返回值】
        - list[dict]: 裁剪后的干净消息列表，每条包含 role / content / tool_calls 等字段
        """
        unconsolidated = self.messages[self.last_consolidated:]
        max_messages = max_messages if max_messages > 0 else 120
        sliced = unconsolidated[-max_messages:]

        # 尽量避免从“半个 turn 中间”开始回放；
        # 但如果上一条是主动推送给用户的 assistant 消息，用户可能正在回复它，
        # 这时允许把那条 assistant 一起保留。
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        # 如果切片前端恰好落在孤儿 tool result 上，要把它去掉，
        # 否则模型会看到一条“没有前置 tool_call 声明”的工具结果。
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            if message.get("_command"):
                continue
            content = message.get("content", "")
            role = message.get("role")
            if role == "assistant" and isinstance(content, str):
                content = _sanitize_assistant_replay_text(content)
            # 历史回放时，原始图片块通常不会再完整塞回去，
            # 但至少要补一条文字 breadcrumb，让模型知道“这里曾有一张图”。
            media = message.get("media")
            if role == "user" and isinstance(media, list) and media and isinstance(content, str):
                breadcrumbs = "\n".join(
                    image_placeholder_text(p) for p in media if isinstance(p, str) and p
                )
                content = f"{content}\n{breadcrumbs}" if content else breadcrumbs
            cli_apps = message.get("cli_apps")
            if role == "user" and isinstance(cli_apps, list) and cli_apps and isinstance(content, str):
                cli_lines: list[str] = []
                for item in cli_apps[:8]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    if not name:
                        continue
                    entry = str(item.get("entry_point") or "unknown").strip() or "unknown"
                    cli_lines.append(
                        f"[CLI App Attachment: @{name}; tool=run_cli_app; entry_point={entry}; "
                        f"skill=skills/cli-app-{name}/SKILL.md]"
                    )
                if cli_lines:
                    breadcrumbs = "\n".join(cli_lines)
                    content = f"{content}\n{breadcrumbs}" if content else breadcrumbs
            mcp_presets = message.get("mcp_presets")
            if (
                role == "user"
                and isinstance(mcp_presets, list)
                and mcp_presets
                and isinstance(content, str)
            ):
                mcp_lines: list[str] = []
                for item in mcp_presets[:8]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    if not name:
                        continue
                    transport = str(item.get("transport") or "mcp").strip() or "mcp"
                    mcp_lines.append(
                        f"[MCP Preset Attachment: @{name}; tool_prefix=mcp_{name}_; "
                        f"transport={transport}]"
                    )
                if mcp_lines:
                    breadcrumbs = "\n".join(mcp_lines)
                    content = f"{content}\n{breadcrumbs}" if content else breadcrumbs
            if include_timestamps:
                content = self._annotate_message_time(message, content)
            if role == "assistant" and isinstance(content, str) and not content.strip():
                if not any(key in message for key in ("tool_calls", "reasoning_content", "thinking_blocks")):
                    continue
            entry: dict[str, Any] = {"role": message["role"], "content": content}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)

        if max_tokens > 0 and out:
            kept: list[dict[str, Any]] = []
            used = 0
            for message in reversed(out):
                tokens = estimate_message_tokens(message)
                if kept and used + tokens > max_tokens:
                    break
                kept.append(message)
                used += tokens
            kept.reverse()

            # 尽量让裁剪后的历史从第一个可见 user turn 开始。
            first_user = next((i for i, m in enumerate(kept) if m.get("role") == "user"), None)
            if first_user is not None:
                kept = kept[first_user:]
            else:
                # token 预算太紧时，可能会只剩 assistant 尾巴。
                # 这里宁可略微超预算，也尽量把最近的 user turn 找回来。
                recovered_user = next(
                    (i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"),
                    None,
                )
                if recovered_user is not None:
                    kept = out[recovered_user:]

            # 同时还要保证前端边界在 tool-call 语义上合法。
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
            out = kept
        return out

    def clear(self) -> None:
        """清空会话历史并重置关键状态。"""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()
        self.metadata.pop("_last_summary", None)

    def retain_recent_legal_suffix(self, max_messages: int) -> tuple[list[dict], int]:
        """保留最近的一段“合法后缀”，并严格受 ``max_messages`` 上限约束。

        这里的“合法”主要指：
        - 尽量从 user turn 开始
        - 不能从孤儿 tool result 开头
        - 最终保留条数不能超过上限
        """
        if max_messages <= 0:
            dropped = list(self.messages)
            lc = self.last_consolidated
            self.clear()
            return dropped, min(lc, len(dropped))
        if len(self.messages) <= max_messages:
            return [], 0

        original = list(self.messages)
        before_lc = self.last_consolidated

        retained = list(self.messages[-max_messages:])

        # 优先让保留片段从 user turn 开始，这样更符合对话回放习惯。
        first_user = next((i for i, m in enumerate(retained) if m.get("role") == "user"), None)
        if first_user is not None:
            retained = retained[first_user:]
        else:
            # 如果尾部全是 assistant/tool，就回头锚定到全会话最近一个 user turn。
            latest_user = next(
                (i for i in range(len(self.messages) - 1, -1, -1)
                 if self.messages[i].get("role") == "user"),
                None,
            )
            if latest_user is not None:
                retained = list(self.messages[latest_user: latest_user + max_messages])

        # 和 get_history 保持一致：前端不能是孤儿 tool result。
        start = find_legal_message_start(retained)
        if start:
            retained = retained[start:]

        # 最终硬保证：绝不超过 max_messages。
        if len(retained) > max_messages:
            retained = retained[-max_messages:]
            start = find_legal_message_start(retained)
            if start:
                retained = retained[start:]

        # 用对象 identity 而不是值比较来计算 dropped，
        # 避免非连续切片场景下误删或重复消息。
        retained_ids = set(id(m) for m in retained)
        dropped = [m for m in original if id(m) not in retained_ids]

        # 统计 dropped 中有多少本来就在“已压缩前缀”里。
        # 这不是简单 min() 能表达的，因为 dropped 可能横跨压缩前后边界。
        already_consolidated = sum(
            1 for i, m in enumerate(original)
            if i < before_lc and id(m) not in retained_ids
        )

        # 重新计算新的 last_consolidated 边界。
        new_lc = sum(
            1 for i, m in enumerate(original)
            if i < before_lc and id(m) in retained_ids
        )

        self.messages = retained
        self.last_consolidated = new_lc
        self.updated_at = datetime.now()
        return dropped, already_consolidated

    def enforce_file_cap(
        self,
        on_archive: Any = None,
        limit: int = FILE_MAX_MESSAGES,
    ) -> None:
        """限制 session 文件无限增长：必要时归档并裁剪旧前缀。"""
        if limit <= 0 or len(self.messages) <= limit:
            return

        dropped, already_consolidated = self.retain_recent_legal_suffix(limit)
        if not dropped:
            return

        archive_chunk = dropped[already_consolidated:]
        if archive_chunk and on_archive:
            on_archive(archive_chunk)
        logger.info(
            "Session file cap hit for {}: dropped {}, raw-archived {}, kept {}",
            self.key,
            len(dropped),
            len(archive_chunk),
            len(self.messages),
        )


class SessionManager:
    """会话管理器。

    【核心职责】
    1. 把 ``session_key`` 映射到内存中的 ``Session`` 对象
    2. 负责从磁盘 JSONL 文件加载/修复/保存会话
    3. 为 WebUI / API 提供列出、删除、分叉会话等能力
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    @staticmethod
    def safe_key(key: str) -> str:
        """把任意 session key 映射成稳定且安全的文件名。"""
        return safe_filename(key.replace(":", "_"))

    def _get_session_path(self, key: str) -> Path:
        """返回某个 session 的 JSONL 文件路径。"""
        return self.sessions_dir / f"{self.safe_key(key)}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """返回旧版全局 session 路径，用于迁移兜底。"""
        return self.legacy_sessions_dir / f"{self.safe_key(key)}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """获取或创建一个会话。

        【中文名称】获取或创建会话

        【功能说明】
        这是 SessionManager 最核心的入口方法。AgentLoop 每次处理入站消息时，
        都会先调用它来拿到当前对话的 Session 容器。

        【读取顺序】
        1. 先看内存缓存（self._cache）—— 最快，避免重复磁盘 I/O
        2. 缓存没有就从磁盘 JSONL 文件加载 —— 恢复持久化状态
        3. 还没有就创建全新 Session —— 首次对话或文件丢失时

        【参数说明】
        - key: str → 会话唯一键，通常是 "channel:chat_id"（如 "telegram:123456"）

        【返回值】
        - Session: 包含消息历史、元数据、压缩边界的完整会话对象
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """从磁盘加载一个会话；必要时尝试从旧目录迁移。"""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            # JSONL 第一行通常是 metadata，后面每行是一条 message。
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            repaired = self._repair(key)
            if repaired is not None:
                logger.info("Recovered session {} from corrupt file ({} messages)", key, len(repaired.messages))
            return repaired

    def _repair(self, key: str) -> Session | None:
        """尝试从损坏的 JSONL 会话文件中尽量恢复可读内容。"""
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            skipped = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            with suppress(ValueError, TypeError):
                                created_at = datetime.fromisoformat(data["created_at"])
                        if data.get("updated_at"):
                            with suppress(ValueError, TypeError):
                                updated_at = datetime.fromisoformat(data["updated_at"])
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            if skipped:
                logger.warning("Skipped {} corrupt lines in session {}", skipped, key)

            if not messages and not metadata:
                return None

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Repair failed for session {}: {}", key, e)
            return None

    @staticmethod
    def _session_payload(session: Session) -> dict[str, Any]:
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "messages": session.messages,
        }

    def save(self, session: Session, *, fsync: bool = False) -> None:
        """以原子方式把 Session 保存到磁盘。

        【中文名称】原子保存会话

        【功能说明】
        这是 nanobot 会话持久化可靠性的核心实现。它不是简单地覆盖写文件，
        而是通过"写临时文件 → fsync → 原子 rename"的三步策略来保证：
        即使进程在中途崩溃，也不会留下半截损坏的 JSONL 文件。

        【原子写流程】
        1. 先把整个 Session（metadata + 所有消息）写入 .jsonl.tmp 临时文件
        2. 如果 fsync=True，则先 flush + fsync 临时文件，确保数据真正落到磁盘
        3. 调用 os.replace(tmp_path, path) 原子替换 —— 系统保证 rename 要么全成功要么全失败
        4. 如果 fsync=True，再 fsync 父目录，让 rename 的元数据更新也尽量持久化
           （注意：Windows 下目录 fsync 会触发 PermissionError，所以这里捕获并跳过）

        【JSONL 文件格式约定】
        第一行是 _type="metadata" 的特殊行，包含 key、时间戳、元数据和 last_consolidated。
        后续每一行是一条 role/content 消息的 JSON 对象。

        【参数说明】
        - session: Session → 要持久化的会话对象
        - fsync: bool → 是否强制刷盘（程序退出前使用，常规保存可以 False）

        【可靠性设计】
        如果写入过程中发生任何异常（包括 BaseException），
        finally 块会主动删除残留的临时文件，避免留下垃圾。
        """
        path = self._get_session_path(session.key)
        tmp_path = path.with_suffix(".jsonl.tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # 约定：JSONL 第一行是 metadata，后续每一行是一条消息。
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())

            os.replace(tmp_path, path)

            if fsync:
                # 进一步 fsync 父目录，让 rename 的元数据更新也尽量持久。
                # Windows 下目录 fsync 会触发 PermissionError，所以跳过。
                with suppress(PermissionError):
                    fd = os.open(str(path.parent), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        self._cache[session.key] = session

    def flush_all(self) -> int:
        """在程序优雅退出时，把所有缓存会话都做一次带 fsync 的落盘。"""
        flushed = 0
        for key, session in list(self._cache.items()):
            try:
                self.save(session, fsync=True)
                flushed += 1
            except Exception:
                logger.warning("Failed to flush session {}", key, exc_info=True)
        return flushed

    def invalidate(self, key: str) -> None:
        """从内存缓存中移除某个会话。"""
        self._cache.pop(key, None)

    def delete_session(self, key: str) -> bool:
        """删除某个会话的磁盘文件和缓存。"""
        path = self._get_session_path(key)
        self.invalidate(key)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.warning("Failed to delete session file {}: {}", path, e)
            return False

    def fork_session_before_user_index(
        self,
        source_key: str,
        target_key: str,
        before_user_index: int,
    ) -> Session | None:
        """在某个 user 消息索引之前，对现有会话做分叉复制。

        这主要给 WebUI 的“从某个历史点开分支继续聊”功能使用。
        """
        if before_user_index < 0:
            return None
        source = self._cache.get(source_key) or self._load(source_key)
        if source is None:
            return None

        copied: list[dict[str, Any]] = []
        user_index = 0
        found_target = False
        for message in source.messages:
            if message.get("role") == "user":
                if user_index == before_user_index:
                    found_target = True
                    break
                user_index += 1
            copied.append(deepcopy(message))
        if user_index == before_user_index:
            found_target = True
        if not found_target:
            return None

        metadata = deepcopy(source.metadata)
        # 分叉会话不应该继承“只对原会话当前运行态有效”的易变元数据。
        for key in _FORK_VOLATILE_METADATA_KEYS:
            metadata.pop(key, None)

        last_consolidated = min(source.last_consolidated, len(copied))
        if source.last_consolidated > len(copied):
            metadata.pop("_last_summary", None)
            last_consolidated = 0

        now = datetime.now()
        target = Session(
            key=target_key,
            messages=copied,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            last_consolidated=last_consolidated,
        )
        self.save(target, fsync=True)
        return target

    def read_session_file(self, key: str) -> dict[str, Any] | None:
        """只读方式读取 session 文件，不放进缓存。

        适合 HTTP 接口等“读取一下就走”的场景，避免无意义污染内存缓存。
        """
        path = self._get_session_path(key)
        if not path.exists():
            return None
        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: str | None = None
            updated_at: str | None = None
            stored_key: str | None = None
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = data.get("created_at")
                        updated_at = data.get("updated_at")
                        stored_key = data.get("key")
                    else:
                        messages.append(data)
            return {
                "key": stored_key or key,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata,
                "messages": messages,
            }
        except Exception as e:
            logger.warning("Failed to read session {}: {}", key, e)
            repaired = self._repair(key)
            if repaired is not None:
                logger.info("Recovered read-only session view {} from corrupt file", key)
                return self._session_payload(repaired)
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话的基础信息，用于 WebUI / API 展示列表。"""
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            fallback_key = path.stem.replace("_", ":", 1)
            try:
                # 仅读 metadata 和少量 preview，避免列会话时把大文件整份读进来。
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            metadata = data.get("metadata", {})
                            title = _metadata_title(metadata)
                            preview = ""
                            fallback_preview = ""
                            scanned_records = 0
                            scanned_chars = 0
                            for line in f:
                                if not line.strip():
                                    continue
                                scanned_records += 1
                                scanned_chars += len(line)
                                if (
                                    scanned_records > _SESSION_LIST_PREVIEW_MAX_RECORDS
                                    or scanned_chars > _SESSION_LIST_PREVIEW_MAX_CHARS
                                ):
                                    break
                                item = json.loads(line)
                                if item.get("_type") == "metadata":
                                    continue
                                text = _message_preview_text(item)
                                if not text:
                                    continue
                                if item.get("role") == "user":
                                    preview = text
                                    break
                                if not fallback_preview and item.get("role") == "assistant":
                                    fallback_preview = text
                            preview = preview or fallback_preview
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "title": title,
                                "preview": preview,
                                "path": str(path)
                            })
            except Exception:
                repaired = self._repair(fallback_key)
                if repaired is not None:
                    sessions.append({
                        "key": repaired.key,
                        "created_at": repaired.created_at.isoformat(),
                        "updated_at": repaired.updated_at.isoformat(),
                        "title": _metadata_title(repaired.metadata),
                        "preview": next(
                            (
                                text
                                for msg in repaired.messages
                                if (text := _message_preview_text(msg))
                            ),
                            "",
                        ),
                        "path": str(path)
                    })
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
