"""记忆系统：文件型记忆存储层 + 轻量级历史压缩器。

这个模块主要负责两件事：

1. 把 Agent 的长期记忆、归档历史、安全人格文件等内容落到磁盘。
2. 当会话历史太长时，把旧消息压缩成摘要，避免模型上下文被撑爆。

可以把它理解成：

- ``MemoryStore``：偏底层，负责“存”和“取”
- ``Consolidator``：偏策略，负责“压”和“裁”
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import weakref
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import tiktoken
from loguru import logger

from nanobot.session.manager import Session
from nanobot.utils.gitstore import GitStore
from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    strip_think,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


# ---------------------------------------------------------------------------
# MemoryStore：纯文件 I/O 层
# ---------------------------------------------------------------------------

class MemoryStore:
    """管理记忆相关文件的底层读写。

    这里不直接决定“模型怎么使用这些记忆”，而是专注于文件层面的职责：

    - ``MEMORY.md``：长期事实记忆
    - ``history.jsonl``：历史摘要/归档日志
    - ``SOUL.md``：Agent 的自我设定或长期风格
    - ``USER.md``：用户长期偏好/背景信息
    """

    _DEFAULT_MAX_HISTORY = 1000
    _INTERNAL_HISTORY_SESSION_PREFIXES = ("cron:", "dream:")
    _INTERNAL_HISTORY_SESSION_KEYS = {"heartbeat"}
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._corruption_logged = False  # rate-limit non-int cursor warning
        self._oversize_logged = False  # rate-limit oversized-entry warning
        self._append_lock = threading.Lock()  # serialize cursor allocation + append
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor",
        ])
        self._maybe_migrate_legacy_history()

    @property
    def git(self) -> GitStore:
        return self._git

    # -- 通用辅助方法 --------------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """把旧版 ``HISTORY.md`` 一次性迁移到新版 ``history.jsonl``。

        这里采用“尽量保内容、不过度苛求格式完美”的策略：

        - 能拆成多条就尽量拆
        - 拆不精确也尽量保留原始文本
        - 迁移完成后备份旧文件，避免重复迁移
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # 默认把 Dream 游标也推进到最后，避免升级后第一次启动时
                # 把整份旧历史重新灌给 Dream 流程重复处理。
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md：长期事实记忆 --------------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md：Agent 自我设定 --------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md：用户长期信息 ----------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- 上下文注入（供 context.py 调用） -----------------------------------

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- history.jsonl：追加写日志，JSONL 格式 -------------------------------

    def append_history(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        """向 ``history.jsonl`` 追加一条记录，并返回自增游标 ``cursor``。

        关键行为：

        - 写入前先通过 ``strip_think`` 清掉思维链残留、``<channel|>`` 等内部痕迹
        - 如果清洗后内容为空，也不会回退去写原始泄漏文本
        - 每条记录都有单调递增的 ``cursor``，后续 Dream / 压缩流程靠它判断
          “哪些历史已经处理过”

        ``max_chars`` 是最后一道兜底保险，防止调用方忘记限长后，
        把超大文本直接写进历史文件。正常情况下，各上层调用方仍应先自行控长。
        """
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = entry.rstrip()
        if len(raw) > limit:
            if not self._oversize_logged:
                self._oversize_logged = True
                logger.warning(
                    "history entry exceeds {} chars ({}); truncating. "
                    "Usually means a caller forgot its own cap; "
                    "further occurrences suppressed.",
                    limit, len(raw),
                )
            raw = truncate_text(raw, limit)
        content = strip_think(raw)
        # “分配游标”和“真正写入”必须是一个原子过程，
        # 否则并发写入时可能都读到同一个旧 cursor，产生重复编号。
        with self._append_lock:
            cursor = self._next_cursor()
            if raw and not content:
                logger.debug(
                    "history entry {} stripped to empty (likely template leak); "
                    "persisting empty content to avoid re-polluting context",
                    cursor,
                )
            record = {"cursor": cursor, "timestamp": ts, "content": content}
            if session_key:
                record["session_key"] = session_key
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """只接受真正的 int 游标，显式排除 bool。"""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        """遍历带合法整数游标的历史项；遇到损坏数据只告警一次。"""
        poisoned: Any = None
        for entry in self._read_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            if cursor is None:
                poisoned = raw
                continue
            yield entry, cursor
        if poisoned is not None and not self._corruption_logged:
            self._corruption_logged = True
            logger.warning(
                "history.jsonl contains a non-int cursor ({!r}); dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                poisoned,
            )

    def _next_cursor(self) -> int:
        """读取当前游标计数器，并计算下一条记录的 cursor。"""
        if self._cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
        # 快路径：如果文件尾记录正常，就直接拿最后一条 + 1。
        # 慢路径：如果尾部损坏，就全量扫描取最大值，尽量自愈外部错误写入。
        last = self._read_last_entry() or {}
        cursor = self._valid_cursor(last.get("cursor"))
        if cursor is not None:
            return cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """返回 ``cursor > since_cursor`` 的未处理历史。"""
        return [e for e, c in self._iter_valid_entries() if c > since_cursor]

    @classmethod
    def _is_internal_history_session(cls, session_key: str | None) -> bool:
        if not session_key:
            return False
        return (
            session_key in cls._INTERNAL_HISTORY_SESSION_KEYS
            or session_key.startswith(cls._INTERNAL_HISTORY_SESSION_PREFIXES)
        )

    def read_recent_history_for_prompt(
        self,
        since_cursor: int,
        *,
        session_key: str | None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """读取适合注入到当前回合提示词里的“未处理历史”。

        这里的关键不是“把新历史全拿出来”，而是：
        - 普通会话模式下，只拿当前 session 的增量历史
        - unified_session 模式下，允许共享非内部任务历史
        - 明确排除 cron / dream 这类内部 session 的噪声
        """
        entries = self.read_unprocessed_history(since_cursor=since_cursor)
        if session_key is None:
            return entries
        if not unified_session:
            return [e for e in entries if e.get("session_key") == session_key]

        return [
            entry
            for entry in entries
            if (entry_session := entry.get("session_key")) == session_key
            or not self._is_internal_history_session(entry_session)
        ]

    def compact_history(self) -> None:
        """如果历史条数超限，就丢弃最旧记录。"""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        kept = entries[-self.max_history_entries:]
        self._write_entries(kept)

    # -- JSONL 辅助方法 -----------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """读取 ``history.jsonl`` 中的全部记录。"""
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """高效读取 JSONL 文件的最后一条记录。"""
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """以原子写方式整文件覆盖 ``history.jsonl``。"""
        tmp_path = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.history_file)

            # 再 fsync 父目录，尽量保证 rename 的目录项更新也持久化。
            # Windows 上目录 fsync 通常不可用，所以这里允许跳过。
            with suppress(PermissionError):
                fd = os.open(str(self.history_file.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    # -- Dream 游标 ----------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    def build_dream_prompt(self, *, max_entries: int = 20) -> tuple[str, int] | None:
        """把尚未被 Dream 处理过的历史拼成一次 Dream 提示词。

        返回：

        - ``(prompt, last_cursor)``：本次提示词，以及处理到的最后游标
        - ``None``：没有新历史，不需要跑 Dream
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.get_last_dream_cursor()
        entries = self.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return None

        batch = entries[:max_entries]
        history_text = "\n".join(
            f"[{e['timestamp']}] {truncate_text(e['content'], 500)}"
            for e in batch
        )
        skill_creator_path = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        template = render_template(
            "agent/dream.md", strip=True, skill_creator_path=skill_creator_path,
        )
        prompt = f"{template}\n\n## Conversation History\n{history_text}"
        return (prompt, batch[-1]["cursor"])

    def build_dream_tools(self):
        """构建 Dream 专用的受限工具集。

        Dream 不是正常主回合，而是一个更受限的后台整理流程，
        因此这里只开放少量文件读写/补丁工具。
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.apply_patch import ApplyPatchTool
        from nanobot.agent.tools.file_state import FileStates
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
        from nanobot.agent.tools.registry import ToolRegistry

        tools = ToolRegistry()
        file_states = FileStates()
        workspace = self.workspace
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        editable_roots = [self.soul_file, self.user_file, skills_dir]

        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
            file_states=file_states,
        ))
        tools.register(EditFileTool(
            workspace=workspace,
            allowed_dir=self.memory_dir,
            extra_allowed_dirs=editable_roots,
            file_states=file_states,
        ))
        tools.register(ApplyPatchTool(
            workspace=workspace,
            allowed_dir=self.memory_dir,
            extra_allowed_dirs=editable_roots,
            file_states=file_states,
        ))
        tools.register(WriteFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            file_states=file_states,
        ))
        return tools

    @staticmethod
    def dream_run_completed(resp: object | None) -> bool:
        """判断一次 Dream 临时回合是否真正“正常完成”。"""
        metadata = getattr(resp, "metadata", None)
        return isinstance(metadata, dict) and metadata.get("_stop_reason") == "completed"

    # -- 历史格式化辅助 -----------------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> None:
        """降级兜底：当 LLM 摘要失败时，直接把原始消息归档到 history。

        这样即使压缩模型临时不可用，我们也不会丢失历史，只是把这段历史
        以较原始的 ``[RAW]`` 形式保存下来，后续仍可追溯。
        """
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        formatted = truncate_text(self._format_messages(messages), limit)
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{formatted}",
            session_key=session_key,
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )

    # ------------------------------------------------------------------
    # Dream 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def dream_session_key() -> str:
        """生成 Dream 任务专用的 session key。"""
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}"

    @staticmethod
    def build_dream_commit_message(prefix: str, resp: object | None) -> str:
        """构建 Dream 自动提交说明；如果有模型摘要就附在后面。"""
        msg = prefix
        if resp is not None and getattr(resp, "content", None):
            msg = f"{msg}\n\n{resp.content.strip()}"
        return msg

    @staticmethod
    def prune_dream_sessions(sessions_dir: Path, *, keep: int = 10) -> None:
        """删除最旧的 Dream 会话文件，只保留最近 ``keep`` 个。

        这里只会处理 ``dream_*.jsonl``，不会碰普通用户会话文件。
        """
        dream_files = sorted(
            sessions_dir.glob("dream_*.jsonl"), key=lambda p: p.stat().st_mtime,
        )
        if len(dream_files) <= keep:
            return

        to_remove = dream_files[: len(dream_files) - keep]
        for path in to_remove:
            try:
                path.unlink()
                logger.debug("Pruned old dream session: {}", path.stem)
            except OSError:
                logger.warning("Failed to prune dream session {}", path)


# ---------------------------------------------------------------------------
# Consolidator：基于 token 预算的轻量压缩器
# ---------------------------------------------------------------------------

# 各调用方原则上都应自行限制写入长度；这里的硬上限只是最后一道保险，
# 防止有新调用方忘记限长，把超大文本直接写进 history。
_RAW_ARCHIVE_MAX_CHARS = 16_000       # fallback dump (LLM failed)
_ARCHIVE_SUMMARY_MAX_CHARS = 8_000    # LLM-produced consolidation summary
_HISTORY_ENTRY_HARD_CAP = 64_000      # emergency cap in append_history


class Consolidator:
    """轻量级历史压缩器：把旧消息总结后写入 ``history.jsonl``。

    当会话太长、快要超出模型上下文窗口时，它会选择一段旧消息：

    - 交给模型压缩成摘要
    - 把摘要持久化到 history
    - 更新 session 的“已压缩边界”
    """

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        consolidation_ratio: float = 0.5,
        unified_session: bool = False,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.consolidation_ratio = consolidation_ratio
        self.unified_session = unified_session
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def set_provider(
        self,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = provider.generation.max_tokens

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """返回某个 session 的共享压缩锁，避免并发压缩同一会话。"""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """寻找一个尽量按 user turn 对齐的安全压缩边界。"""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    @staticmethod
    def _full_unconsolidated_history(
        session: Session,
        *,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """取出尚未压缩的完整历史尾部，供压缩决策使用。"""
        unconsolidated_count = len(session.messages) - session.last_consolidated
        if unconsolidated_count <= 0:
            return []
        return session.get_history(
            max_messages=unconsolidated_count,
            include_timestamps=include_timestamps,
        )

    @staticmethod
    def _replay_overflow_boundary(
        session: Session,
        replay_max_messages: int | None,
    ) -> int | None:
        if not replay_max_messages or replay_max_messages <= 0:
            return None
        tail = list(enumerate(session.messages[session.last_consolidated:], session.last_consolidated))
        if len(tail) <= replay_max_messages:
            return None

        sliced = tail[-replay_max_messages:]
        for i, (_idx, message) in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1][1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        legal_start = find_legal_message_start([message for _idx, message in sliced])
        if legal_start:
            sliced = sliced[legal_start:]
        if not sliced:
            return len(session.messages)

        first_visible_idx = sliced[0][0]
        if first_visible_idx <= session.last_consolidated:
            return None
        return first_visible_idx

    async def _consolidate_replay_overflow(
        self,
        session: Session,
        replay_max_messages: int | None,
    ) -> str | None:
        """提前归档那些反正也会被消息回放窗口裁掉的旧消息。"""
        end_idx = self._replay_overflow_boundary(session, replay_max_messages)
        if end_idx is None:
            return None
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            return None
        logger.info(
            "Replay-window consolidation for {}: chunk={} msgs, replay_max={}",
            session.key,
            len(chunk),
            replay_max_messages,
        )
        summary = await self.archive(chunk, session_key=session.key)
        session.last_consolidated = end_idx
        self.sessions.save(session)
        return summary

    def _persist_last_summary(self, session: Session, summary: str | None) -> None:
        """把最近一次压缩摘要写回 session metadata。"""
        if summary and summary != "(nothing)":
            session.metadata["_last_summary"] = {
                "text": summary,
                "last_active": session.updated_at.isoformat(),
            }
            self.sessions.save(session)

    def estimate_session_prompt_tokens(
        self,
        session: Session,
    ) -> tuple[int, str]:
        """估算当前会话尾部完整送给模型时会占多少 prompt token。"""
        history = self._full_unconsolidated_history(session, include_timestamps=True)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        # 估算时也要把已归档摘要算进去，否则会低估真实 prompt 体积。
        meta = session.metadata.get("_last_summary")
        summary = meta.get("text") if isinstance(meta, dict) else (meta if isinstance(meta, str) else None)
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            sender_id=None,
            session_summary=summary,
            session_metadata=session.metadata,
            session_key=session.key,
            unified_session=self.unified_session,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    @property
    def _input_token_budget(self) -> int:
        """压缩摘要调用可用的输入 token 预算。"""
        return self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER

    def _truncate_to_token_budget(self, text: str) -> str:
        """把待压缩文本裁到预算内，避免压缩模型自己也超窗。"""
        budget = self._input_token_budget
        if budget <= 0:
            return truncate_text(text, _RAW_ARCHIVE_MAX_CHARS)
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= budget:
                return text
            return enc.decode(tokens[:budget]) + "\n... (truncated)"
        except Exception:
            return truncate_text(text, budget * 4)

    async def archive(
        self,
        messages: list[dict],
        *,
        session_key: str | None = None,
    ) -> str | None:
        """调用 LLM 摘要一批旧消息，并把摘要写入 history.jsonl。

        【中文名称】归档压缩

        【功能说明】
        把一批旧消息交给 LLM 生成摘要文本，然后将摘要持久化到 MemoryStore 的
        history.jsonl 文件中。这样后续回合就可以通过"最近历史摘要"的方式
        间接保留对话上下文，而不用把完整原始消息都塞进 prompt。

        【降级策略】
        如果 LLM 调用失败（网络错误、模型不可用等），会自动走 raw_archive() 降级：
        不依赖模型摘要，直接把原始消息以 [RAW] 标记写入 history.jsonl。
        这样至少不会丢历史，只是这段历史以后续迁的方式保留。

        【完整处理流程】
        1. 如果 messages 为空 → 直接返回 None
        2. 用 MemoryStore._format_messages() 把消息格式化成文本
        3. _truncate_to_token_budget() 裁剪到安全 token 预算内
        4. 调用 provider.chat_with_retry() 让 LLM 生成摘要（不带工具）
        5. 如果 LLM 返回错误 → 抛异常进入降级分支
        6. 把摘要文本通过 store.append_history() 写入 history.jsonl
        7. 如果任何步骤失败 → store.raw_archive() 做降级保存

        【参数说明】
        - messages: list[dict] → 要压缩的一组消息
        - session_key: str | None → 关联的会话键（写入 history 时标记归属）

        【返回值】
        - str | None: 成功时返回 LLM 生成的摘要文本；无内容时返回 None
        """
        if not messages:
            return None
        try:
            formatted = MemoryStore._format_messages(messages)
            formatted = self._truncate_to_token_budget(formatted)
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/consolidator_archive.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM returned error: {response.content}")
            summary = response.content or "[no summary]"
            self.store.append_history(
                summary,
                max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
                session_key=session_key,
            )
            return summary
        except Exception:
            logger.warning("Consolidation LLM call failed, raw-dumping to history")
            self.store.raw_archive(messages, session_key=session_key)
            return None

    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        replay_max_messages: int | None = None,
    ) -> None:
        """循环压缩旧消息，直到 prompt 回到安全预算范围内。

        【中文名称】按 token 预算压缩会话

        【功能说明】
        这是 CloudMemory 两阶段记忆架构中的"背景整合"阶段。当会话的未压缩尾部
        prompt token 超出安全阈值（context_window_tokens - max_completion_tokens - 1024），
        就会循环执行多轮压缩，每轮：
        1. 先做 replay overflow 压缩 → 把会被消息回放窗口裁掉的消息提前归档
        2. 估算当前 prompt token 数
        3. 如果超出预算，则 pick_consolidation_boundary 找到安全的 user turn 边界
        4. 调用 archive() 让 LLM 把这段消息压缩成摘要
        5. 推进 session.last_consolidated 边界
        6. 重新估算，如果仍超标则继续下一轮（最多 5 轮）

        【预算计算】
        input_token_budget = context_window_tokens - max_completion_tokens - 1024
        consolidation_target = input_budget * consolidation_ratio（默认 0.5）

        【调用时机】
        - AgentLoop._state_build() 中每次回合开始前
        - 作为后台任务在回合结束后异步执行

        【参数说明】
        - session: Session → 要压缩的会话对象
        - replay_max_messages: int | None → 消息回放窗口上限
        """
        if self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            # 先刷新一下 session 引用，因为 AutoCompact 之类流程可能替换过对象。
            fresh = self.sessions.get_or_create(session.key)
            if fresh is not session:
                session = fresh
            if not session.messages:
                return

            budget = self._input_token_budget
            target = int(budget * self.consolidation_ratio)
            last_summary = await self._consolidate_replay_overflow(
                session,
                replay_max_messages,
            )
            try:
                estimated, source = self.estimate_session_prompt_tokens(
                    session,
                )
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                self._persist_last_summary(session, last_summary)
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                self._persist_last_summary(session, last_summary)
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                summary = await self.archive(chunk, session_key=session.key)
                # 无论摘要成功还是失败，都推进 last_consolidated：
                # 失败时 archive() 已经做了 raw_archive 兜底，
                # 下次再处理同一段只会产生重复 [RAW] 记录。
                # 无论结果如何，都要推进边界：
                # 成功时，这段历史已经变成摘要；
                # 失败时，archive() 也已经用 raw_archive 留下“面包屑”。
                # 如果下次还从同一段重新归档，只会制造重复的 [RAW] 记录。
                if summary:
                    last_summary = summary
                session.last_consolidated = end_idx
                self.sessions.save(session)
                if not summary:
                    # 如果这次调用里摘要模型已经失效，就别在同一轮里继续硬打了；
                    # 下次调用再处理新的 chunk。
                    break

                try:
                    estimated, source = self.estimate_session_prompt_tokens(
                        session,
                    )
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    break

            # 把最后摘要写回 metadata，这样下次构造运行时上下文时可以重新注入。
            self._persist_last_summary(session, last_summary)

    async def compact_idle_session(
        self,
        session_key: str,
        max_suffix: int = 8,
    ) -> str | None:
        """对闲置会话做一次更激进的裁剪，且全过程受压缩锁保护。

        这个方法主要给 AutoCompact 使用，确保对会话的裁剪/归档都走统一的安全路径。
        """
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key)
            session = self.sessions.get_or_create(session_key)

            tail = list(session.messages[session.last_consolidated:])
            if not tail:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return ""

            probe = Session(
                key=session.key,
                messages=tail.copy(),
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata={},
                last_consolidated=0,
            )
            dropped, already_consolidated = probe.retain_recent_legal_suffix(max_suffix)
            kept = probe.messages
            archive_msgs = dropped[already_consolidated:]

            if not archive_msgs and not kept:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return ""

            last_active = session.updated_at
            summary: str | None = ""
            if archive_msgs:
                summary = await self.archive(archive_msgs, session_key=session_key)

            if summary and summary != "(nothing)":
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

            session.messages = kept
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)

            if archive_msgs:
                logger.info(
                    "Idle-session compact for {}: archived={}, kept={}, summary={}",
                    session_key,
                    len(archive_msgs),
                    len(kept),
                    bool(summary),
                )

            return summary
