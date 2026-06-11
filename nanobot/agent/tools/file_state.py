"""跟踪文件读写状态：用于编辑前提醒与重复读取去重。

这个模块主要服务两个体验目标：

1. 如果模型还没读过文件就想改，给出“先读后改”的提醒
2. 如果同一个文件内容没变，避免重复把整份内容再次返回给模型
"""

from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReadState:
    mtime: float
    offset: int
    limit: int | None
    content_hash: str | None
    can_dedup: bool


def _hash_file(p: str) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


class FileStates:
    """按 session 隔离的文件读写状态跟踪器。

    每个会话都有自己独立的状态字典，这样：

    - “文件未变化，无需重复返回” 只在当前会话内生效
    - “你还没读过这个文件就想编辑” 的提醒也只作用于当前会话

    不会把一个会话的读写痕迹串到另一个会话上。
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: dict[str, ReadState] = {}

    def record_read(self, path: str | Path, offset: int = 1, limit: int | None = None) -> None:
        """记录某个文件已被读取。"""
        p = str(Path(path).resolve())
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            return
        self._state[p] = ReadState(
            mtime=mtime,
            offset=offset,
            limit=limit,
            content_hash=_hash_file(p),
            can_dedup=True,
        )

    def record_write(self, path: str | Path) -> None:
        """记录某个文件已被写入，并刷新状态里的 mtime/hash。"""
        p = str(Path(path).resolve())
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            self._state.pop(p, None)
            return
        self._state[p] = ReadState(
            mtime=mtime,
            offset=1,
            limit=None,
            content_hash=_hash_file(p),
            can_dedup=False,
        )

    def check_read(self, path: str | Path) -> str | None:
        """检查文件是否读过，且读取结果是否仍然新鲜。

        返回值：

        - ``None``：可以安全继续
        - 警告字符串：提示应先重新读取

        这里还会尽量避免误报：
        如果 mtime 变了，但内容 hash 实际没变，就不强行提示“文件已过期”。
        """
        p = str(Path(path).resolve())
        entry = self._state.get(p)
        if entry is None:
            return "Warning: file has not been read yet. Read it first to verify content before editing."
        try:
            current_mtime = os.path.getmtime(p)
        except OSError:
            return None
        if current_mtime != entry.mtime:
            if entry.content_hash and _hash_file(p) == entry.content_hash:
                entry.mtime = current_mtime
                return None
            return "Warning: file has been modified since last read. Re-read to verify content before editing."
        # 即使 mtime 没变，也再比一次 hash，防止极短时间内的修改漏检。
        if entry.content_hash and _hash_file(p) != entry.content_hash:
            return "Warning: file has been modified since last read. Re-read to verify content before editing."
        return None

    def is_unchanged(self, path: str | Path, offset: int = 1, limit: int | None = None) -> bool:
        """判断文件是否在“同样读取参数下”保持未变。"""
        p = str(Path(path).resolve())
        entry = self._state.get(p)
        if entry is None:
            return False
        if not entry.can_dedup:
            return False
        if entry.offset != offset or entry.limit != limit:
            return False
        try:
            current_mtime = os.path.getmtime(p)
        except OSError:
            return False
        if current_mtime != entry.mtime:
            # mtime 变了，再看内容 hash 是否也真的变了。
            current_hash = _hash_file(p)
            if current_hash != entry.content_hash:
                # 内容确实变了，这次不能去重。
                entry.can_dedup = False
                return False
            # mtime 变了但内容没变（例如 touch / 编辑器重存）：
            # 本次可以认为“没变”，但下次强制完整重读一次更稳妥。
            entry.can_dedup = False
            return True
        # mtime 也没变，则可以认为内容未变。
        return True

    def get(self, path: str | Path) -> ReadState | None:
        """返回某个路径对应的原始 ``ReadState``。"""
        return self._state.get(str(Path(path).resolve()))

    def clear(self) -> None:
        """清空全部跟踪状态，常用于测试。"""
        self._state.clear()


class FileStateStore:
    """按 session key 保存 ``FileStates`` 的查找表。"""

    __slots__ = ("_states_by_key",)

    def __init__(self) -> None:
        self._states_by_key: dict[str, FileStates] = {}

    def for_session(self, session_key: str | None) -> FileStates:
        key = session_key or "__default__"
        states = self._states_by_key.get(key)
        if states is None:
            states = FileStates()
            self._states_by_key[key] = states
        return states

    def clear(self) -> None:
        self._states_by_key.clear()


_current_file_states: ContextVar[FileStates | None] = ContextVar(
    "nanobot_file_states",
    default=None,
)


def current_file_states(default: FileStates) -> FileStates:
    """返回当前异步任务绑定的 ``FileStates``，没有则回退到默认值。"""
    return _current_file_states.get() or default


def bind_file_states(file_states: FileStates) -> Token[FileStates | None]:
    """为当前异步任务绑定一份文件状态跟踪器。"""
    return _current_file_states.set(file_states)


def reset_file_states(token: Token[FileStates | None]) -> None:
    _current_file_states.reset(token)


# 模块级默认实例，主要为了兼容旧测试和直接引用模块级状态的旧调用方。
# 新代码更推荐按 session 持有自己的 ``FileStates``。
_default = FileStates()


def record_read(path: str | Path, offset: int = 1, limit: int | None = None) -> None:
    _default.record_read(path, offset=offset, limit=limit)


def record_write(path: str | Path) -> None:
    _default.record_write(path)


def check_read(path: str | Path) -> str | None:
    return _default.check_read(path)


def is_unchanged(path: str | Path, offset: int = 1, limit: int | None = None) -> bool:
    return _default.is_unchanged(path, offset=offset, limit=limit)


def clear() -> None:
    _default.clear()


# 兼容旧调用方：以前有人会直接访问模块级 ``_state`` 字典。
# 这里保留一个类似属性访问器的入口，避免旧导入立刻失效。
def __getattr__(name: str):
    if name == "_state":
        return _default._state
    raise AttributeError(name)
