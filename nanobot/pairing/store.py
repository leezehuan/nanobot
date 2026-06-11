"""私聊发送者配对授权的存储层。

【中文名称】配对授权存储

数据会持久化到 `~/.nanobot/pairing.json`，里面主要保存两类内容：

1. 已授权用户（approved）
2. 待审核配对码（pending）

这个设计很轻量，适合“个人助手 / 小规模使用”场景：

- 用 JSON 文件即可；
- 用线程锁保护并发；
- 不引入数据库。
"""

from __future__ import annotations

import json
import secrets
import string
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_data_dir
from nanobot.utils.helpers import _write_text_atomic

# 这里使用线程锁，是为了让这套存储函数既能被同步 CLI 调用，
# 也能被异步频道处理器安全复用。由于配对文件很小、操作很短，
# 这种粗粒度锁在当前规模下是可以接受的。
_LOCK = threading.Lock()
_ALPHABET = string.ascii_uppercase + string.digits
_CODE_LENGTH = 8  # e.g. ABCD-EFGH
_TTL_DEFAULT_S = 600  # 10 minutes


def _store_path() -> Path:
    """返回配对授权 JSON 文件的固定存储路径。"""
    return get_data_dir() / "pairing.json"


def _load() -> dict[str, Any]:
    """从磁盘加载配对授权数据。

    返回的 `approved` 会被转成 `set`，这样后续查找某个 sender 是否已授权时
    可以做到 O(1)。
    """
    path = _store_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"approved": {}, "pending": {}}
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupted pairing store, resetting")
        return {"approved": {}, "pending": {}}

    # 把磁盘上的 list 转成 set，便于快速判断“某个用户是否已授权”。
    for channel, users in data.get("approved", {}).items():
        data["approved"][channel] = set(users)
    return data


def _save(data: dict[str, Any]) -> None:
    """把内存中的配对数据安全写回磁盘。"""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON 不能直接序列化 set，所以落盘前要转回 list。
    payload = {
        "approved": {ch: sorted(list(users)) for ch, users in data.get("approved", {}).items()},
        "pending": dict(data.get("pending", {})),
    }
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _gc_pending(data: dict[str, Any]) -> None:
    """原地清理已经过期的待审核配对码。"""
    now = time.time()
    pending: dict[str, Any] = data.get("pending", {})
    expired = [code for code, info in pending.items() if info.get("expires_at", 0) < now]
    for code in expired:
        del pending[code]


def generate_code(
    channel: str,
    sender_id: str,
    ttl: int = _TTL_DEFAULT_S,
) -> str:
    """为某个频道里的发送者生成新的配对码。

    返回值形如 `ABCD-EFGH`，方便用户手动抄写或复制给主人审核。
    """
    with _LOCK:
        data = _load()
        _gc_pending(data)
        raw = "".join(secrets.choice(_ALPHABET) for _ in range(_CODE_LENGTH))
        code = f"{raw[:4]}-{raw[4:]}"

        data.setdefault("pending", {})[code] = {
            "channel": channel,
            "sender_id": sender_id,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }
        _save(data)
        logger.info("Generated pairing code {} for {}@{}", code, sender_id, channel)
        return code


def approve_code(code: str) -> tuple[str, str] | None:
    """批准一个待审核配对码。

    成功时返回 `(channel, sender_id)`，这样调用方就知道究竟放行了谁。
    如果配对码不存在或已经过期，则返回 `None`。
    """
    with _LOCK:
        data = _load()
        _gc_pending(data)
        pending: dict[str, Any] = data.get("pending", {})
        info = pending.pop(code, None)
        if info is None:
            return None
        channel = info["channel"]
        sender_id = info["sender_id"]
        data.setdefault("approved", {}).setdefault(channel, set()).add(sender_id)
        _save(data)
        logger.info("Approved pairing code {} for {}@{}", code, sender_id, channel)
        return channel, sender_id


def deny_code(code: str) -> bool:
    """拒绝一个待审核配对码，并把它从待审核列表中删除。"""
    with _LOCK:
        data = _load()
        _gc_pending(data)
        pending: dict[str, Any] = data.get("pending", {})
        if code in pending:
            del pending[code]
            _save(data)
            logger.info("Denied pairing code {}", code)
            return True
        return False


def is_approved(channel: str, sender_id: str) -> bool:
    """检查某个发送者在指定频道里是否已经通过授权。"""
    with _LOCK:
        data = _load()
        approved: dict[str, set[str]] = data.get("approved", {})
        return str(sender_id) in approved.get(channel, set())


def list_pending() -> list[dict[str, Any]]:
    """返回所有仍未过期的待审核配对请求。"""
    with _LOCK:
        data = _load()
        _gc_pending(data)
        return [
            {"code": code, **info}
            for code, info in data.get("pending", {}).items()
        ]


def revoke(channel: str, sender_id: str) -> bool:
    """撤销某个已授权发送者在指定频道里的访问权限。"""
    with _LOCK:
        data = _load()
        approved: dict[str, set[str]] = data.get("approved", {})
        users = approved.get(channel, set())
        if sender_id in users:
            users.discard(sender_id)
            if not users:
                del approved[channel]
            _save(data)
            logger.info("Revoked {} from {}", sender_id, channel)
            return True
        return False


def get_approved(channel: str) -> list[str]:
    """列出某个频道中已授权的所有发送者 ID。"""
    with _LOCK:
        data = _load()
        return sorted(data.get("approved", {}).get(channel, set()))


def format_pairing_reply(code: str) -> str:
    """生成发给“陌生私聊用户”的配对提示消息。"""
    return (
        "Hi there! This assistant only responds to approved users.\n\n"
        f"Your pairing code is: `{code}`\n\n"
        "To get access, ask the owner to approve this code:\n"
        f"- In this chat: send `/pairing approve {code}`"
    )


def format_expiry(expires_at: float) -> str:
    """把过期时间格式化成用户可读文本。"""
    remaining = int(expires_at - time.time())
    return f"{remaining}s" if remaining > 0 else "expired"


def handle_pairing_command(channel: str, subcommand_text: str) -> str:
    """执行一条 `/pairing` 子命令，并返回给用户的回复文本。

    你可以把它看成配对系统自己的一个“小命令路由器”：

    - `list`
    - `approve <code>`
    - `deny <code>`
    - `revoke ...`

    它本身不依赖具体频道实现，因此既能给 CLI 用，也能给 Agent 的
    CommandRouter 复用。
    """
    parts = subcommand_text.split()
    sub = parts[0] if parts else "list"
    arg = parts[1] if len(parts) > 1 else None

    if sub in ("list",):
        pending = list_pending()
        if not pending:
            return "No pending pairing requests."
        lines = ["Pending pairing requests:"]
        for item in pending:
            expiry = format_expiry(item.get("expires_at", 0))
            lines.append(
                f"- `{item['code']}` | {item['channel']} | {item['sender_id']} | {expiry}"
            )
        return "\n".join(lines)

    elif sub == "approve":
        if arg is None:
            return "Usage: `/pairing approve <code>`"
        result = approve_code(arg)
        if result is None:
            return f"Invalid or expired pairing code: `{arg}`"
        ch, sid = result
        return f"Approved pairing code `{arg}` — {sid} can now access {ch}"

    elif sub == "deny":
        if arg is None:
            return "Usage: `/pairing deny <code>`"
        if deny_code(arg):
            return f"Denied pairing code `{arg}`"
        return f"Pairing code `{arg}` not found or already expired"

    elif sub == "revoke":
        if len(parts) == 2:
            return (
                f"Revoked {arg} from {channel}"
                if revoke(channel, arg)
                else f"{arg} was not in the approved list for {channel}"
            )
        if len(parts) == 3:
            return (
                f"Revoked {parts[2]} from {arg}"
                if revoke(arg, parts[2])
                else f"{parts[2]} was not in the approved list for {arg}"
            )
        return "Usage: `/pairing revoke <user_id>` or `/pairing revoke <channel> <user_id>`"

    return (
        "Unknown pairing command.\n"
        "Usage: `/pairing [list|approve <code>|deny <code>|revoke <user_id>|revoke <channel> <user_id>]`"
    )
