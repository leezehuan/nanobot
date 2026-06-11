"""私聊发送者配对授权模块。

【中文名称】配对授权入口

在一些私聊场景里，nanobot 不希望对所有陌生人直接响应。
这里提供一套“生成配对码 -> 主人审核 -> 通过后放行”的轻量授权机制。

这个 `__init__` 文件负责重新导出常用函数，并定义两类元数据键：

- `PAIRING_CODE_META_KEY`：标记某条消息关联的配对码
- `PAIRING_COMMAND_META_KEY`：标记某条消息是配对管理指令
"""

from nanobot.pairing.store import (
    approve_code,
    deny_code,
    format_expiry,
    format_pairing_reply,
    generate_code,
    get_approved,
    handle_pairing_command,
    is_approved,
    list_pending,
    revoke,
)

# 这些元数据键会被频道层和命令层使用，用来识别“配对相关消息”。
PAIRING_CODE_META_KEY = "_pairing_code"
PAIRING_COMMAND_META_KEY = "_pairing_command"

__all__ = [
    "approve_code",
    "deny_code",
    "format_expiry",
    "format_pairing_reply",
    "generate_code",
    "get_approved",
    "handle_pairing_command",
    "is_approved",
    "list_pending",
    "revoke",
    "PAIRING_CODE_META_KEY",
    "PAIRING_COMMAND_META_KEY",
]
