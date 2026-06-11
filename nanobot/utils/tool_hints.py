"""工具提示格式化：把 tool call 压缩成人类容易扫读的短提示。

【中文名称】工具调用提示格式化

【功能说明】
在进度条、聊天气泡、状态提示等需要展示"Agent 正在做什么"的场景中，
原始 tool call JSON（如 {"name": "read_file", "arguments": {"path": "D:/very/long/path/foo.py"}}）
太长且不友好。本模块负责将其压缩为人类可读的短文本（如 "read …/foo.py"）。

【压缩策略】
1. 已知工具用注册表渲染：每个工具注册了"优先参数"和"渲染模板"
2. MCP 工具按 "server::tool" 格式展示
3. 未知工具用兜底逻辑：取第一个参数值作为展示文本
4. 连续重复的相同提示会折叠成 "× N" 形式

【注册表示例】
read_file → 优先取 path/file_path 参数 → 渲染为 "read …/foo.py"
exec     → 优先取 command 参数 → 缩写命令中嵌入的路径 → 渲染为 "$ ..."
"""

from __future__ import annotations

import re

from nanobot.utils.path import abbreviate_path

# 注册表格式：
# tool_name -> (优先读取哪些参数, 渲染模板, 是否按路径缩写, 是否按命令缩写)
_TOOL_FORMATS: dict[str, tuple[list[str], str, bool, bool]] = {
    "read_file":  (["path", "file_path"],              "read {}",     True,  False),
    "write_file": (["path", "file_path"],              "write {}",    True,  False),
    "edit":       (["file_path", "path"],              "edit {}",     True,  False),
    "find_files": (["query", "glob", "path"],           "find {}",     False, False),
    "grep":       (["pattern"],                        'grep "{}"',   False, False),
    "exec":       (["command"],                        "$ {}",        False, True),
    "list_exec_sessions": ([],                          "exec sessions", False, False),
    "web_search": (["query"],                          'search "{}"', False, False),
    "web_fetch":  (["url"],                            "fetch {}",    True,  False),
    "list_dir":   (["path"],                           "ls {}",       True,  False),
}

# 匹配 shell 命令里嵌入的路径，支持带空格的单引号/双引号路径。
_PATH_IN_CMD_RE = re.compile(
    r'"(?P<double>(?:[A-Za-z]:[/\\]|~/|/)[^"]+)"'
    r"|'(?P<single>(?:[A-Za-z]:[/\\]|~/|/)[^']+)'"
    r"|(?P<bare>(?:[A-Za-z]:[/\\]|~/|(?<=\s)/)[^\s;&|<>\"']+)"
)


def format_tool_hints(tool_calls: list, max_length: int = 40) -> str:
    r"""把一组 tool call 格式化成短提示字符串。

    【中文名称】格式化工具提示

    【完整的 4 个阶段】

    Phase 1: 逐个格式化 tool call
      - 已知工具（在 _TOOL_FORMATS 注册表中）→ _fmt_known() 按模板渲染
      - MCP 工具（以 "mcp_" 开头）→ _fmt_mcp() 格式化为 "server::tool"
      - 未知工具 → _fmt_fallback() 取第一个参数值作为展示

    Phase 2: 路径/命令缩写
      - 被标记为 is_path 的工具参数 → abbreviate_path() 压缩长路径
      - 被标记为 is_command 的工具参数 → _abbreviate_command() 压缩内嵌路径

    Phase 3: 重复折叠
      - 连续相同提示合并为 "hint × N"（例如 "search \"weather\" × 3"）

    Phase 4: 拼接输出
      - 多个提示用 ", " 连接输出

    【参数说明】
    - tool_calls: ToolCall 对象列表，每个对象应包含 name、arguments 属性
    - max_length: 单个提示的最大字符数（默认 40）

    【返回值】
    - str: 逗号分隔的短提示字符串（例如 'read …/foo.py, search "weather"'）
    - "": tool_calls 为空
    """
    if not tool_calls:
        return ""

    formatted = []
    for tc in tool_calls:
        fmt = _TOOL_FORMATS.get(tc.name)
        if fmt:
            formatted.append(_fmt_known(tc, fmt, max_length))
        elif tc.name.startswith("mcp_"):
            formatted.append(_fmt_mcp(tc, max_length))
        else:
            formatted.append(_fmt_fallback(tc, max_length))

    hints = []
    for hint in formatted:
        if hints and hints[-1][0] == hint:
            hints[-1] = (hint, hints[-1][1] + 1)
        else:
            hints.append((hint, 1))

    return ", ".join(
        f"{h} \u00d7 {c}" if c > 1 else h for h, c in hints
    )


def _get_args(tc) -> dict:
    """从 ``tc.arguments`` 中安全提取参数字典。"""
    if tc.arguments is None:
        return {}
    if isinstance(tc.arguments, list):
        return tc.arguments[0] if tc.arguments else {}
    if isinstance(tc.arguments, dict):
        return tc.arguments
    return {}


def _extract_arg(tc, key_args: list[str]) -> str | None:
    """按优先级提取最适合展示的那个字符串参数。"""
    args = _get_args(tc)
    if not isinstance(args, dict):
        return None
    for key in key_args:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    for val in args.values():
        if isinstance(val, str) and val:
            return val
    return None


def _fmt_known(tc, fmt: tuple, max_length: int = 40) -> str:
    """按注册表模板格式化一个已知工具。"""
    if not fmt[0] and "{}" not in fmt[1]:
        return fmt[1]
    val = _extract_arg(tc, fmt[0])
    if val is None:
        return tc.name
    if fmt[2]:  # is_path
        val = abbreviate_path(val, max_len=max_length)
    elif fmt[3]:  # is_command
        val = _abbreviate_command(val, max_len=max_length)
    return fmt[1].format(val)


def _abbreviate_command(cmd: str, max_len: int = 40) -> str:
    """先缩写命令中的路径，再在整体长度上做截断。"""
    path_max = max(max_len // 2, 25)

    def _replace_path(match: re.Match[str]) -> str:
        if match.group("double") is not None:
            return f'"{abbreviate_path(match.group("double"), max_len=path_max)}"'
        if match.group("single") is not None:
            return f"'{abbreviate_path(match.group('single'), max_len=path_max)}'"
        return abbreviate_path(match.group("bare"), max_len=path_max)

    abbreviated = _PATH_IN_CMD_RE.sub(_replace_path, cmd)
    if len(abbreviated) <= max_len:
        return abbreviated
    return abbreviated[:max_len - 1] + "\u2026"


def _fmt_mcp(tc, max_length: int = 40) -> str:
    """把 MCP 工具格式化成 ``server::tool`` 风格。"""
    name = tc.name
    if "__" in name:
        parts = name.split("__", 1)
        server = parts[0].removeprefix("mcp_")
        tool = parts[1]
    else:
        rest = name.removeprefix("mcp_")
        parts = rest.split("_", 1)
        server = parts[0] if parts else rest
        tool = parts[1] if len(parts) > 1 else ""
    if not tool:
        return name
    args = _get_args(tc)
    val = next((v for v in args.values() if isinstance(v, str) and v), None)
    if val is None:
        return f"{server}::{tool}"
    return f'{server}::{tool}("{abbreviate_path(val, max_length)}")'


def _fmt_fallback(tc, max_length: int = 40) -> str:
    """未知工具的兜底格式化逻辑。"""
    args = _get_args(tc)
    val = next(iter(args.values()), None) if isinstance(args, dict) else None
    if not isinstance(val, str):
        return tc.name
    return f'{tc.name}("{abbreviate_path(val, max_length)}")' if len(val) > max_length else f'{tc.name}("{val}")'
