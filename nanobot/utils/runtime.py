"""运行时辅助工具：存放 Agent 执行阶段常用的小型策略函数和提示常量。

这个模块里的东西看起来零散，但其实都服务于“让一轮 Agent 执行更稳定”：
- 工具结果为空时如何兜底
- 输出打满时如何提示模型继续
- 工具预算耗尽后如何强制模型给最终答复
- 重复外部查询或越权访问时如何限流/拦截
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import stringify_text_blocks

_MAX_REPEAT_EXTERNAL_LOOKUPS = 2

# 同一个 turn 里，第 3 次尝试访问同一类工作区外路径时，升级为“明确停止重试”。
_MAX_REPEAT_WORKSPACE_VIOLATIONS = 2

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "I completed the tool steps but couldn't produce a final answer. "
    "Please try again or narrow the task."
)

FINALIZATION_RETRY_PROMPT = (
    "Please provide your response to the user based on the conversation above."
)

BUDGET_EXHAUSTED_FINALIZATION_PROMPT = (
    "The tool-call budget for this turn is exhausted. Based only on the "
    "conversation and tool results above, provide a concise final response to "
    "the user. Do not call or request tools. Do not claim the task is complete "
    "unless the evidence above clearly shows it is complete. State what was "
    "done, what remains, and the best next step if anything is incomplete."
)

LENGTH_RECOVERY_PROMPT = (
    "Output limit reached. Continue exactly where you left off "
    "— no recap, no apology. Break remaining work into smaller steps if needed."
)

SUSTAINED_GOAL_CONTINUE_PROMPT = (
    "You have an active sustained goal. Please continue working toward the "
    "objective using your tools, or call complete_goal if the work is truly finished."
)


def empty_tool_result_message(tool_name: str) -> str:
    """为“工具成功执行但没有可见输出”的情况生成一个短占位文本。

    这样模型在后续整理上下文时，不会把“空字符串”误以为是工具没有运行。
    """
    return f"({tool_name} completed with no output)"


def ensure_nonempty_tool_result(tool_name: str, content: Any) -> Any:
    """把“语义上为空”的工具结果替换成占位文本。

    这里处理的不只是 ``None``，还包括：
    - 空字符串
    - 空列表
    - 由文本块组成、但拼起来仍然全空白的列表
    """
    if content is None:
        return empty_tool_result_message(tool_name)
    if isinstance(content, str) and not content.strip():
        return empty_tool_result_message(tool_name)
    if isinstance(content, list):
        if not content:
            return empty_tool_result_message(tool_name)
        text_payload = stringify_text_blocks(content)
        if text_payload is not None and not text_payload.strip():
            return empty_tool_result_message(tool_name)
    return content


def is_blank_text(content: str | None) -> bool:
    """判断一段文本是否为空或全是空白字符。"""
    return content is None or not content.strip()


def build_finalization_retry_message() -> dict[str, str]:
    """构造“不要再调用工具，只补最终答案”的恢复提示。"""
    return {"role": "user", "content": FINALIZATION_RETRY_PROMPT}


def build_budget_exhausted_finalization_message() -> dict[str, str]:
    """当工具预算耗尽时，构造强制收尾提示。

    它的重点是告诉模型：
    - 现在不能再调工具
    - 只能基于现有证据收尾
    - 不要虚构“已经完成”
    """
    return {"role": "user", "content": BUDGET_EXHAUSTED_FINALIZATION_PROMPT}


def build_length_recovery_message() -> dict[str, str]:
    """当输出长度打满时，提示模型从中断处继续。"""
    return {"role": "user", "content": LENGTH_RECOVERY_PROMPT}


def build_goal_continue_message(custom: str | None = None) -> dict[str, str]:
    """当 sustained goal 仍处于激活状态时，提示模型继续推进。"""
    return {"role": "user", "content": custom or SUSTAINED_GOAL_CONTINUE_PROMPT}


def external_lookup_signature(tool_name: str, arguments: Any) -> str | None:
    """为外部查询生成稳定签名，用于识别“重复查同一个目标”。

    例如：
    - ``web_fetch`` 的同一个 URL
    - ``web_search`` 的同一个 query
    """
    if not isinstance(arguments, dict):
        return None
    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "").strip()
        if url:
            return f"web_fetch:{url.lower()}"
    if tool_name == "web_search":
        query = str(arguments.get("query") or arguments.get("search_term") or "").strip()
        if query:
            return f"web_search:{query.lower()}"
    return None


def repeated_external_lookup_error(
    tool_name: str,
    arguments: Any,
    seen_counts: dict[str, int],
) -> str | None:
    """在短时间内重复查同一个外部目标时返回拦截错误。

    目的不是完全禁止联网，而是防止模型在失败后机械地重复同一次搜索/抓取。
    """
    signature = external_lookup_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_EXTERNAL_LOOKUPS:
        return None
    logger.warning(
        "Blocking repeated external lookup {} on attempt {}",
        signature[:160],
        count,
    )
    return (
        "Error: repeated external lookup blocked. "
        "Use the results you already have to answer, or try a meaningfully different source."
    )


# 工作区越界访问先按“软错误”处理，但会对同一目标做节流与升级。

_OUTSIDE_PATH_PATTERN = re.compile(r"(?:^|[\s|>'\"])((?:/[^\s\"'>;|<]+)|(?:~[^\s\"'>;|<]+))")


def workspace_violation_signature(
    tool_name: str,
    arguments: Any,
) -> str | None:
    """为“工作区外访问目标”生成跨工具通用签名。

    这样无论模型尝试用：
    - 文件工具
    - shell / exec
    - source / destination 参数
    只要本质上指向同一个越界路径，都会落到同一计数器上。
    """
    if not isinstance(arguments, dict):
        return None
    for key in ("path", "file_path", "target", "source", "destination"):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_violation_target(val.strip())

    if tool_name in {"exec", "shell"}:
        cmd = str(arguments.get("command") or "").strip()
        if cmd:
            match = _OUTSIDE_PATH_PATTERN.search(cmd)
            if match:
                return _normalize_violation_target(match.group(1))
        cwd = str(arguments.get("working_dir") or "").strip()
        if cwd:
            return _normalize_violation_target(cwd)

    return None


def _normalize_violation_target(raw: str) -> str:
    """规范化路径写法，让等价路径命中同一个签名键。"""
    try:
        normalized = Path(raw).expanduser().resolve().as_posix()
    except Exception:
        normalized = raw.replace("\\", "/")
    return f"violation:{normalized}".lower()


def repeated_workspace_violation_error(
    tool_name: str,
    arguments: Any,
    seen_counts: dict[str, int],
) -> str | None:
    """重复尝试越过工作区边界时，返回更强硬的错误提示。

    这里的目标是让模型停止“换个参数再试一次”的无效尝试。
    """
    signature = workspace_violation_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_WORKSPACE_VIOLATIONS:
        return None
    logger.warning(
        "Escalating repeated workspace bypass attempt {} (attempt {})",
        signature[:160],
        count,
    )
    target = signature.split("violation:", 1)[1] if "violation:" in signature else signature
    return (
        "Error: refusing repeated workspace-bypass attempts.\n"
        f"You have tried to access '{target}' (or an equivalent path) "
        f"{count} times in this turn. This is a hard policy boundary -- "
        "switching tools, shell tricks, working_dir overrides, symlinks, "
        "or base64 piping will NOT change the answer. Stop retrying. "
        "If the user genuinely needs this resource, tell them you cannot "
        "access it and ask how they want to proceed (e.g. copy the file "
        "into the workspace, or disable restrict_to_workspace for this run)."
    )
