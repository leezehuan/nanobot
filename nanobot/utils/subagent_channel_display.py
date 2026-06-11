"""子 Agent 展示清洗工具：把内部 subagent 注入文本裁剪成人类可读版本。

【中文名称】子 Agent 结果清洗

【功能说明】
当主 Agent 启动 subagent 执行子任务时，subagent 完成后会通过"注入事件"
（injected_event="subagent_result"）把完整结果送回消息列表。这些结果包含了：
- 完整的任务描述（给 subagent 看的）
- RESULT: 头部 + 完整执行结果
- "summarize this naturally" 等给模型看的额外指令

这些内部文本直接展示给外部渠道（Telegram、Discord、CLI）会让用户困惑。
本模块负责裁剪掉不适合外部展示的部分，保留干净的"结果摘要"。

【清洗策略】

Step 1: 提取 Subagent 标题头
  - 如果第一行以 "[Subagent" 开头，提取这一行作为 header

Step 2: 定位 RESULT 段落
  - 查找 "\nresult:\n" 或 "\nresult:" 标记
  - 如果找不到 → 只返回 header（或原始文本）
  - 如果找到 → 取标记之后的内容

Step 3: 裁剪模型指令
  - 在 RESULT 结尾，subagent 可能附带了 "summarize this naturally" 等给模型的提示
  - 找到这个标记 → 截断之前的内容

Step 4: 长度裁剪
  - 限制为 _SUBAGENT_CHANNEL_RESULT_MAX_CHARS（800 字符）
  - 超长时截断并追加 "…"

Step 5: 组装返回
  - header + "\n\n" + body（如果两部分都有）
  - 否则只返回 header 或 body

【注意】
完整的 subagent 结果仍然保存在磁盘/内部上下文（history.jsonl）中，
这里的清洗只影响对外展示的渠道消息。
"""

from __future__ import annotations

from typing import Any

# 限制展示给渠道的 Result 长度；完整文本仍然保留在磁盘/内部上下文中。
_SUBAGENT_CHANNEL_RESULT_MAX_CHARS = 800


def scrub_subagent_announce_body(content: str) -> str:
    """把完整 subagent announce 文本裁剪成适合渠道展示的版本。

    【中文名称】清洗子 Agent 公告正文

    【完整的 5 个阶段】

    Phase 1: 提取 Subagent 标题头
      - 第一行以 "[Subagent" 开头 → 作为 header 保留
      - 原因是 "Subagent X completed task Y in Zs" 这种标题对用户有信息价值

    Phase 2: 定位 RESULT 段落
      - 搜索 "\nresult:\n" → 取标记后的内容（不区分大小写）
      - 找不到标记 → 只返回 header（如果 header 存在）
      - 找不到 header 和标记 → 返回原始 stripped 文本

    Phase 3: 裁剪 "summarize this naturally" 指令
      - 这是给 subagent 看的模型提示，不应展示给外部用户
      - 找到该标记 → 截断之前的内容

    Phase 4: 长度裁剪
      - 如果 body 超过 800 字符 → 截断并追加 "…"

    Phase 5: 拼接返回
      - 如果 header 和 body 都存在 → "header\n\nbody"
      - 只存在一方 → 返回那一方
      - 都不存在 → 返回原始 stripped 文本

    【参数说明】
    - content: subagent 注入事件的完整文本内容

    【返回值】
    - str: 清洗后的精简文本
    """
    stripped = content.replace("\r\n", "\n").strip()
    lines = stripped.splitlines()
    header = ""
    if lines and lines[0].startswith("[Subagent"):
        header = lines[0].strip()

    lower = stripped.lower()
    key = "\nresult:\n"
    ri = lower.find(key)
    if ri == -1:
        key = "\nresult:"
        ri = lower.find(key)
    if ri == -1:
        return header if header else stripped

    after = stripped[ri + len(key) :].lstrip()
    summ_marker = "summarize this naturally"
    si = after.lower().find(summ_marker)
    if si != -1:
        after = after[:si].rstrip()

    body = after.strip()
    limit = _SUBAGENT_CHANNEL_RESULT_MAX_CHARS
    if limit and len(body) > limit:
        body = body[: limit - 1].rstrip() + "…"
    if header and body:
        return f"{header}\n\n{body}"
    return header or body or stripped


def scrub_subagent_messages_for_channel(messages: list[dict[str, Any]]) -> None:
    """原地修改消息列表中携带 ``subagent_result`` 注入的消息内容。

    【中文名称】清洗消息列表中的子 Agent 注入

    【功能说明】
    遍历消息列表，找到所有 injected_event == "subagent_result" 的消息，
    将其 content 替换为清洗后的版本。

    【参数说明】
    - messages: 消息字典列表（原地修改，不创建新列表）

    【副作用】
    - 修改匹配消息的 content 字段为清洗后的文本
    - 不影响其他字段（role、tool_calls、reasoning_content 等）
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("injected_event") != "subagent_result":
            continue
        raw = msg.get("content")
        if not isinstance(raw, str) or not raw.strip():
            continue
        msg["content"] = scrub_subagent_announce_body(raw)
