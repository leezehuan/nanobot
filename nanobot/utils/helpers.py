"""nanobot 通用工具函数集合。

这个文件里放的是很多模块都会复用的“小而关键”的底层函数，例如：

- 清理模型泄漏的 think 标签
- 提取 reasoning
- 估算 token
- 持久化超长工具输出
- 生成状态文本

初学者读这个文件时，可以把它当成“项目级杂项基础设施库”。
"""

import base64
import json
import re
import shutil
import time
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from loguru import logger


def strip_think(text: str) -> str:
    """Remove thinking blocks, unclosed trailing tags, and tokenizer-level
    template leaks occasionally emitted by some models (notably Gemma 4's
    Ollama renderer).

    【中文名称】清理 thinking 标签泄漏

    【功能说明】
    部分模型（如 Gemma 4 在 Ollama 上运行时）会在输出中泄漏内部的 reasoning
    标签模板。这个函数负责把所有形式的泄漏标签清理干净，确保用户看到的文本
    不包含底层推理标记。

    【处理的 6 类泄漏】
    1. 完整闭合块：` thinking... response` 和 `<thought>...</thought>`
    2. 流式中途断开的 opening tag（从未闭合）
    3. 坏掉的 opening tag：如 `<think广场...`（标签后缺少 `>`，模型直接把
       中文内容粘在后面输出）
    4. Harmony 风格 channel marker：`<channel|>` / `<|channel|>`（仅清理文首）
    5. 孤儿 closing tag：` response` / `</thought>`（仅清理文首/文尾，
       不碰正文中讨论这些标签的正当内容）
    6. 流式分片残片：如 `</thi`, `</thin`, `<|ch` 等（仅清理文尾）

    # 由于 strip_think 也会在写入历史前调用（memory.py），
    # (4)(5) 仅在边缘清理是刻意设计——如果在正文中间也清理，
    # 就会错误地修改用户/助手讨论这些标签本身的合法消息。

    【参数说明】
    - text: str → 需要清理的原始文本

    【返回值】
    - str → 清理后的纯文本

    Covers:
      1. Well-formed ` thinking... response` and `<thought>...</thought>` blocks.
      2. Streaming prefixes where the block is never closed.
      3. *Malformed* opening tags missing the `>` — e.g. `<think广场…`. The
         model sometimes emits the tag name directly followed by user-facing
         content with no delimiter; without this step the literal `<think`
         leaks into the rendered message.
      4. Harmony-style channel markers like `<channel|>` / `<|channel|>`
         **at the start of the text** — conservative to avoid eating
         explanatory prose that mentions these tokens.
      5. Orphan closing tags ` response` / `</thought>` **at the very start
         or end of the text** only, for the same reason.
      6. Trailing partial control tags split across stream chunks, such as
         `<thi`, `<thin`, or `<tho`.

    Since this is also applied before persisting to history (memory.py),
    the edge-only stripping of (4) and (5) is deliberate: stripping those
    tokens mid-text would silently rewrite any message where a user or the
    assistant discusses the tokens themselves.
    """
    # 先处理结构完整的 think/thought 块。
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"^\s*<think>[\s\S]*$", "", text)
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text)
    text = re.sub(r"^\s*<thought>[\s\S]*$", "", text)
    # 再处理“坏掉的 opening tag”，比如 `<think广场...` 这种模型泄漏。
    # 这里不能直接用 `\w`，因为 Python 默认 Unicode 模式下，
    # 中文也会被当成单词字符，反而会放过这类泄漏。
    text = re.sub(r"<think(?![A-Za-z0-9_\-:>/])", "", text)
    text = re.sub(r"<thought(?![A-Za-z0-9_\-:>/])", "", text)
    # 只清理文本边缘上的孤儿 closing tag，避免误改正文讨论内容。
    text = re.sub(r"^\s*</think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*$", "", text)
    text = re.sub(r"^\s*</thought>\s*", "", text)
    text = re.sub(r"\s*</thought>\s*$", "", text)
    # 只清理文本开头处的 channel marker 泄漏。
    text = re.sub(r"^\s*<\|?channel\|?>\s*", "", text)
    # 流式分片可能刚好切在控制标签中间，所以这里只清理已知的“尾部残片”。
    partial_control_tag = (
        r"</?(?:t|th|thi|thin|think|tho|thou|thoug|though|thought)>?"
        r"|<\|?(?:c|ch|cha|chan|chann|channe|channel)(?:\|?>?)?"
    )
    text = re.sub(rf"(?:{partial_control_tag})$", "", text)
    text = re.sub(r"^\s*<\|?$", "", text)
    return text.strip()


def extract_think(text: str) -> tuple[str | None, str]:
    """Extract thinking content from inline ``<think>`` / ``<thought>`` blocks.

    Returns ``(thinking_text, cleaned_text)``. Only closed blocks are
    extracted; unclosed streaming prefixes are stripped from the cleaned
    text but not surfaced — :func:`strip_think` handles that case.
    """
    parts: list[str] = []
    for m in re.finditer(r"<think>([\s\S]*?)</think>", text):
        parts.append(m.group(1).strip())
    for m in re.finditer(r"<thought>([\s\S]*?)</thought>", text):
        parts.append(m.group(1).strip())
    thinking = "\n\n".join(parts) if parts else None
    return thinking, strip_think(text)


class IncrementalThinkExtractor:
    """面向流式输出的增量 think 提取器。

    某些 provider 不会单独给 reasoning 通道，而是把它混在内容流里。
    这个类负责在流式过程中“只增量吐出新的 thinking 文本”，避免重复发旧内容。
    """

    __slots__ = ("_emitted",)

    def __init__(self) -> None:
        self._emitted = ""

    def reset(self) -> None:
        self._emitted = ""

    async def feed(self, buf: str, emit: Any) -> bool:
        """从当前缓冲区里提取新出现的 thinking 文本并发射出去。"""
        thinking, _ = extract_think(buf)
        if not thinking or thinking == self._emitted:
            return False
        new = thinking[len(self._emitted):].strip()
        self._emitted = thinking
        if not new:
            return False
        await emit(new)
        return True


def extract_reasoning(
    reasoning_content: str | None,
    thinking_blocks: list[dict[str, Any]] | None,
    content: str | None,
) -> tuple[str | None, str | None]:
    """统一提取一次模型响应里的 reasoning 与清洗后的正文。"""
    if reasoning_content:
        return reasoning_content, strip_think(content) if content else content
    if thinking_blocks:
        parts = [
            tb.get("thinking", "")
            for tb in thinking_blocks
            if isinstance(tb, dict) and tb.get("type") == "thinking"
        ]
        joined = "\n\n".join(p for p in parts if p)
        return (joined or None), strip_think(content) if content else content
    if content:
        return extract_think(content)
    return None, content


def detect_image_mime(data: bytes) -> str | None:
    """仅根据魔数识别图片 MIME 类型，不依赖文件扩展名。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def build_image_content_blocks(
    raw: bytes, mime: str, path: str, label: str
) -> list[dict[str, Any]]:
    """构造图片内容块，并附上一小段文本标签。"""
    b64 = base64.b64encode(raw).decode()
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_meta": {"path": path},
        },
        {"type": "text", "text": label},
    ]


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """返回当前 ISO 时间戳字符串。"""
    return datetime.now().isoformat()


def current_time_str(timezone: str | None = None) -> str:
    """返回适合显示给用户/模型看的当前时间字符串。"""
    from zoneinfo import ZoneInfo

    try:
        tz = ZoneInfo(timezone) if timezone else None
    except (KeyError, Exception):
        tz = None

    now = datetime.now(tz=tz) if tz else datetime.now().astimezone()
    offset = now.strftime("%z")
    offset_fmt = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    tz_name = timezone or (time.strftime("%Z") or "UTC")
    return f"{now.strftime('%Y-%m-%d %H:%M (%A)')} ({tz_name}, UTC{offset_fmt})"


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')
_TOOL_RESULT_PREVIEW_CHARS = 1200
_TOOL_RESULTS_DIR = ".nanobot/tool-results"
_TOOL_RESULT_RETENTION_SECS = 7 * 24 * 60 * 60
_TOOL_RESULT_MAX_BUCKETS = 32


def safe_filename(name: str) -> str:
    """把不安全文件名字符替换成下划线。"""
    return _UNSAFE_CHARS.sub("_", name).strip()


def image_placeholder_text(path: str | None, *, empty: str = "[image]") -> str:
    """构造图片占位文本。"""
    return f"[image: {path}]" if path else empty


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本，并附加稳定的截断后缀。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """找到一个“工具调用语义合法”的消息起点。"""
    declared: set[str] = set()
    start = 0
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared.add(str(tc["id"]))
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid and str(tid) not in declared:
                start = i + 1
                declared.clear()
    return start


def stringify_text_blocks(content: list[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        if block.get("type") != "text":
            return None
        text = block.get("text")
        if not isinstance(text, str):
            return None
        parts.append(text)
    return "\n".join(parts)


def _render_tool_result_reference(
    filepath: Path,
    *,
    original_size: int,
    preview: str,
    truncated_preview: bool,
) -> str:
    result = (
        f"[tool output persisted]\n"
        f"Full output saved to: {filepath}\n"
        f"Original size: {original_size} chars\n"
        f"Preview:\n{preview}"
    )
    if truncated_preview:
        result += "\n...\n(Read the saved file if you need the full output.)"
    return result


def _bucket_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _cleanup_tool_result_buckets(root: Path, current_bucket: Path) -> None:
    siblings = [path for path in root.iterdir() if path.is_dir() and path != current_bucket]
    cutoff = time.time() - _TOOL_RESULT_RETENTION_SECS
    for path in siblings:
        if _bucket_mtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
    keep = max(_TOOL_RESULT_MAX_BUCKETS - 1, 0)
    siblings = [path for path in siblings if path.exists()]
    if len(siblings) <= keep:
        return
    siblings.sort(key=_bucket_mtime, reverse=True)
    for path in siblings[keep:]:
        shutil.rmtree(path, ignore_errors=True)


def _write_text_atomic(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def maybe_persist_tool_result(
    workspace: Path | None,
    session_key: str | None,
    tool_call_id: str,
    content: Any,
    *,
    max_chars: int,
) -> Any:
    """把超长工具输出落盘，并用稳定引用文本替换原始内容。

    【中文名称】超长工具输出落盘

    【功能说明】
    当工具调用返回的结果超过 max_chars 字符时，直接把它塞进 LLM 上下文
    会浪费大量 token，还可能撑爆 context window。此函数做三件事：

    1. 判断是否需要落盘：内容长度 > max_chars 时才处理
    2. 写入文件：保存在 workspace/.nanobot/tool-results/{session_key}/{tool_call_id}.txt
       （或 .json，取决于内容是否 list 格式）
    3. 生成引用文本：返回一段短的"预览 + 文件路径"文本，
       代替原始超长内容送入 LLM 上下文

    【清理策略】
    - 每个 session 最多保留 32 个 bucket 目录
    - 超过 7 天的旧 bucket 自动删除
    - 使用原子写入（.tmp → replace），避免读到半写入文件

    【参数说明】
    - workspace: Path | None → 工作区根目录（None 时不做落盘）
    - session_key: str | None → 会话标识（用于分目录存储）
    - tool_call_id: str → 工具调用的唯一 ID
    - content: Any → 工具返回的原始内容（str 或 list 格式）
    - max_chars: int → 保留在上下文中的最大字符数

    【返回值】
    - Any → 如果未超长则返回原始 content，否则返回引用文本
    """
    if workspace is None or max_chars <= 0:
        return content

    text_payload: str | None = None
    suffix = "txt"
    if isinstance(content, str):
        text_payload = content
    elif isinstance(content, list):
        text_payload = stringify_text_blocks(content)
        if text_payload is None:
            return content
        suffix = "json"
    else:
        return content

    if len(text_payload) <= max_chars:
        return content

    root = ensure_dir(workspace / _TOOL_RESULTS_DIR)
    bucket = ensure_dir(root / safe_filename(session_key or "default"))
    try:
        _cleanup_tool_result_buckets(root, bucket)
    except Exception:
        logger.exception("Failed to clean stale tool result buckets in {}", root)
    path = bucket / f"{safe_filename(tool_call_id)}.{suffix}"
    if not path.exists():
        if suffix == "json" and isinstance(content, list):
            _write_text_atomic(path, json.dumps(content, ensure_ascii=False, indent=2))
        else:
            _write_text_atomic(path, text_payload)

    preview = text_payload[:_TOOL_RESULT_PREVIEW_CHARS]
    return _render_tool_result_reference(
        path,
        original_size=len(text_payload),
        preview=preview,
        truncated_preview=len(text_payload) > _TOOL_RESULT_PREVIEW_CHARS,
    )


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    【中文名称】消息分片

    【功能说明】
    将超长文本按指定长度切分为多个片段。优先在换行符处切分，
    其次在空格处，实在找不到才硬截断。常用于 Telegram（4000 字限制）
    和 Discord（2000 字限制）的消息发送。

    【分片策略】
    1. 如果文本长度 <= max_len，直接返回原文本
    2. 否则：取前 max_len 字符，从后往前找换行符 → 空格 → 硬截断
    3. 切剩下的部分继续循环处理

    【参数说明】
    - content: str → 待切分的文本
    - max_len: int → 每个分片的最大长度（默认 2000）

    【返回值】
    - list[str] → 分片后的消息列表
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # Try to break at newline first, then space, then hard break
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """构造一条对各类 Provider 更安全的 assistant 消息。

    【中文名称】构造 assistant 消息

    【功能说明】
    所有 Provider 写入历史时都应通过这个函数来构造 assistant 消息。
    它保证消息结构是各 Provider 都能接受的格式，特别是：
    - content 不为 None（至少设为空字符串）
    - reasoning_content 明确设为 "" 而非省略（DeepSeek 等推理模型需要）
    - thinking_blocks 保留 Anthropic 思考过程（用于回放和下一次对话）

    【参数说明】
    - content: str | None → 回复文本
    - tool_calls: list | None → 工具调用列表
    - reasoning_content: str | None → 推理内容
    - thinking_blocks: list | None → Anthropic 思考块

    【返回值】
    - dict → 标准格式的 assistant 消息
    """
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None or thinking_blocks:
        msg["reasoning_content"] = reasoning_content if reasoning_content is not None else ""
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """使用 tiktoken 粗略估算一组消息会占多少 prompt token。

    【中文名称】Prompt Token 估算

    【功能说明】
    在 context compaction（上下文压缩）决策中使用。遍历所有消息，
    收集所有文本内容（content / reasoning_content / tool_calls / name 等），
    拼接后用 cl100k_base 编码器计算 token 数量。

    【注意】
    这是粗略估算，不包含各 Provider 的消息格式开销（如 role 标记、特殊 token 等）。
    真正的精确计数需要走 Provider 自己的 tokenizer。此处仅用于预判"是否需要压缩"。

    【参数说明】
    - messages: 消息历史列表
    - tools: 工具定义列表（可选，也会计入 token 估算）

    【返回值】
    - int → 估算的 token 数量（失败时返回 0）
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                        if txt:
                            parts.append(txt)

            tc = msg.get("tool_calls")
            if tc:
                parts.append(json.dumps(tc, ensure_ascii=False))

            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc:
                parts.append(rc)

            for key in ("name", "tool_call_id"):
                value = msg.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)

        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))

        per_message_overhead = len(messages) * 4
        return len(enc.encode("\n".join(parts))) + per_message_overhead
    except Exception:
        return 0


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """估算单条持久化消息大约会贡献多少 prompt token。"""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))

    rc = message.get("reasoning_content")
    if isinstance(rc, str) and rc:
        parts.append(rc)

    payload = "\n".join(parts)
    if not payload:
        return 4
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(4, len(enc.encode(payload)) + 4)
    except Exception:
        return max(4, len(payload) // 4 + 4)


def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """优先走 Provider 自带计数器，否则回退到 tiktoken 估算。

    【中文名称】Token 估算（链式优先级）

    【功能说明】
    上下文压缩时优先使用 Provider 自身的精确计数方法，
    只有 Provider 不支持时才回退到通用的 tiktoken 估算。
    这样可以获得更准确的 token 计数（如 Anthropic 的 prompt caching 补偿）。

    【参数说明】
    - provider: Provider 对象（检查是否有 estimate_prompt_tokens 方法）
    - model: 当前使用的模型名
    - messages: 消息历史
    - tools: 工具定义

    【返回值】
    - (token_count, source) 元组 → source 为 "provider_counter" 或 "tiktoken" 或 "none"
    """
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        with suppress(Exception):
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return int(estimated), "tiktoken"
    return 0, "none"


def build_status_content(
    *,
    version: str,
    model: str,
    start_time: float,
    last_usage: dict[str, int],
    context_window_tokens: int,
    session_msg_count: int,
    context_tokens_estimate: int,
    search_usage_text: str | None = None,
    active_task_count: int = 0,
    max_completion_tokens: int = 8192,
) -> str:
    """构造一份适合展示给用户的运行时状态快照。"""
    uptime_s = int(time.time() - start_time)
    uptime = (
        f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
        if uptime_s >= 3600
        else f"{uptime_s // 60}m {uptime_s % 60}s"
    )
    last_in = last_usage.get("prompt_tokens", 0)
    last_out = last_usage.get("completion_tokens", 0)
    cached = last_usage.get("cached_tokens", 0)
    ctx_total = max(context_window_tokens, 0)
    # 这里的预算公式与 Consolidator 保持一致。
    ctx_budget = max(ctx_total - int(max_completion_tokens) - 1024, 1)
    ctx_pct = min(int((context_tokens_estimate / ctx_budget) * 100), 999) if ctx_budget > 0 else 0
    ctx_used_str = (
        f"{context_tokens_estimate // 1000}k"
        if context_tokens_estimate >= 1000
        else str(context_tokens_estimate)
    )
    ctx_total_str = f"{ctx_total // 1000}k" if ctx_total > 0 else "n/a"
    token_line = f"\U0001f4ca Tokens: {last_in} in / {last_out} out"
    if cached and last_in:
        token_line += f" ({cached * 100 // last_in}% cached)"
    lines = [
        f"\U0001f408 nanobot v{version}",
        f"\U0001f9e0 Model: {model}",
        token_line,
        f"\U0001f4da Context: {ctx_used_str}/{ctx_total_str} ({ctx_pct}% of input budget)",
        f"\U0001f4ac Session: {session_msg_count} messages",
        f"\u23f1 Uptime: {uptime}",
        f"\u26a1 Tasks: {active_task_count} active",
    ]
    if search_usage_text:
        lines.append(search_usage_text)
    return "\n".join(lines)


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """把内置模板同步到 workspace，只补缺失文件，不覆盖用户已有文件。"""
    from importlib.resources import files as pkg_files

    try:
        tpl = pkg_files("nanobot") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        content = src.read_text(encoding="utf-8") if src else ""
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    for item in tpl.iterdir():
        if item.name.endswith(".md") and not item.name.startswith("."):
            _write(item, workspace / item.name)
    _write(tpl / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    _write(None, workspace / "memory" / "history.jsonl")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console

        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")

    # 顺手初始化 memory 相关文件的 git 版本跟踪。
    try:
        from nanobot.utils.gitstore import GitStore

        gs = GitStore(
            workspace,
            tracked_files=[
                "SOUL.md",
                "USER.md",
                "memory/MEMORY.md",
            ],
        )
        gs.init()
    except Exception:
        logger.exception("Failed to initialize git store for {}", workspace)

    return added


def load_bundled_template(template_name: str) -> str | None:
    """从 nanobot 包内读取一个内置模板文件。"""
    from importlib.resources import files as pkg_files

    with suppress(Exception):
        tpl = pkg_files("nanobot") / "templates" / template_name
        if tpl.is_file():
            return tpl.read_text(encoding="utf-8")
    return None
