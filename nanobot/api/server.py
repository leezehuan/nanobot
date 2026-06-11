"""OpenAI 兼容 HTTP API 服务器。

【中文名称】兼容 OpenAI 的 API 服务端

这个模块把 nanobot 包装成一个对外 HTTP 服务，核心目标是让外部程序能像
调用 OpenAI 一样调用 nanobot。

当前主要提供三个路由：

1. `/v1/chat/completions`
   最核心的对话入口，支持普通 JSON 请求和带文件的 multipart 请求。
2. `/v1/models`
   返回当前服务暴露出来的模型名。
3. `/health`
   提供健康检查。

【一个非常重要的设计点】

这里不是为每个请求都临时创建一个全新 Agent，而是把请求路由到一个持久会话。
因此它还要负责处理：

- session_key 的映射
- 同一会话的串行锁
- 流式 SSE 输出
- 文件上传解析
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import time
import uuid
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
)
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
)
from nanobot.utils.media_decode import (
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"


# ---------------------------------------------------------------------------
# 响应辅助函数
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    """按 OpenAI 风格返回错误 JSON。"""
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    """构造一个非流式 chat completion 响应体。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_text(value: Any) -> str:
    """把 `process_direct` 的返回值统一归一化成纯文本。

    nanobot 内部返回值可能是：

    - `None`
    - 拥有 `.content` 属性的对象
    - 直接就是字符串

    对 API 层来说，最终都要转换成稳定的 assistant 文本输出。
    """
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)

# ---------------------------------------------------------------------------
# SSE（流式输出）辅助函数
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """格式化一个 OpenAI 兼容的 SSE 分片。"""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# 上传/请求体解析辅助函数
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict) -> tuple[str, list[str]]:
    """解析 JSON 请求体，返回 `(文本, 媒体文件路径列表)`。

    这里兼容两种 `content` 形态：

    1. 纯字符串
    2. OpenAI 风格的多段内容数组（text / image_url）

    注意：这里不允许远程图片 URL，只接受 base64 data URL，
    这是为了避免服务端替用户访问未知外链。
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: web.Request) -> tuple[str, list[str], str | None, str | None]:
    """解析 `multipart/form-data` 请求。

    返回值依次是：

    - 文本消息
    - 落盘后的媒体文件路径列表
    - session_id
    - 请求方显式指定的 model
    """
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    media_paths: list[str] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model


# ---------------------------------------------------------------------------
# 路由处理函数
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response:
    """处理 `/v1/chat/completions` 请求。

    【完整流程】

    1. 识别请求体类型（JSON / multipart）
    2. 解析文本、媒体、session_id、model
    3. 校验请求的 model 是否等于当前服务配置的 model
    4. 根据 session_id 生成 session_key，并拿到该会话的串行锁
    5. 如果 `stream=true`，走 SSE 流式输出路径
    6. 否则走普通一次性 JSON 返回路径

    【为什么要有 session_lock】

    同一个会话如果并发进入两条请求，历史消息和工具状态可能互相打架。
    所以这里按 `session_key` 串行化，保证同一会话同一时间只跑一个请求。
    """
    content_type = request.content_type or ""
    if not isinstance(content_type, str):
        content_type = ""

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "nanobot")

    stream = False
    try:
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model = await _parse_multipart(request)
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    # === 流式返回路径 ===
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_failed = False
        emitted_content = False

        async def _on_stream(token: str) -> None:
            """把 Agent 流式吐出的 token 推到 SSE 队列。"""
            nonlocal emitted_content
            if token:
                emitted_content = True
            await queue.put(token)

        async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
            # Agent 内部的 stream_end 只表示“当前生成片段结束了”，
            # 但如果中间还会继续调工具、再生成新的片段，整个 HTTP SSE
            # 连接还不能立刻关掉。真正的关闭时机是 process_direct 返回。
            return None

        async def _run() -> None:
            """后台运行真正的 Agent 请求，并把结果持续塞进队列。"""
            nonlocal stream_failed
            try:
                async with session_lock:
                    response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                        ),
                        timeout=timeout_s,
                    )
                    # 某些情况下，底层没有逐 token 回调，但最终会直接返回完整文本。
                    # 这时我们把完整文本补进队列，避免前端拿到空流。
                    if not emitted_content:
                        response_text = _response_text(response)
                        if response_text.strip():
                            await queue.put(response_text)
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                await resp.write(_sse_chunk(token, model_name, chunk_id))
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if not stream_failed:
            await resp.write(_sse_chunk("", model_name, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # === 非流式返回路径 ===
    fallback = EMPTY_FINAL_RESPONSE_MESSAGE

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                # 如果上游返回了空文本，再重试一次；两次都空时才使用兜底文案。
                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, retrying", session_key)
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning("Empty response after retry, using fallback")
                        response_text = fallback

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(_chat_completion_response(response_text, model_name))


async def handle_models(request: web.Request) -> web.Response:
    """返回当前 API 服务暴露的模型列表。"""
    model_name = request.app.get("model_name", "nanobot")
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    """健康检查接口。常用于容器探针或反向代理存活检测。"""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app(
    agent_loop, model_name: str = "nanobot", request_timeout: float = 120.0
) -> web.Application:
    """创建 aiohttp 应用实例。

    参数说明：

    - `agent_loop`：已经初始化好的 AgentLoop，API 请求最终都会转给它处理
    - `model_name`：对外宣称的模型名字
    - `request_timeout`：单个请求最大处理时长，单位秒
    """
    app = web.Application(client_max_size=20 * 1024 * 1024)  # 允许 20MB 级别的 base64 图片请求。
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # 会话级锁表：key 是 session_key，value 是 asyncio.Lock。

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app
