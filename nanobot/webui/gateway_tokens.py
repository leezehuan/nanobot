"""嵌入式 WebUI 网关的短期令牌存储。

【中文名称】网关令牌仓库

WebUI 网关里会用到两类短期 token：

1. 一次性或短生命周期的连接/握手 token
2. 允许访问 WebUI API 的 token

这个模块就是专门管理这些 token 的内存仓库。
它不做长期持久化，生命周期只跟随当前 gateway 进程。
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from websockets.http11 import Request as WsRequest

from nanobot.webui.http_utils import bearer_token, parse_query, query_first


@dataclass
class GatewayTokenStore:
    """管理当前网关进程里的短生命周期 token。"""

    max_tokens: int = 10_000
    issued_tokens: dict[str, float] = field(default_factory=dict)
    api_tokens: dict[str, float] = field(default_factory=dict)

    def check_api_token(self, request: WsRequest) -> bool:
        """检查请求是否携带有效的 API token。"""
        self._purge_expired_api_tokens()
        token = bearer_token(request.headers) or query_first(
            parse_query(request.path), "token"
        )
        if not token:
            return False
        expiry = self.api_tokens.get(token)
        if expiry is None or time.monotonic() > expiry:
            self.api_tokens.pop(token, None)
            return False
        return True

    def can_issue(self, *, include_api_token: bool = False) -> bool:
        """判断当前仓库是否还能继续发 token。"""
        self._purge_expired_issued_tokens()
        self._purge_expired_api_tokens()
        if len(self.issued_tokens) >= self.max_tokens:
            return False
        if include_api_token and len(self.api_tokens) >= self.max_tokens:
            return False
        return True

    def issue_token(self, ttl_s: int | float, *, api_token: bool = False) -> str:
        """签发一个新 token，并记录过期时间。"""
        token_value = f"nbwt_{secrets.token_urlsafe(32)}"
        expiry = time.monotonic() + float(ttl_s)
        self.issued_tokens[token_value] = expiry
        if api_token:
            self.api_tokens[token_value] = expiry
        return token_value

    def take_issued_token_if_valid(self, token_value: str | None) -> bool:
        """验证并消费一个已签发 token。

        这里使用“取出即作废”的语义，适合握手类一次性票据。
        """
        if not token_value:
            return False
        self._purge_expired_issued_tokens()
        expiry = self.issued_tokens.pop(token_value, None)
        if expiry is None:
            return False
        if time.monotonic() > expiry:
            return False
        return True

    def clear(self) -> None:
        """清空当前进程内所有 token 状态。"""
        self.issued_tokens.clear()
        self.api_tokens.clear()

    def _purge_expired_api_tokens(self) -> None:
        """清理已经过期的 API token。"""
        now = time.monotonic()
        for token_key, expiry in list(self.api_tokens.items()):
            if now > expiry:
                self.api_tokens.pop(token_key, None)

    def _purge_expired_issued_tokens(self) -> None:
        """清理已经过期的普通签发 token。"""
        now = time.monotonic()
        for token_key, expiry in list(self.issued_tokens.items()):
            if now > expiry:
                self.issued_tokens.pop(token_key, None)


def token_response_payload(token: str, expires_in: Any) -> dict[str, Any]:
    """构造统一的 token 返回载荷。"""
    return {"token": token, "expires_in": expires_in}
