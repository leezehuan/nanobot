"""GitHub Copilot Provider：通过 GitHub OAuth 换取 Copilot 访问令牌。

和常见“直接填 API Key”型 provider 不同，这个文件走的是 GitHub 设备码登录流程：

1. 用户先登录 GitHub
2. 本地保存 GitHub OAuth token
3. 运行时再用 GitHub token 换取 Copilot 短期访问 token
4. 最终仍然复用 OpenAI 兼容 provider 的请求逻辑
"""

from __future__ import annotations

import time
import webbrowser
from collections.abc import Awaitable, Callable
from contextlib import suppress

import httpx
from oauth_cli_kit.models import OAuthToken
from oauth_cli_kit.storage import FileTokenStorage

from nanobot.providers.openai_compat_provider import OpenAICompatProvider

DEFAULT_GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
DEFAULT_GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_GITHUB_USER_URL = "https://api.github.com/user"
DEFAULT_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_COPILOT_BASE_URL = "https://api.githubcopilot.com"
GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_COPILOT_SCOPE = "read:user"
TOKEN_FILENAME = "github-copilot.json"
TOKEN_APP_NAME = "nanobot"
USER_AGENT = "nanobot/0.1"
EDITOR_VERSION = "vscode/1.99.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.26.0"
_EXPIRY_SKEW_SECONDS = 60
_LONG_LIVED_TOKEN_SECONDS = 315360000


def get_storage() -> FileTokenStorage:
    """返回保存 GitHub OAuth token 的本地存储对象。"""
    return FileTokenStorage(
        token_filename=TOKEN_FILENAME,
        app_name=TOKEN_APP_NAME,
        import_codex_cli=False,
    )


def _copilot_headers(token: str) -> dict[str, str]:
    """构造访问 GitHub/Copilot 相关接口时的标准请求头。"""
    return {
        "Authorization": f"token {token}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
    }


def _load_github_token() -> OAuthToken | None:
    """从本地存储读取 GitHub OAuth token。"""
    token = get_storage().load()
    if not token or not token.access:
        return None
    return token


def get_github_copilot_login_status() -> OAuthToken | None:
    """查看本地是否已经保存了 GitHub 登录令牌。"""
    return _load_github_token()


def login_github_copilot(
    print_fn: Callable[[str], None] | None = None,
    prompt_fn: Callable[[str], str] | None = None,
) -> OAuthToken:
    """执行 GitHub Device Flow，并持久化 Copilot 后续要用的 GitHub token。

    这是一个典型的“设备码登录”流程：
    - 当前进程向 GitHub 申请 ``device_code`` / ``user_code``
    - 用户在浏览器里完成授权
    - 当前进程轮询 GitHub，直到 access token 就绪
    """
    del prompt_fn
    printer = print_fn or print
    timeout = httpx.Timeout(20.0, connect=20.0)

    with httpx.Client(timeout=timeout, follow_redirects=True, trust_env=True) as client:
        response = client.post(
            DEFAULT_GITHUB_DEVICE_CODE_URL,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            data={"client_id": GITHUB_COPILOT_CLIENT_ID, "scope": GITHUB_COPILOT_SCOPE},
        )
        response.raise_for_status()
        payload = response.json()

        device_code = str(payload["device_code"])
        user_code = str(payload["user_code"])
        verify_url = str(payload.get("verification_uri") or payload.get("verification_uri_complete") or "")
        verify_complete = str(payload.get("verification_uri_complete") or verify_url)
        interval = max(1, int(payload.get("interval") or 5))
        expires_in = int(payload.get("expires_in") or 900)

        printer(f"Open: {verify_url}")
        printer(f"Code: {user_code}")
        if verify_complete:
            # 能自动打开浏览器最好；失败也不影响用户手动复制链接授权。
            with suppress(Exception):
                webbrowser.open(verify_complete)

        deadline = time.time() + expires_in
        current_interval = interval
        access_token = None
        token_expires_in = _LONG_LIVED_TOKEN_SECONDS
        while time.time() < deadline:
            poll = client.post(
                DEFAULT_GITHUB_ACCESS_TOKEN_URL,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                data={
                    "client_id": GITHUB_COPILOT_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            poll.raise_for_status()
            poll_payload = poll.json()

            access_token = poll_payload.get("access_token")
            if access_token:
                token_expires_in = int(poll_payload.get("expires_in") or _LONG_LIVED_TOKEN_SECONDS)
                break

            error = poll_payload.get("error")
            # 轮询状态机：根据 GitHub 返回的 error 类型决定继续等待、放慢轮询或直接失败。
            if error == "authorization_pending":
                time.sleep(current_interval)
                continue
            if error == "slow_down":
                current_interval += 5
                time.sleep(current_interval)
                continue
            if error == "expired_token":
                raise RuntimeError("GitHub device code expired. Please run login again.")
            if error == "access_denied":
                raise RuntimeError("GitHub device flow was denied.")
            if error:
                desc = poll_payload.get("error_description") or error
                raise RuntimeError(str(desc))
            time.sleep(current_interval)
        else:
            raise RuntimeError("GitHub device flow timed out.")

        user = client.get(
            DEFAULT_GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": USER_AGENT,
            },
        )
        user.raise_for_status()
        user_payload = user.json()
        account_id = user_payload.get("login") or str(user_payload.get("id") or "") or None

    expires_ms = int((time.time() + token_expires_in) * 1000)
    token = OAuthToken(
        access=str(access_token),
        refresh="",
        expires=expires_ms,
        account_id=str(account_id) if account_id else None,
    )
    get_storage().save(token)
    return token


class GitHubCopilotProvider(OpenAICompatProvider):
    """先用 GitHub OAuth token 换 Copilot token，再复用 OpenAICompatProvider 的 Provider。

    继承 ``OpenAICompatProvider`` 说明：
    在 nanobot 看来，Copilot 的“消息格式、流式行为、工具调用协议”整体仍然近似
    OpenAI 兼容接口，只是鉴权环节更复杂。
    """

    def __init__(self, default_model: str = "github-copilot/gpt-4.1"):
        from nanobot.providers.registry import find_by_name

        self._copilot_access_token: str | None = None
        self._copilot_expires_at: float = 0.0
        super().__init__(
            api_key="no-key",
            api_base=DEFAULT_COPILOT_BASE_URL,
            default_model=default_model,
            extra_headers={
                "Editor-Version": EDITOR_VERSION,
                "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
                "User-Agent": USER_AGENT,
            },
            spec=find_by_name("github_copilot"),
        )

    async def _get_copilot_access_token(self) -> str:
        """获取可用的 Copilot 访问 token，并做本地缓存。"""
        now = time.time()
        if self._copilot_access_token and now < self._copilot_expires_at - _EXPIRY_SKEW_SECONDS:
            return self._copilot_access_token

        # 第一步：确认用户已经登录 GitHub。
        github_token = _load_github_token()
        if not github_token or not github_token.access:
            raise RuntimeError("GitHub Copilot is not logged in. Run: nanobot provider login github-copilot")

        timeout = httpx.Timeout(20.0, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=True) as client:
            response = await client.get(
                DEFAULT_COPILOT_TOKEN_URL,
                headers=_copilot_headers(github_token.access),
            )
            response.raise_for_status()
            payload = response.json()

        # 第二步：拿 GitHub token 去 Copilot 内部接口换真正可调用模型的 token。
        token = payload.get("token")
        if not token:
            raise RuntimeError("GitHub Copilot token exchange returned no token.")

        expires_at = payload.get("expires_at")
        if isinstance(expires_at, (int, float)):
            self._copilot_expires_at = float(expires_at)
        else:
            refresh_in = payload.get("refresh_in") or 1500
            self._copilot_expires_at = time.time() + int(refresh_in)
        self._copilot_access_token = str(token)
        return self._copilot_access_token

    async def _refresh_client_api_key(self) -> str:
        """刷新底层 OpenAI 兼容 client 所使用的 api_key。"""
        token = await self._get_copilot_access_token()
        client = await self._ensure_client()
        self.api_key = token
        client.api_key = token
        return token

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ):
        """非流式对话：先刷新 Copilot token，再复用父类实现。"""
        await self._refresh_client_api_key()
        return await super().chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
        on_content_delta: Callable[[str], None] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ):
        """流式对话：先刷新 Copilot token，再复用父类实现。"""
        await self._refresh_client_api_key()
        return await super().chat_stream(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
        )
