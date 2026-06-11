"""搜索额度查询工具：供 ``/status`` 命令展示 Web 搜索 provider 用量。

【中文名称】搜索额度查询

【功能说明】
当用户在聊天中输入 /status 查看 agent 运行状态时，其中一个展示项是
"Web 搜索还剩多少额度"。这个模块负责查询并格式化这部分信息。

【支持的 Provider】
- Tavily: 有官方的 /usage API，返回 plan_usage、plan_limit、search_usage 等字段
- 其他 (Brave、DuckDuckGo、SearXNG、Jina): 没有用量 API，统一返回 "not available"

【完整的查询流程（以 Tavily 为例）】

Phase 1: 确定 Provider
  - 从配置中读取 search provider 名（如 "tavily"、"brave"）
  - 只有 "tavily" 走远程 API 查询路径，其余返回 supported=False

Phase 2: 获取 API Key
  - 优先用传入的 api_key 参数
  - 其次用环境变量 TAVILY_API_KEY
  - 两者都没有 → 返回 error="TAVILY_API_KEY not configured"

Phase 3: 调用 Tavily /usage API
  - GET https://api.tavily.com/usage
  - Authorization: Bearer <api_key>
  - 超时 8 秒

Phase 4: 解析响应 → SearchUsageInfo
  - used = account.plan_usage
  - limit = account.plan_limit
  - remaining = max(0, limit - used)（推算值，Tavily 原始响应不直接给）
  - search_used, extract_used, crawl_used（Tavily 三种搜索类型的细分）

Phase 5: 格式化输出（SearchUsageInfo.format）
  - 有数据 → "Usage: 150 / 1000 requests\nBreakdown: Search: 120 | Extract: 20 | Crawl: 10"
  - 无 API → "Usage tracking: not available for this provider"
  - API 调用失败 → "Usage: unavailable (HTTP 429)"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchUsageInfo:
    """某个搜索 provider 的结构化用量信息。

    【字段说明】
    - provider: provider 名（"tavily"、"brave"、"duckduckgo" 等）
    - supported: 该 provider 是否有可查的用量 API（Tavily=True，其余=False）
    - error: API 调用失败时的错误信息（None 表示正常）
    - used: 已用量（None 表示不支持或查询失败）
    - limit: 总额度（None 表示不支持）
    - remaining: 剩余可用量（由 limit - used 推算）
    - reset_date: 额度重置日期（Tavily 不提供此字段）
    - search_used: Tavily 细分——搜索 API 已用量
    - extract_used: Tavily 细分——提取 API 已用量
    - crawl_used: Tavily 细分——爬取 API 已用量
    """

    provider: str
    supported: bool = False
    error: str | None = None

    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    reset_date: str | None = None

    # Tavily-specific breakdown
    search_used: int | None = None
    extract_used: int | None = None
    crawl_used: int | None = None

    def format(self) -> str:
        """格式化成适合 ``/status`` 展示的多行文本。

        【中文名称】格式化搜索额度为展示文本

        【输出示例（Tavily 正常）】
        🔍 Web Search: tavily
           Usage: 150 / 1000 requests
           Breakdown: Search: 120 | Extract: 20 | Crawl: 10
           Remaining: 850 requests

        【输出示例（无 API）】
        🔍 Web Search: brave
           Usage tracking: not available for this provider

        【输出示例（API 调用失败）】
        🔍 Web Search: tavily
           Usage: unavailable (HTTP 429)
        """
        lines = [f"🔍 Web Search: {self.provider}"]

        if not self.supported:
            lines.append("   Usage tracking: not available for this provider")
            return "\n".join(lines)

        if self.error:
            lines.append(f"   Usage: unavailable ({self.error})")
            return "\n".join(lines)

        if self.used is not None and self.limit is not None:
            lines.append(f"   Usage: {self.used} / {self.limit} requests")
        elif self.used is not None:
            lines.append(f"   Usage: {self.used} requests")

        # Tavily breakdown
        breakdown_parts = []
        if self.search_used is not None:
            breakdown_parts.append(f"Search: {self.search_used}")
        if self.extract_used is not None:
            breakdown_parts.append(f"Extract: {self.extract_used}")
        if self.crawl_used is not None:
            breakdown_parts.append(f"Crawl: {self.crawl_used}")
        if breakdown_parts:
            lines.append(f"   Breakdown: {' | '.join(breakdown_parts)}")

        if self.remaining is not None:
            lines.append(f"   Remaining: {self.remaining} requests")

        if self.reset_date:
            lines.append(f"   Resets: {self.reset_date}")

        return "\n".join(lines)


async def fetch_search_usage(
    provider: str,
    api_key: str | None = None,
) -> SearchUsageInfo:
    """查询当前配置搜索 provider 的额度信息。

    【中文名称】获取搜索额度

    【参数说明】
    - provider: 搜索 provider 名（如 "tavily"、"brave"、"duckduckgo"）
    - api_key: API key（可选，Tavily 会读取环境变量 TAVILY_API_KEY 作为兜底）

    【返回值】
    - SearchUsageInfo: 结构化额度信息，supported=False 表示该 provider 不支持用量查询
    """
    p = (provider or "duckduckgo").strip().lower()

    if p == "tavily":
        return await _fetch_tavily_usage(api_key)
    else:
        # brave, duckduckgo, searxng, jina, unknown — no usage API
        return SearchUsageInfo(provider=p, supported=False)


# Tavily 是这里真正支持"远程额度查询 API"的 provider。

async def _fetch_tavily_usage(api_key: str | None) -> SearchUsageInfo:
    """调用 Tavily 的 ``/usage`` 接口读取额度统计。

    【中文名称】查询 Tavily 用量

    【完整的 3 个阶段】

    Phase 1: 获取 API Key
      - 优先用传入参数，其次读环境变量 TAVILY_API_KEY
      - 两者都没有 → 返回 error 结果

    Phase 2: HTTP 调用
      - GET https://api.tavily.com/usage
      - Header: Authorization: Bearer <key>
      - 超时 8 秒，使用 httpx.AsyncClient

    Phase 3: 解析响应
      - 调用 _parse_tavily_usage 将 JSON 转回 SearchUsageInfo
      - HTTP 错误或网络异常 → 返回 error 结果
    """
    import httpx

    key = api_key or os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return SearchUsageInfo(
            provider="tavily",
            supported=True,
            error="TAVILY_API_KEY not configured",
        )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://api.tavily.com/usage",
                headers={"Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
        data: dict[str, Any] = r.json()
        return _parse_tavily_usage(data)
    except httpx.HTTPStatusError as e:
        return SearchUsageInfo(
            provider="tavily",
            supported=True,
            error=f"HTTP {e.response.status_code}",
        )
    except Exception as e:
        return SearchUsageInfo(
            provider="tavily",
            supported=True,
            error=str(e)[:80],
        )


def _parse_tavily_usage(data: dict[str, Any]) -> SearchUsageInfo:
    """解析 Tavily ``/usage`` 返回的 JSON 结构。

    【中文名称】解析 Tavily 用量响应

    【Tavily /usage API 返回结构示例】
    {
      "account": {
        "plan_usage": 150,     → used
        "plan_limit": 1000,    → limit
        "search_usage": 120,   → search_used
        "extract_usage": 20,   → extract_used
        "crawl_usage": 10      → crawl_used
      }
    }

    【注意】
    Tavily 原始响应不直接返回 remaining，这里通过 limit - used 推算。
    """
    account = data.get("account") or {}
    used = account.get("plan_usage")
    limit = account.get("plan_limit")

    # 用"总额度 - 已用额度"推算剩余额度。
    remaining = None
    if used is not None and limit is not None:
        remaining = max(0, limit - used)

    return SearchUsageInfo(
        provider="tavily",
        supported=True,
        used=used,
        limit=limit,
        remaining=remaining,
        search_used=account.get("search_usage"),
        extract_used=account.get("extract_usage"),
        crawl_used=account.get("crawl_usage"),
    )
