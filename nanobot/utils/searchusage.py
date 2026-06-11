"""搜索额度查询工具：供 ``/status`` 命令展示 Web 搜索 provider 用量。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchUsageInfo:
    """某个搜索 provider 的结构化用量信息。"""

    provider: str
    supported: bool = False          # True if the provider has a usage API
    error: str | None = None         # Set when the API call failed

    # Usage counters (None = not available for this provider)
    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    reset_date: str | None = None    # ISO date string, e.g. "2026-05-01"

    # Tavily-specific breakdown
    search_used: int | None = None
    extract_used: int | None = None
    crawl_used: int | None = None

    def format(self) -> str:
        """格式化成适合 ``/status`` 展示的多行文本。"""
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
    """查询当前配置搜索 provider 的额度信息。"""
    p = (provider or "duckduckgo").strip().lower()

    if p == "tavily":
        return await _fetch_tavily_usage(api_key)
    else:
        # brave, duckduckgo, searxng, jina, unknown — no usage API
        return SearchUsageInfo(provider=p, supported=False)


# Tavily 是这里真正支持“远程额度查询 API”的 provider。

async def _fetch_tavily_usage(api_key: str | None) -> SearchUsageInfo:
    """调用 Tavily 的 ``/usage`` 接口读取额度统计。"""
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
    """解析 Tavily ``/usage`` 返回的 JSON 结构。"""
    account = data.get("account") or {}
    used = account.get("plan_usage")
    limit = account.get("plan_limit")

    # 用“总额度 - 已用额度”推算剩余额度。
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


