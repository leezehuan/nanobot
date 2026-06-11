"""LLM 运行时小工具：把“当前 provider + 当前 model”打包传递。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nanobot.providers.base import LLMProvider


@dataclass(frozen=True)
class LLMRuntime:
    """描述一次当前生效的 LLM 运行时组合。"""
    provider: LLMProvider
    model: str


LLMRuntimeResolver = Callable[[], LLMRuntime]


def static_llm_runtime(provider: LLMProvider, model: str) -> LLMRuntimeResolver:
    """返回一个固定不变的 runtime resolver。"""
    runtime = LLMRuntime(provider=provider, model=model)
    return lambda: runtime
