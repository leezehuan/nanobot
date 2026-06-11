"""LLM 运行时小工具：把"当前 provider + 当前 model"打包传递。

【中文名称】LLM 运行时组合

【功能说明】
nanobot 内部很多地方需要同时知道"用哪个 provider"和"用哪个 model"，
但这两个值在运行时会随配置/重启/热更新动态变化。
这个模块定义了一个不可变的数据类和一个"延迟求值"的 resolver 模式。

【LLMRuntime 数据类】
- 描述一次当前生效的 LLM 运行时组合
- frozen=True：不可变，避免运行时意外改值
- 两个字段：provider (LLMProvider) + model (str)

【LLMRuntimeResolver 回调模式】
- 类型: Callable[[], LLMRuntime]
- 目的: 延迟求值。因为启动时还不知道最终用哪个 model，
  所以不直接传 LLMRuntime 对象，而是传一个"需要时再调用"的工厂函数。
- static_llm_runtime() 用于启动时就确定 model 的场景（最常见）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nanobot.providers.base import LLMProvider


@dataclass(frozen=True)
class LLMRuntime:
    """描述一次当前生效的 LLM 运行时组合。

    【字段说明】
    - provider: LLMProvider 实例（如 OpenAIProvider、AnthropicProvider）
    - model: 模型名（如 "gpt-4o"、"claude-sonnet-4-20250514"）
    """

    provider: LLMProvider
    model: str


LLMRuntimeResolver = Callable[[], LLMRuntime]


def static_llm_runtime(provider: LLMProvider, model: str) -> LLMRuntimeResolver:
    """返回一个固定不变的 runtime resolver。

    【中文名称】创建静态运行时解析器

    【功能说明】
    适用于启动时已确定 provider + model 的场景。
    返回的 lambda 每次调用都返回同一个 LLMRuntime 对象。

    【参数说明】
    - provider: LLMProvider 实例
    - model: 模型名

    【返回值】
    - LLMRuntimeResolver: 一个零参数的 lambda 函数，调用即返回 LLMRuntime

    【使用示例】
    resolver = static_llm_runtime(openai_provider, "gpt-4o")
    runtime = resolver()  # → LLMRuntime(provider=openai_provider, model="gpt-4o")
    """
    runtime = LLMRuntime(provider=provider, model=model)
    return lambda: runtime
