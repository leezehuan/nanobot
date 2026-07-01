"""根据配置创建 LLM Provider。

这个模块负责把“配置层的 provider 信息”真正变成“可用的 Provider 实例”。
它还负责：
- 应用 model preset
- 构建 fallback provider 链
- 生成 runtime snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nanobot.config.schema import Config, InlineFallbackConfig, ModelPresetConfig
from nanobot.providers.base import LLMProvider
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.registry import find_by_name


@dataclass(frozen=True)
class ProviderSnapshot:
    """Provider 运行时快照。

    它把当前生效的 provider、model、上下文窗口和签名打包在一起，
    方便 AgentLoop 在运行时热切换模型配置。
    """
    provider: LLMProvider
    model: str
    context_window_tokens: int
    signature: tuple[object, ...]


def _resolve_model_preset(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> ModelPresetConfig:
    """解析本次要使用的模型预设。
    
    实现方法：先规范化名字或路径，再结合配置、预设和默认值得到最终可执行对象。"""
    return preset if preset is not None else config.resolve_preset(preset_name)


def _make_provider_core(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
    model: str | None = None,
) -> LLMProvider:
    """创建一个“纯 Provider”，不包 fallback 逻辑。
    
    实现方法：先解析模型预设并从配置中找 provider/spec/backend，再按 backend 分支实例化
    OpenAI-compatible、Anthropic、Azure、Bedrock、Codex 或 Copilot provider；最后把预设里的生成参数写入
    provider.generation。"""
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    model = model or resolved.model
    provider_name = config.get_provider_name(model, preset=resolved)
    p = config.get_provider(model, preset=resolved)
    spec = find_by_name(provider_name) if provider_name else None
    if spec and spec.is_transcription_only:
        raise ValueError(f"Provider '{provider_name}' only supports transcription.")
    backend = spec.backend if spec else "openai_compat"

    # 这里先做 provider 级基本校验，再进入具体实现分支。
    if backend == "azure_openai":
        if not p or not p.api_base:
            raise ValueError("Azure OpenAI requires api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key or "",
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model, preset=resolved),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    elif backend == "bedrock":
        from nanobot.providers.bedrock_provider import BedrockProvider

        provider = BedrockProvider(
            api_key=p.api_key if p else None,
            api_base=p.api_base if p else None,
            default_model=model,
            region=getattr(p, "region", None) if p else None,
            profile=getattr(p, "profile", None) if p else None,
            extra_body=p.extra_body if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model, preset=resolved),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
            extra_body=p.extra_body if p else None,
            api_type=p.api_type if p and provider_name == "openai" else "auto",
            extra_query=p.extra_query if p else None,
        )

    provider.generation = resolved.to_generation_settings()
    return provider


def _inline_fallback_preset(
    primary: ModelPresetConfig,
    fallback: InlineFallbackConfig,
) -> ModelPresetConfig:
    """把内联 fallback 配置扩展成完整的 ``ModelPresetConfig``。
    
    实现方法：以 fallback 自己的 model/provider 为主，同时把未显式填写的 max_tokens、temperature、context_window_tokens
    等字段从 primary 预设继承过来。"""
    return ModelPresetConfig(
        model=fallback.model,
        provider=fallback.provider,
        max_tokens=fallback.max_tokens if fallback.max_tokens is not None else primary.max_tokens,
        context_window_tokens=(
            fallback.context_window_tokens
            if fallback.context_window_tokens is not None
            else primary.context_window_tokens
        ),
        temperature=(
            fallback.temperature if fallback.temperature is not None else primary.temperature
        ),
        reasoning_effort=fallback.reasoning_effort,
    )


def _resolve_fallback_presets(config: Config, primary: ModelPresetConfig) -> list[ModelPresetConfig]:
    """解析主预设对应的所有 fallback 预设。
    
    实现方法：遍历 agents.defaults.fallback_models；字符串项按名字从 config.model_presets 取完整预设，内联对象则调用
    _inline_fallback_preset 补齐默认值。"""
    presets: list[ModelPresetConfig] = []
    for fallback in config.agents.defaults.fallback_models:
        if isinstance(fallback, str):
            presets.append(config.model_presets[fallback])
        else:
            presets.append(_inline_fallback_preset(primary, fallback))
    return presets


def make_provider(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
    model: str | None = None,
) -> LLMProvider:
    """创建最终对外可用的 Provider。

    【中文名称】创建 Provider 实例

    【功能说明】
    这是 Provider 层的核心工厂函数。它把配置对象中的 provider 信息
    真正转成一个可用的 LLMProvider 实例。

    【创建流程（3 步）】
    1. 解析模型预设 → 确定用哪个 model + provider 组合
    2. 根据 backend 类型实例化具体 Provider：
       - "openai_compat" → OpenAICompatProvider（大部分 provider 走这个）
       - "anthropic" → AnthropicProvider
       - "azure_openai" → AzureOpenAIProvider
       - "bedrock" → BedrockProvider
       - "openai_codex" → OpenAICodexProvider
       - "github_copilot" → GitHubCopilotProvider
    3. 如果配置了 fallback_models → 包一层 FallbackProvider 链

    【FallbackProvider 链的作用】
    当主模型调用失败时，FallbackProvider 会按 fallback_models 列表中
    的预设顺序依次尝试备用模型，提高整体可用性。

    【参数说明】
    - config: Config → 全局配置对象
    - preset_name: str | None → 模型预设名（如 "fast"、"smart"）
    - preset: ModelPresetConfig | None → 直接传入的预设（优先级高于 preset_name）
    - model: str | None → 模型名覆盖

    【返回值】
    - LLMProvider: 可直接用于 chat() / chat_stream() 调用的 Provider 实例

    实现方法：先创建不带 fallback 的主 provider，再解析 fallback 预设；如果有备用模型，就用 FallbackProvider 包装主 provider，并传入可按
    fallback 预设重新创建 provider 的工厂函数。
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    provider = _make_provider_core(config, preset_name=preset_name, preset=preset, model=model)
    fallback_presets = _resolve_fallback_presets(config, resolved)

    if fallback_presets:
        provider = FallbackProvider(
            primary=provider,
            fallback_presets=fallback_presets,
            provider_factory=lambda fb: _make_provider_core(
                config, preset_name=preset_name, preset=fb
            ),
        )

    return provider


def provider_signature(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> tuple[object, ...]:
    """返回足以标识当前 provider 链配置的签名元组。

    这个签名会被用来判断：
    “当前运行时 provider 配置是否真的变化了，需要热更新吗？”

    实现方法：把模型名、provider 名、API key/base、extra headers/body/query、区域配置、生成参数和 fallback
    签名都打进元组；只要这些输入有变化，签名就会不同。
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    p = config.get_provider(resolved.model, preset=resolved)
    fallback_presets = _resolve_fallback_presets(config, resolved)

    def _fallback_signature(fallback: ModelPresetConfig) -> tuple[object, ...]:
        """fallback signature。
        
        实现方法：读取 fallback 预设和它对应的 provider 配置，把模型/provider/API 参数/生成参数整理成可比较的不可变元组。"""
        fp = config.get_provider(fallback.model, preset=fallback)
        return (
            fallback.model,
            fallback.provider,
            config.get_provider_name(fallback.model, preset=fallback),
            config.get_api_key(fallback.model, preset=fallback),
            config.get_api_base(fallback.model, preset=fallback),
            fp.extra_headers if fp else None,
            fp.extra_body if fp else None,
            fp.api_type if fp else "auto",
            fp.extra_query if fp else None,
            getattr(fp, "region", None) if fp else None,
            getattr(fp, "profile", None) if fp else None,
            fallback.max_tokens,
            fallback.temperature,
            fallback.reasoning_effort,
            fallback.context_window_tokens,
        )

    return (
        resolved.model,
        resolved.provider,
        config.get_provider_name(resolved.model, preset=resolved),
        config.get_api_key(resolved.model, preset=resolved),
        config.get_api_base(resolved.model, preset=resolved),
        p.extra_headers if p else None,
        p.extra_body if p else None,
        p.api_type if p else "auto",
        p.extra_query if p else None,
        getattr(p, "region", None) if p else None,
        getattr(p, "profile", None) if p else None,
        resolved.max_tokens,
        resolved.temperature,
        resolved.reasoning_effort,
        resolved.context_window_tokens,
        tuple(_fallback_signature(fallback) for fallback in fallback_presets),
    )


def build_provider_snapshot(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> ProviderSnapshot:
    """构建完整 ProviderSnapshot。
    
    实现方法：从配置、上下文和运行时状态收集所需字段，再组装成后续组件可直接使用的数据结构。"""
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    fallback_windows = [
        fallback.context_window_tokens
        for fallback in _resolve_fallback_presets(config, resolved)
    ]
    return ProviderSnapshot(
        provider=make_provider(config, preset=resolved),
        model=resolved.model,
        context_window_tokens=min([resolved.context_window_tokens, *fallback_windows]),
        signature=provider_signature(config, preset=resolved),
    )


def load_provider_snapshot(
    config_path: Path | None = None,
    *,
    preset_name: str | None = None,
) -> ProviderSnapshot:
    """从磁盘配置文件加载并构建 ProviderSnapshot。
    
    实现方法：先用 load_config 读取配置文件，再解析环境变量占位符，最后交给 build_provider_snapshot 生成可热切换的运行时快照。"""
    from nanobot.config.loader import load_config, resolve_config_env_vars

    return build_provider_snapshot(
        resolve_config_env_vars(load_config(config_path)),
        preset_name=preset_name,
    )
