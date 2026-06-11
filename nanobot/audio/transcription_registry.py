"""语音转文字 provider 注册表。

【中文名称】转写服务注册表

这个模块是应用层关于“有哪些转写 provider 可用”的单一事实来源。
它记录的信息包括：

- provider 正式名称
- 默认模型名
- 适配器类的导入路径
- 可选别名

真正的 HTTP 适配器实现仍然在 `nanobot.providers.transcription`，
而这里负责“登记”和“查找”。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol


class TranscriptionProviderAdapter(Protocol):
    """转写适配器在运行期必须满足的协议。

    这里使用 `Protocol` 的好处是：

    - 不要求所有 provider 真的继承同一个基类；
    - 只要它们“长得像”这个接口，就能被当成合法适配器使用。
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ) -> None: ...

    async def transcribe(self, file_path: str | Path) -> str: ...


@dataclass(frozen=True)
class TranscriptionProviderSpec:
    """单个转写 provider 的静态描述信息。"""
    name: str
    default_model: str
    adapter: str
    aliases: tuple[str, ...] = ()

    def load_adapter(self) -> type[TranscriptionProviderAdapter]:
        """按 `模块路径:类名` 的形式动态加载适配器类。"""
        module_name, _, class_name = self.adapter.partition(":")
        if not module_name or not class_name:
            raise RuntimeError(f"Invalid transcription adapter path: {self.adapter}")
        adapter = getattr(import_module(module_name), class_name)
        return adapter


TRANSCRIPTION_PROVIDERS: tuple[TranscriptionProviderSpec, ...] = (
    TranscriptionProviderSpec(
        name="groq",
        default_model="whisper-large-v3",
        adapter="nanobot.providers.transcription:GroqTranscriptionProvider",
    ),
    TranscriptionProviderSpec(
        name="openai",
        default_model="whisper-1",
        adapter="nanobot.providers.transcription:OpenAITranscriptionProvider",
    ),
    TranscriptionProviderSpec(
        name="openrouter",
        default_model="openai/whisper-1",
        adapter="nanobot.providers.transcription:OpenRouterTranscriptionProvider",
    ),
    TranscriptionProviderSpec(
        name="xiaomi_mimo",
        default_model="mimo-v2.5-asr",
        adapter="nanobot.providers.transcription:XiaomiMiMoTranscriptionProvider",
        aliases=("mimo", "xiaomi"),
    ),
    TranscriptionProviderSpec(
        name="stepfun",
        default_model="stepaudio-2.5-asr",
        adapter="nanobot.providers.transcription:StepFunTranscriptionProvider",
    ),
    TranscriptionProviderSpec(
        name="assemblyai",
        default_model="universal-3-pro,universal-2",
        adapter="nanobot.providers.transcription:AssemblyAITranscriptionProvider",
    ),
)

_BY_NAME = {spec.name: spec for spec in TRANSCRIPTION_PROVIDERS}
_BY_ALIAS = {alias: spec for spec in TRANSCRIPTION_PROVIDERS for alias in spec.aliases}


def transcription_provider_names() -> tuple[str, ...]:
    """返回所有已注册 provider 的正式名称。"""
    return tuple(spec.name for spec in TRANSCRIPTION_PROVIDERS)


def get_transcription_provider(name: str) -> TranscriptionProviderSpec | None:
    """按正式名称获取 provider 描述对象。"""
    return _BY_NAME.get(name)


def resolve_transcription_provider(value: Any) -> TranscriptionProviderSpec | None:
    """按正式名称或别名解析 provider。"""
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    return _BY_NAME.get(name) or _BY_ALIAS.get(name)
