"""WebUI 图片生成意图辅助工具。"""

from __future__ import annotations

from typing import Any

IMAGE_GENERATION_METADATA_KEY = "image_generation"


def image_generation_prompt(content: str, metadata: dict[str, Any] | None) -> str:
    """当 WebUI 开启图片生成模式时，为用户提示词追加工具使用指令。

    它不会覆盖原始用户输入，只是在末尾补一段“请优先调用 generate_image”的说明。
    """
    raw = (metadata or {}).get(IMAGE_GENERATION_METADATA_KEY)
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        return content

    aspect_ratio = raw.get("aspect_ratio")
    if isinstance(aspect_ratio, str) and aspect_ratio.strip():
        instruction = (
            "The user selected WebUI image generation mode. Use the generate_image tool. "
            f"When calling generate_image, pass aspect_ratio={aspect_ratio!r}."
        )
    else:
        instruction = (
            "The user selected WebUI image generation mode. Use the generate_image tool. "
            "Choose the most suitable aspect_ratio yourself from the prompt and intended use."
        )
    return f"{content}\n\n[WebUI image generation instruction: {instruction}]"
