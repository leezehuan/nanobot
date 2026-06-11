"""WebUI 图片生成意图辅助工具。

【中文名称】图片生成意图辅助

【功能说明】
WebUI 前端提供了一个"图片生成模式"开关。当用户开启这个模式时，
前端会在消息的 metadata 中注入 image_generation 标记。
本模块负责检测这个标记，并在用户提示词末尾追加工具使用指令，
告诉 LLM "请优先使用 generate_image 工具"。

【完整的处理流程】

Phase 1: 检测 metadata
  - 从 metadata 字典中读取 image_generation 字段
  - 如果该字段不存在 / 不是 dict / enabled != True → 不追加指令，原样返回

Phase 2: 构造指令文本
  - 如果用户指定了 aspect_ratio → 追加带特定宽高比的指令
    例如 "When calling generate_image, pass aspect_ratio='16:9'."
  - 如果用户没指定 → 追加通用指令
    "Choose the most suitable aspect_ratio yourself from the prompt and intended use."

Phase 3: 拼接返回
  - 原始用户输入 + "\n\n[WebUI image generation instruction: ...]"
  - 不修改原始用户输入，只做追加

【设计意图】
不是让前端直接发送 "generate a cat image"，而是在用户原始输入后追加
工具选择提示。这样既保留了用户原始意图的完整语义，又给了 LLM 明确
的工具选择方向（用 generate_image 而不是写代码或搜索图片）。
"""

from __future__ import annotations

from typing import Any

IMAGE_GENERATION_METADATA_KEY = "image_generation"


def image_generation_prompt(content: str, metadata: dict[str, Any] | None) -> str:
    """当 WebUI 开启图片生成模式时，为用户提示词追加工具使用指令。

    【中文名称】构造图片生成增强提示

    【参数说明】
    - content: 用户原始输入文本
    - metadata: 前端附带的消息元数据，其中可能包含 image_generation 字段

    【返回值】
    - str: 如果开启了图片生成模式 → 原始内容 + 工具指令后缀
    - str: 如果未开启 → 原始内容原样返回

    【不修改原始输入的原因】
    追加而非覆盖，保证 LLM 仍能看到用户最初想表达的含义。
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
