"""生成媒体工件（artifact）的持久化辅助函数。

【中文名称】媒体工件持久化

【功能说明】
当 generate_image 等工具产出 base64 data URL（例如 data:image/png;base64,...），
这个模块负责：
1. 解码 data URL → bytes（decode_image_data_url）
2. 按日期分桶保存到 media 目录（store_generated_image_artifact）
3. 同步写入 JSON sidecar 元数据文件（同目录 + 同 ID + .json 后缀）

【完整的落盘结构】
media/
  generated/              # save_dir（按工具/来源分类）
    2026-06-11/            # 按日期分桶（YYYY-MM-DD）
      img_a1b2c3d4e5f6.png   # 解码后的实际图片
      img_a1b2c3d4e5f6.json  # 配套元数据 sidecar
      img_7890abcdef01.webp  # 另一张图片
      img_7890abcdef01.json  # 对应的元数据 sidecar

【安全防护】
- save_dir 必须是不含 ".." 的相对路径，防止路径穿越
- 最终路径必须落在 media 根目录内（resolve + relative_to 双重验证）
- 即使 declared_mime 和 detected_mime 不一致，也会以实际检测为准
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import detect_image_mime, ensure_dir

_DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", re.DOTALL)
_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

class ArtifactError(ValueError):
    """当 artifact 解码或保存过程中的任一环节失败时抛出的异常。"""


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """解码 base64 图片 data URL，返回 (raw_bytes, detected_mime)。

    【中文名称】解码图片 Data URL

    【解码步骤】
    Step 1: 正则匹配 data URL 格式 "data:image/<mime>;base64,<payload>"
    Step 2: base64 解码 payload
    Step 3: 通过魔数检测实际图片格式（不依赖声明的 mime）
    Step 4: 如果声明的 mime 和实际检测不一致，以实际检测为准

    【为什么需要魔数检测】
    base64 前缀里声明的 image/jpeg 可能实际上是 image/png。
    如果只信任输入方声明，会导致文件后缀与内容不匹配，后续展示工具无法正确渲染。

    【参数说明】
    - data_url: 完整的 base64 图片 data URL

    【返回值】
    - (bytes, str): (解码后的原始字节, 实际检测到的 MIME 类型)

    【异常】
    - ArtifactError: 格式不匹配 / base64 解码失败 / 无法识别的图片格式
    """
    match = _DATA_IMAGE_RE.match(data_url.strip())
    if match is None:
        raise ArtifactError("expected a base64 image data URL")

    declared_mime, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ArtifactError("invalid base64 image payload") from exc

    detected_mime = detect_image_mime(raw)
    if detected_mime is None:
        raise ArtifactError("unsupported or unrecognized image data")
    if declared_mime != detected_mime:
        declared_mime = detected_mime
    return raw, declared_mime


def _safe_relative_dir(save_dir: str) -> Path:
    """校验并规范化 save_dir 为安全的相对路径。

    【校验项】
    - 不能为空
    - 不能是绝对路径
    - 路径段不能包含 "." 或 ".."（防止路径穿越）
    """
    normalized = save_dir.replace("\\", "/").strip("/")
    if not normalized:
        raise ArtifactError("save_dir must not be empty")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ArtifactError("save_dir must be a safe relative path")
    return Path(*rel.parts)


def _artifact_root(save_dir: str) -> Path:
    """解析 artifact 根目录，并通过 resolve + relative_to 保证不逃出 media 根目录。"""
    media_root = get_media_dir().resolve()
    root = (media_root / _safe_relative_dir(save_dir)).resolve()
    try:
        root.relative_to(media_root)
    except ValueError as exc:
        raise ArtifactError("artifact directory escapes media root") from exc
    return root


def store_generated_image_artifact(
    data_url: str,
    *,
    prompt: str,
    model: str,
    source_images: list[str] | None = None,
    save_dir: str = "generated",
    provider: str = "openrouter",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """把生成图片及其 sidecar 元数据持久化到 media 目录。

    【中文名称】存储生成图片工件

    【完整的 5 个阶段】

    Phase 1: 解码 data URL → (raw_bytes, mime)
      - 调用 decode_image_data_url 解析 base64 并检测真实图片格式

    Phase 2: 确定磁盘路径
      - artifact_root = media_root / save_dir（安全校验）
      - 按日期分桶: <root>/YYYY-MM-DD/
      - 唯一 ID: "img_" + uuid.hex[:12]
      - 图片文件: <date_dir>/<id>.<ext>
      - 元数据文件: <date_dir>/<id>.json

    Phase 3: 写入图片二进制
      - image_path.write_bytes(raw)

    Phase 4: 写入 sidecar 元数据
      - JSON 格式，包含 id、path、mime、prompt、model、provider、source_images、created_at

    Phase 5: 返回元数据字典
      - 调用方拿到这个字典后可以传给消息工具，将图片交付给用户

    【参数说明】
    - data_url: base64 图片 data URL
    - prompt: 生成图片时使用的提示词（用于元数据追溯）
    - model: 生成所用的模型名（用于元数据追溯）
    - source_images: 参考图的路径列表（可选）
    - save_dir: 存储子目录名（默认 "generated"）
    - provider: 图片生成 provider 名（默认 "openrouter"）
    - created_at: 创建时间（默认当前时间）

    【返回值】
    - dict: 元数据字典，包含 id、path、mime、prompt、model、provider 等字段

    【异常】
    - ArtifactError: data URL 解码失败 / save_dir 不安全 / 路径逃逸 / mime 不支持
    """
    raw, mime = decode_image_data_url(data_url)
    ext = _MIME_EXTENSIONS.get(mime)
    if ext is None:
        raise ArtifactError(f"unsupported image MIME type: {mime}")

    now = created_at or datetime.now().astimezone()
    day_dir = ensure_dir(_artifact_root(save_dir) / now.strftime("%Y-%m-%d"))
    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    image_path = day_dir / f"{artifact_id}{ext}"
    metadata_path = day_dir / f"{artifact_id}.json"

    image_path.write_bytes(raw)
    metadata: dict[str, Any] = {
        "id": artifact_id,
        "path": str(image_path),
        "mime": mime,
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "source_images": list(source_images or []),
        "created_at": now.isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def generated_image_tool_result(artifacts: list[dict[str, Any]]) -> str:
    """返回暴露给 LLM 的紧凑结构化结果。

    【中文名称】生成图片的工具结果

    【返回值结构】
    {
      "artifacts": [...],
      "next_step": "Use these artifact paths as reference_images..."
    }

    【设计意图】
    - next_step 字段是给 LLM 看的指令提示，告诉它下一步应该怎么做
    - 不直接在这里返回图片二进制，而是返回元数据引用路径
    - 真正的图片交付由"消息工具"完成
    """
    return json.dumps(
        {
            "artifacts": artifacts,
            "next_step": (
                "Use these artifact paths as reference_images for follow-up edits. "
                "Call the message tool with the artifact paths in the media parameter "
                "to deliver the images to the user. Keep raw paths internal unless the "
                "user asks for debug details."
            ),
        },
        ensure_ascii=False,
    )
