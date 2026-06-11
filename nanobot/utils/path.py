"""路径缩写工具：把很长的路径或 URL 压缩成更适合界面展示的短文本。

注意，这个模块只服务“显示体验”，不参与真实文件访问。
也就是说：
- 原始路径仍然保持完整
- 这里只负责生成日志、提示文本、进度面板里的短版本
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


def abbreviate_path(path: str, max_len: int = 40) -> str:
    """缩写文件路径或 URL，同时尽量保留最有辨识度的信息。

    这里不采用“从左截断前 40 个字符”的简单策略，
    因为用户通常更关心的是：
    - 文件名是什么
    - 它靠近哪个子目录

    所以这里会优先保留 basename（最后一级文件名），再视预算补回父目录。
    """
    if not path:
        return path

    # URL 的结构和本地路径不同，单独走一套缩写逻辑更合理。
    if re.match(r"https?://", path):
        return _abbreviate_url(path, max_len)

    # 统一路径分隔符，方便后续按“段”处理，也让 Windows / Linux 展示更一致。
    normalized = path.replace("\\", "/")

    # 把 home 目录替换成 ~，减少无意义冗长前缀。
    home = os.path.expanduser("~").replace("\\", "/")
    if normalized.startswith(home + "/"):
        normalized = "~" + normalized[len(home):]
    elif normalized == home:
        normalized = "~"

    # 规范化之后如果已经足够短，就直接返回。
    if len(normalized) <= max_len:
        return normalized

    # 按路径段拆开，方便从右向左决定要保留哪些目录。
    parts = normalized.rstrip("/").split("/")
    if len(parts) <= 1:
        return normalized[:max_len - 1] + "\u2026"

    # basename 往往是用户最关心的信息，所以优先保留。
    basename = parts[-1]
    # 预算 = 总长度上限 - 前缀“.../” - 最后一个分隔符 - basename 本身长度。
    budget = max_len - len(basename) - 3  # -3 for "…/" + final "/"

    # 从右往左（从父目录往上）尽量多保留几个目录名。
    kept: list[str] = []
    for seg in reversed(parts[:-1]):
        needed = len(seg) + 1  # segment + "/"
        if not kept and needed <= budget:
            kept.append(seg)
            budget -= needed
        elif kept:
            needed_with_sep = len(seg) + 1
            if needed_with_sep <= budget:
                kept.append(seg)
                budget -= needed_with_sep
            else:
                break
        else:
            break

    kept.reverse()
    if kept:
        return "\u2026/" + "/".join(kept) + "/" + basename
    return "\u2026/" + basename


def _abbreviate_url(url: str, max_len: int = 40) -> str:
    """缩写 URL，并尽量保留域名和最后一级资源名。"""
    if len(url) <= max_len:
        return url

    parsed = urlparse(url)
    domain = parsed.netloc
    path_part = parsed.path

    # 最后一个 path 片段通常最像“资源名”或“文件名”。
    segments = path_part.rstrip("/").split("/")
    basename = segments[-1] if segments else ""

    if not basename:
        # URL 没有明显资源名时，只能退化成普通截断。
        return url[: max_len - 1] + "\u2026"

    budget = max_len - len(domain) - len(basename) - 4  # "…/" + "/"
    if budget < 0:
        # 如果域名和资源名本身就很长，至少尽量同时保住它们。
        trunc = max_len - len(domain) - 5  # "…/" + "/"
        return domain + "/\u2026/" + (basename[:trunc] if trunc > 0 else "")

    # 在剩余预算允许时，再补一小部分中间路径。
    kept: list[str] = []
    for seg in reversed(segments[:-1]):
        if len(seg) + 1 <= budget:
            kept.append(seg)
            budget -= len(seg) + 1
        else:
            break

    kept.reverse()
    if kept:
        return domain + "/\u2026/" + "/".join(kept) + "/" + basename
    return domain + "/\u2026/" + basename
