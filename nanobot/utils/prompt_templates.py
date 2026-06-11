"""Prompt 模板工具：加载并渲染 ``nanobot/templates/`` 下的 Jinja2 模板。"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


@lru_cache
def _environment() -> Environment:
    # Prompt 模板本质上是纯文本，不需要像 HTML 模板那样自动转义。
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """渲染指定模板文件，并返回最终文本。

    ``strip=True`` 常用于单行提示语，避免保留模板文件末尾换行。
    """
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
