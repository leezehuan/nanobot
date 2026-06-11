"""Prompt 模板渲染工具：从 nanobot/templates/ 目录加载并渲染 Jinja2 模板。

【中文名称】Prompt 模板渲染器

【功能说明】
nanobot 的 system prompt、工具描述、agent 指令等很多文本不是硬编码在代码里的，
而是放在 nanobot/templates/ 目录下的 .md / .jinja 文件中。
这个模块提供了统一的加载 + 渲染入口。

【使用示例】
```python
system_prompt = render_template("agent/system.md", agent_name="nanobot", tool_descriptions=tools)
user_prompt = render_template("agent/evaluator.md", part="user", task_context="heartbeat", response="ok")
```

【设计决策】
- 使用 Jinja2（非 str.format）：模板里需要条件、循环等复杂逻辑
- FileSystemLoader：直接从 templates/ 目录加载，新增模板无需改代码
- autoescape=False：Prompt 不是 HTML，不需要自动转义
- trim_blocks + lstrip_blocks：去掉模板控制语句产生的多余空行
- lru_cache 缓存 Environment：避免每次渲染都重建 Environment
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


@lru_cache
def _environment() -> Environment:
    """获取 Jinja2 模板环境（带缓存）。

    【配置说明】
    - loader: FileSystemLoader，从 nanobot/templates/ 目录加载
    - autoescape: False（Prompt 不需要 HTML 转义）
    - trim_blocks: True（去掉 {% ... %} 块后的第一个换行）
    - lstrip_blocks: True（去掉 {% ... %} 块前的空白）
    """
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """渲染指定模板文件，并返回最终文本。

    【中文名称】渲染 Prompt 模板

    【参数说明】
    - name: 模板相对路径（相对于 nanobot/templates/），
      例如 "agent/system.md"、"agent/evaluator.md"
    - strip: 是否去掉末尾多余换行（单行提示语建议开启）
    - **kwargs: 传给 Jinja2 模板的变量（如 agent_name、tool_descriptions 等）

    【返回值】
    - str: 渲染后的最终文本

    【异常】
    - jinja2.TemplateNotFound: 模板文件不存在
    - jinja2.UndefinedError: 模板引用了未传入的变量
    """
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
