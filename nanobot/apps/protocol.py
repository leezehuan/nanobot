"""设置页管理的 Agent App 通用清单协议。

【中文名称】应用清单协议

这个模块定义了一种“中立的、描述性的 manifest 结构”，用于描述一个
 Agent App 是什么、能做什么、如何安装、如何卸载、可信度如何。

这里要特别理解两点：

1. 它只负责“描述”应用，不直接负责安装。
   真正执行安装/卸载动作的逻辑，仍然放在各自的适配器里。
2. 它的目标是给 WebUI、注册表、未来的扩展系统提供统一词汇。
   只要大家都认识同一套字段，就能在不共享实现的前提下协同工作。
"""

from __future__ import annotations

from typing import Any

APP_PROTOCOL_SCHEMA = "agent-app.v1"


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    """压缩字典，移除“空的可选字段”。

    【为什么需要它】

    manifest 里很多字段是可选的，比如 `logo_url`、`docs_url`。
    如果这些字段为空，就没必要把它们序列化出去。

    【注意】

    这里不会误删 `False`、`0` 这类“有意义的显式值”，因为它们常常代表
    一个明确状态，而不是“没填”。
    """
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def app_manifest(
    *,
    app_id: str,
    display_name: str,
    description: str,
    category: str,
    source: str,
    capabilities: list[dict[str, Any]],
    install: dict[str, Any],
    remove: dict[str, Any],
    trust: dict[str, Any],
    version: str | None = None,
    logo_url: str | None = None,
    brand_color: str | None = None,
    docs_url: str | None = None,
) -> dict[str, Any]:
    """构造标准化的应用 manifest 字典。

    【返回值是什么】

    返回的是一个适合传给 WebUI、配置页、应用注册表的普通字典。

    【为什么不直接让调用方手写字典】

    因为统一从这里构造，可以保证：

    - 字段名稳定；
    - schema 版本统一；
    - 空字段自动清理；
    - 不同来源的 app 在结构上保持一致。
    """
    return compact_dict({
        "schema": APP_PROTOCOL_SCHEMA,
        "id": app_id,
        "display_name": display_name,
        "version": version,
        "description": description,
        "category": category,
        "source": source,
        "logo_url": logo_url,
        "brand_color": brand_color,
        "docs_url": docs_url,
        "capabilities": capabilities,
        "install": install,
        "remove": remove,
        "trust": trust,
    })
