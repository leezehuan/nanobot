"""Apps 域的公共导出入口。

【中文名称】应用协议辅助入口

这一层很薄，主要做两件事：

1. 把 `nanobot.apps.protocol` 里的核心协议对象重新导出；
2. 给其它模块一个稳定导入路径，避免大家都直接耦合到具体实现文件。

你可以把它理解成“应用层的门面文件（facade）”。
"""

from nanobot.apps.protocol import APP_PROTOCOL_SCHEMA, app_manifest

__all__ = ["APP_PROTOCOL_SCHEMA", "app_manifest"]
