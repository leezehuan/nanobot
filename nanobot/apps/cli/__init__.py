"""CLI App 子域的导出入口。

【中文名称】命令行应用适配层入口

这个模块把 CLI App 相关的核心类型重新导出给外部使用：

- `CliAppError`：面向用户的错误类型
- `CliAppManager`：负责目录、注册表、安装状态、运行
- `CliAppsRuntimeConfig`：CLI App 运行时配置

这样其它模块只需要导入 `nanobot.apps.cli`，不必关心底层文件布局。
"""

from nanobot.apps.cli.service import (
    CliAppError,
    CliAppManager,
    CliAppsRuntimeConfig,
)

__all__ = [
    "CliAppError",
    "CliAppManager",
    "CliAppsRuntimeConfig",
]
