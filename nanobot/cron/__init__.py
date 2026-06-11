"""定时任务服务导出入口。

【中文名称】Cron 调度模块入口

这个包为 nanobot 提供“未来某个时间自动执行任务”的能力。
对 Agent 来说，它相当于一个“延迟触发器”或“计划任务系统”。

为了减少导入开销，`CronService` 采用懒加载：
只有真正访问它时，才会导入 `nanobot.cron.service`。
"""

from nanobot.cron.types import CronJob, CronSchedule

__all__ = ["CronService", "CronJob", "CronSchedule"]

_LAZY = {"CronService": ".service"}


def __getattr__(name: str):
    """按需导入 `CronService`，避免无谓加载较重依赖。"""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    mod = import_module(module_path, __name__)
    val = getattr(mod, name)
    globals()[name] = val
    return val
