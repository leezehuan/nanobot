"""Cron 调度系统使用的数据类型定义。

【中文名称】定时任务模型

这个文件只定义“数据长什么样”，不负责真正调度任务。
你可以把它理解成 Cron 子系统的结构化词汇表：

- `CronSchedule`：什么时候执行
- `CronPayload`：执行时要做什么
- `CronRunRecord`：某次执行结果
- `CronJobState`：运行期状态
- `CronJob`：完整任务对象
- `CronStore`：磁盘上的任务集合
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """定时任务的触发计划。

    `kind` 决定当前使用哪一种调度方式：

    - `at`：某个绝对时间点执行一次
    - `every`：按固定间隔重复执行
    - `cron`：按 cron 表达式执行
    """
    kind: Literal["at", "every", "cron"]
    # `at` 模式下使用：毫秒时间戳。
    at_ms: int | None = None
    # `every` 模式下使用：重复间隔，单位毫秒。
    every_ms: int | None = None
    # `cron` 模式下使用：标准 cron 表达式，例如 "0 9 * * *"。
    expr: str | None = None
    # cron 表达式解释所使用的时区。
    tz: str | None = None


@dataclass
class CronPayload:
    """任务触发后具体要执行的动作。"""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # 是否把执行结果回送到频道/聊天端。
    deliver: bool = False
    channel: str | None = None  # 目标频道名，例如 "whatsapp"。
    to: str | None = None  # 目标地址，例如手机号、用户 ID、会话 ID。
    channel_meta: dict = field(default_factory=dict)  # 频道专属路由信息，例如 Slack 的 thread_ts。
    session_key: str | None = None  # 原始会话 key，保证日志和上下文写回到正确会话。


@dataclass
class CronRunRecord:
    """某个定时任务的一次执行记录。"""
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CronJobState:
    """任务在运行期不断变化的状态。"""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass
class CronJob:
    """完整的定时任务对象。"""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False

    @classmethod
    def from_dict(cls, kwargs: dict):
        """把磁盘/接口里的普通字典还原成 `CronJob` 对象。"""
        state_kwargs = dict(kwargs.get("state", {}))
        state_kwargs["run_history"] = [
            record if isinstance(record, CronRunRecord) else CronRunRecord(**record)
            for record in state_kwargs.get("run_history", [])
        ]
        kwargs["schedule"] = CronSchedule(**kwargs.get("schedule", {"kind": "every"}))
        kwargs["payload"] = CronPayload(**kwargs.get("payload", {}))
        kwargs["state"] = CronJobState(**state_kwargs)
        return cls(**kwargs)


@dataclass
class CronStore:
    """持久化到磁盘上的任务集合。"""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
