"""面向 WebUI 会话的自动化任务载荷拼装。

【中文名称】会话自动化任务筛选器

Cron 服务里可能存着很多任务，但 WebUI 某个会话通常只关心：
“有哪些自动化任务跟我这个会话有关？”

这个模块就负责把全局 Cron 任务中过滤出当前 session 对应的那一小部分。
"""

from __future__ import annotations

from typing import Any, Protocol

from nanobot.cron.types import CronJob


class _CronServiceLike(Protocol):
    """这里不要求真实 `CronService` 类型，只要求具备 `list_jobs` 接口即可。"""
    def list_jobs(self, *, include_disabled: bool = False) -> list[CronJob]: ...


def session_automations_payload(
    cron_service: _CronServiceLike | None,
    session_key: str,
) -> dict[str, Any]:
    """返回绑定到某个 WebUI 会话上的自动化任务。"""
    jobs: list[CronJob] = []
    if cron_service is not None:
        all_jobs = cron_service.list_jobs(include_disabled=True)
        jobs = [job for job in all_jobs if _job_matches_session(job, session_key)]
    return {"jobs": [_serialize_job(job) for job in jobs]}


def _job_matches_session(job: CronJob, session_key: str) -> bool:
    """判断某个 Cron 任务是否属于当前会话。"""
    payload = job.payload
    if payload.kind != "agent_turn":
        return False
    if payload.session_key:
        return payload.session_key == session_key
    if payload.channel and payload.to:
        return f"{payload.channel}:{payload.to}" == session_key
    return False


def _serialize_job(job: CronJob) -> dict[str, Any]:
    """把任务对象裁剪成前端展示所需的精简结构。"""
    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": {
            "kind": job.schedule.kind,
            "at_ms": job.schedule.at_ms,
            "every_ms": job.schedule.every_ms,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "payload": {
            "message": job.payload.message,
        },
        "state": {
            "next_run_at_ms": job.state.next_run_at_ms,
            "last_status": job.state.last_status,
        },
    }
