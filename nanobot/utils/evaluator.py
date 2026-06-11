"""后台任务结果评估器：判断 heartbeat / cron 的执行结果是否值得通知用户。

思路是：
- 先让后台 Agent 执行任务
- 再做一次轻量 LLM 调用判断“要不要打扰用户”
- 避免把例行、空白或无意义结果都推送出去
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_EVALUATE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_notification",
            "description": "Decide whether the user should be notified about this background task result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "should_notify": {
                        "type": "boolean",
                        "description": "true = result contains actionable/important info the user should see; false = routine or empty, safe to suppress",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence reason for the decision",
                    },
                },
                "required": ["should_notify"],
            },
        },
    }
]

async def evaluate_response(
    response: str,
    task_context: str,
    provider: LLMProvider,
    model: str,
    default_notify: bool = True,
) -> bool:
    """判断后台任务结果是否应该发给用户。

    失败兜底策略由 ``default_notify`` 决定：
    - cron 往往倾向于“失败时也通知”
    - heartbeat 往往倾向于“失败时安静忽略”
    """
    try:
        llm_response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": render_template("agent/evaluator.md", part="system")},
                {"role": "user", "content": render_template(
                    "agent/evaluator.md",
                    part="user",
                    task_context=task_context,
                    response=response,
                )},
            ],
            tools=_EVALUATE_TOOL,
            model=model,
            max_tokens=256,
            temperature=0.0,
        )

        if not llm_response.should_execute_tools:
            if llm_response.has_tool_calls:
                logger.warning(
                    "evaluate_response: ignoring tool calls under finish_reason='{}', "
                    "defaulting to notify={}",
                    llm_response.finish_reason,
                    default_notify,
                )
            else:
                logger.warning(
                    "evaluate_response: no tool call returned, defaulting to notify={}",
                    default_notify,
                )
            return default_notify

        args = llm_response.tool_calls[0].arguments
        should_notify = args.get("should_notify", default_notify)
        reason = args.get("reason", "")
        logger.info("evaluate_response: should_notify={}, reason={}", should_notify, reason)
        return bool(should_notify)

    except Exception:
        logger.exception("evaluate_response failed, defaulting to notify={}", default_notify)
        return default_notify
