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
    """使用 LLM 判断后台任务结果是否应该通知用户。

    【中文名称】后台结果评估

    【功能说明】
    当 heartbeat / cron 等后台任务执行完毕后，不是直接把结果推送给用户，
    而是先做一次轻量级 LLM 评估：模型通过调用 evaluate_notification 函数，
    声明 "should_notify=true/false" 来判断结果是否值得打扰用户。

    【为什么需要这个函数】
    如果不做评估，用户会收到大量"无事发生"的例行通知，造成骚扰。
    这个函数起到了"第二次决策"的作用，把"是否需要通知"的决策权下放给 LLM 本身。

    【完整的 3 个阶段】

    Phase 1: LLM 评估调用
      - system prompt: 从 templates/agent/evaluator.md 加载"评估者"角色提示
      - user prompt: 注入 task_context（任务背景）+ response（后台执行结果）
      - tools: 只有一个 evaluate_notification 函数（带 should_notify + reason 参数）
      - 参数: temperature=0.0、max_tokens=256（评估不需要长回答）

    Phase 2: 解析 LLM 决策
      - 正常路径: LLM 返回 tool_call，从 arguments 里取出 should_notify
      - 异常路径: LLM 返回了 tool_calls 但不允许执行 → 回退到 default_notify
      - 无调用路径: LLM 没调用工具 → 回退到 default_notify

    Phase 3: 兜底处理
      - 任何异常都捕获并回退到 default_notify
      - 这样即使 LLM 评估失败了，后台任务也不会被"静默丢弃"

    【参数说明】
    - response: 后台任务执行产生的最终文本（如搜索摘要、执行结论文本）
    - task_context: 任务背景说明，例如 "heartbeat check" 或 "cron job: daily news summary"
    - provider: LLMProvider 实例，用于执行评估 LLM 调用
    - model: 评估所用的模型名（通常比主模型小，因为评估任务轻量）
    - default_notify: 评估失败时的兜底行为
        - True（cron 倾向）: 拿不准时就通知用户，宁可多报也不错失重要信息
        - False（heartbeat 倾向）: 拿不准时就安静忽略，避免骚扰

    【返回值】
    - True: 结果值得通知用户（should_notify=true 或 评估失败 + default_notify=true）
    - False: 结果可忽略（should_notify=false 或 评估失败 + default_notify=false）
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
