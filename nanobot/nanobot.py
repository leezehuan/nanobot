"""nanobot 的高级程序化接口（SDK facade）。

【中文名称】Nanobot SDK 门面

【功能说明】
这是外部 Python 代码使用 nanobot 的推荐入口。它封装了配置加载、
AgentLoop 创建、单次运行和资源释放，提供一个简洁的 async API。

【使用示例】
```python
bot = Nanobot.from_config()
result = await bot.run("Summarize this repo", hooks=[MyHook()])
print(result.content)
```

【完整的生命周期】
1. from_config() 加载 ~/.nanobot/config.json → 创建 AgentLoop
2. run(message) 执行单次 agent 对话，返回 RunResult
3. aclose() 释放 MCP 连接等资源
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook, SDKCaptureHook
from nanobot.agent.loop import AgentLoop
from nanobot.providers.image_generation import image_gen_provider_configs


@dataclass(slots=True)
class RunResult:
    """单次 agent 运行的结果。

    【字段说明】
    - content: Agent 最终输出的文本（对于纯 tool-call 无文本的 turn 可能为空字符串）
    - tools_used: 本轮使用过的工具名列表（通过 SDKCaptureHook 自动收集）
    - messages: 完整的对话消息列表，包含 system prompt、user message、assistant 响应、tool call/results
    """

    content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]


class Nanobot:
    """nanobot agent 的程序化门面类。

    【中文名称】Nanobot SDK 入口

    【内部结构】
    - _loop: AgentLoop 实例（由 from_config() 创建并注入）
    - 通过 _extra_hooks 注入 SDKCaptureHook 来收集本轮的工具调用和消息

    【使用方式】
    ```python
    # 方式 1: 从配置文件构造
    bot = Nanobot.from_config()
    result = await bot.run("Hello!")

    # 方式 2: async context manager（自动清理）
    async with Nanobot.from_config() as bot:
        result = await bot.run("Hello!")
    ```
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> Nanobot:
        """从配置文件创建 Nanobot 实例。

        【中文名称】从配置文件构造

        【完整的 4 个阶段】

        Phase 1: 加载配置文件
          - 如果指定 config_path，使用该路径，否则用默认 ~/.nanobot/config.json
          - 文件不存在时抛出 FileNotFoundError

        Phase 2: 解析环境变量
          - resolve_config_env_vars() 会把 ${VAR} 占位符替换为实际环境变量值
          - 这是 nanobot 唯一支持"配置模板化"的环节

        Phase 3: 覆盖 workspace
          - 如果传了 workspace 参数，覆盖配置中的 agents.defaults.workspace
          - 这样同一个 SDK 可以在不修改配置文件的前提下"临时换工作目录"

        Phase 4: 创建 AgentLoop
          - AgentLoop.from_config(config, image_gen_providers) 解析 provider、tools、channels、memory 等
          - 返回包装好的 Nanobot 实例

        【参数说明】
        - config_path: 配置文件路径（可选，默认 ~/.nanobot/config.json）
        - workspace: 覆盖配置文件中的工作区目录（可选）

        【返回值】
        - Nanobot: 新实例，可用。调用 run() 开始对话
        """
        from nanobot.config.loader import load_config, resolve_config_env_vars
        from nanobot.config.schema import Config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(load_config(resolved))
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )

        loop = AgentLoop.from_config(
            config,
            image_generation_provider_configs=image_gen_provider_configs(config),
        )
        return cls(loop)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[AgentHook] | None = None,
    ) -> RunResult:
        """执行一次 agent 对话并返回结果。

        【中文名称】运行一次 Agent 对话

        【完整的 6 个阶段】

        Phase 1: 注入 SDK 收集 hook
          - 创建 SDKCaptureHook 实例
          - 它会在 agent 执行过程中自动收集 tools_used 和 messages
          - 保存在 _extra_hooks 的最前面

        Phase 2: 合并用户 hooks
          - 如果调用方传了自定义 hooks，追加到收集 hook 后面
          - 如果 _extra_hooks 原本就有值（前一次 run 有残留），以新传入的为准

        Phase 3: 执行 AgentLoop.process_direct()
          - 这是 nanobot 的真正"大脑"
          - 内部会：构建 system prompt → 调用 LLM → 执行 tool calls → 循环直到 LLM 不再调工具
          - message 作为用户消息传入，session_key 用于隔离不同会话的历史记录

        Phase 4: 恢复 hooks 状态
          - 在 finally 块中把 _extra_hooks 恢复为 run() 调用前的状态
          - 这是为了防止 SDKCaptureHook 在多轮调用中持续累积

        Phase 5: 提取最终内容
          - 从 response.content 获取 LLM 最终输出文本
          - 如果为 None 则退化为空字符串

        Phase 6: 构造 RunResult
          - content: LLM 最终输出
          - tools_used: SDKCaptureHook 收集的工具名列表
          - messages: 完整对话记录

        【参数说明】
        - message: 用户消息文本（例如 "Summarize this repo"）
        - session_key: 会话标识符，不同 key 拥有独立的对话历史
          （默认 "sdk:default"）
        - hooks: 可选的额外生命周期钩子列表

        【返回值】
        - RunResult: 包含 content、tools_used、messages 的结果对象
        """
        capture = SDKCaptureHook()
        prev = self._loop._extra_hooks
        base_hooks = list(hooks) if hooks is not None else list(prev or [])
        self._loop._extra_hooks = [capture, *base_hooks]
        try:
            response = await self._loop.process_direct(
                message, session_key=session_key,
            )
        finally:
            self._loop._extra_hooks = prev

        content = (response.content if response else None) or ""
        return RunResult(
            content=content,
            tools_used=capture.tools_used,
            messages=capture.messages,
        )

    async def aclose(self) -> None:
        """释放本实例持有的资源（MCP 连接等）。

        【中文名称】关闭并释放资源

        【注意】
        如果不调用 aclose()，MCP 连接可能一直处于打开状态。
        推荐使用 async with 语法自动释放。
        """
        await self._loop.close_mcp()

    async def __aenter__(self) -> Nanobot:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

