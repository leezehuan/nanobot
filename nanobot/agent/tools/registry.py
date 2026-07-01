"""工具注册表：统一管理 Agent 可调用工具。

模型并不会直接调用 Python 函数，而是先输出“我要调用哪个工具 + 参数是什么”。
这个注册表就是工具系统的中央目录，负责回答三类问题：

1. 当前有哪些工具可用？
2. 某个工具的 schema 定义是什么？
3. 模型给出的工具调用能否被解析、校验并执行？
"""

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """Agent 工具中心注册表。

    所有内置工具、MCP 包装工具、插件工具最终都会注册到这里，
    AgentRunner 再通过它完成工具定义获取与真实执行。
    """

    def __init__(self):
        """init。
        
        初始化 ToolRegistry 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """注册一个工具，并清空 schema 缓存。
        
        实现方法：把对象写入内部映射表，并清理依赖该映射生成的缓存。"""
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """按名字注销工具，并清空 schema 缓存。
        
        实现方法：把对象写入内部映射表，并清理依赖该映射生成的缓存。"""
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """按名字获取工具实例。
        
        实现方法：优先从显式参数或实例状态读取目标值，缺失时回退到默认配置，并把结果整理成调用方期望的类型。"""
        return self._tools.get(name)

    @staticmethod
    def _lookup_key(name: str) -> str:
        """生成“建议匹配键”。

        注意这里只用于“拼写建议”，绝不会用于真正执行。
        工具执行必须严格按原始名字匹配，避免模糊匹配带来风险。

        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
        """
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        """suggest name。
        
        实现方法：对输入名字生成宽松匹配键，只用于错误提示中的建议，不参与真实执行。"""
        key = self._lookup_key(str(name or ""))
        if not key:
            return None
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def has(self, name: str) -> bool:
        """判断某个工具是否已注册。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """从不同风格的 schema 结构里提取工具名。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """返回工具定义列表，并尽量保持稳定顺序。

        【为什么强调稳定顺序】
        工具定义会进入 prompt。顺序越稳定，越有利于 prompt cache 和结果可复现。

        【排序策略】
        - 内置工具排前面
        - MCP 工具排后面
        - 两组内部再按名字排序

        实现方法：把每个工具转换成 schema 后拆成内置工具和 MCP 工具两组分别排序，再拼回缓存列表，保证 prompt 中工具顺序稳定。
        """
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        """解析、类型转换并校验一次工具调用。
        
        实现方法：先按严格名称查找工具，找不到时只生成拼写建议；随后把字符串 JSON 参数还原、按工具 schema 做类型转换和校验，最后返回工具、参数和可恢复错误文本。"""
        tool = self._tools.get(name)
        if not tool:
            suggestion = self._suggest_name(str(name))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            return None, params, (
                f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}"
            )

        params = self._coerce_params(tool, params)
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got "
                f"{type(params).__name__}. Use named parameters like "
                'tool_name(param1="value1", param2="value2") matching the tool schema.'
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    @classmethod
    def _coerce_argument_value(cls, value: Any) -> Any:
        """把字符串形式的 JSON 参数尽量还原成真实对象。
        
        实现方法：把字符串、JSON 或宽松类型尽量转换成 schema 期望的 Python 值。"""
        if value is None:
            return {}
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return {}

        if not stripped.startswith(("{", "[")):
            return value

        try:
            parsed = json.loads(stripped)
        except Exception:
            return value

        return parsed

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any) -> Any:
        """coerce params。
        
        实现方法：先把 provider 传来的字符串参数尽量 JSON 解析，再兼容 arguments 包裹层，保证后续校验看到的是工具真正需要的参数对象。"""
        params = cls._coerce_argument_value(params)
        return cls._unwrap_arguments_payload(tool, params)

    @classmethod
    def _unwrap_arguments_payload(cls, tool: Tool, params: Any) -> Any:
        """兼容某些 provider 把参数再包一层 ``arguments`` 的情况。
        
        实现方法：只有当参数对象恰好只包含 arguments 且工具 schema 本身没有 arguments 字段时，才拆掉外层包装，避免误伤合法同名参数。"""
        if not isinstance(params, dict) or set(params) != {"arguments"}:
            return params
        properties = (tool.parameters or {}).get("properties", {})
        if isinstance(properties, dict) and "arguments" in properties:
            return params
        return cls._coerce_argument_value(params.get("arguments"))

    async def execute(self, name: str, params: Any) -> Any:
        """按名字执行工具。

        这里除了真正执行，还负责把常见错误包装成“可让模型继续恢复”的文本，
        这样模型看到报错后还能改参数再试，而不是整轮直接崩掉。

        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。
        """
        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            return error + hint

        try:
            assert tool is not None  # guarded by prepare_call()
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + hint

    @property
    def tool_names(self) -> list[str]:
        """返回当前已注册工具名列表。
        
        返回当前注册表里的工具名快照。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """len。
        
        实现方法：直接代理到底层集合或映射，使对象可以使用 Python 内置协议进行长度统计或成员判断。"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """contains。
        
        实现方法：直接代理到底层集合或映射，使对象可以使用 Python 内置协议进行长度统计或成员判断。"""
        return name in self._tools
