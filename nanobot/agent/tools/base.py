"""工具系统基础抽象。

这个文件定义了 nanobot 工具系统最底层的两个概念：

1. ``Schema``：工具参数的 JSON Schema 片段描述
2. ``Tool``：一个可被模型调用的能力单元

理解这里之后，再看 ``filesystem.py`` / ``shell.py`` / ``web.py`` 等具体工具实现，
会清楚很多。
"""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar

if typing.TYPE_CHECKING:
    from pydantic import BaseModel

    from nanobot.agent.tools.context import ToolContext

_ToolT = TypeVar("_ToolT", bound="Tool")

# 这个映射定义了 JSON Schema 基本类型到 Python 运行时类型的对应关系。
# ``Tool._cast_value`` 和 ``Schema.validate_json_schema_value`` 都会用到它。
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Schema(ABC):
    """工具参数 JSON Schema 片段的抽象基类。

    具体子类在 ``nanobot.agent.tools.schema`` 里，比如：
    - ``StringSchema``
    - ``IntegerSchema``
    - ``ObjectSchema``

    这些类的意义是：让工具作者可以用更 Python 化的方式描述参数结构，
    最后再统一导出成模型函数调用所需的 JSON Schema。
    """

    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        """从 JSON Schema ``type`` 中提取非 null 的主类型。
        
        实现方法：先规范化名字或路径，再结合配置、预设和默认值得到最终可执行对象。"""
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t  # type: ignore[return-value]

    @staticmethod
    def subpath(path: str, key: str) -> str:
        """拼接错误路径，例如 ``foo.bar``。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(val: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        """按 JSON Schema 片段校验一个值，返回错误信息列表。

        返回空列表表示校验通过。
        这是 ``Tool.validate_params`` 以及各具体 Schema 的公共校验核心。

        实现方法：先做格式与安全边界检查，再把失败原因转换成调用方可读的错误信息。
        """
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(val, _JSON_TYPE_MAP[t]):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            for k, v in val.items():
                if k in props:
                    errors.extend(Schema.validate_json_schema_value(v, props[k], Schema.subpath(path, k)))
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(val):
                    errors.extend(
                        Schema.validate_json_schema_value(item, schema["items"], prefix.format(i))
                    )
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        """把 Schema 实例或原始 dict 统一规范成 JSON Schema 片段 dict。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        # 先尝试 ``to_json_schema()``，以区分“Schema 实例”和“本来就是 dict 的 schema”
        to_js = getattr(value, "to_json_schema", None)
        if callable(to_js):
            return to_js()
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """导出成 JSON Schema 片段字典。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        """校验单个值；返回空列表表示通过。
        
        实现方法：先做格式与安全边界检查，再把失败原因转换成调用方可读的错误信息。"""
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


class Tool(ABC):
    """Agent 可调用工具的抽象基类。

    你可以把 Tool 理解成“暴露给模型的一项能力”，例如：
    - 读文件
    - 写文件
    - 执行 shell
    - 网页搜索
    - 发送消息
    """

    _TYPE_MAP = _JSON_TYPE_MAP
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        """Pick first non-null type from JSON Schema unions like ``['string','null']``.
        
        实现方法：先规范化名字或路径，再结合配置、预设和默认值得到最终可执行对象。"""
        return Schema.resolve_json_schema_type(t)

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名：模型发起函数调用时使用的唯一名字。
        
        返回工具或 provider 对外暴露的稳定名称。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具说明：会进入 prompt，帮助模型理解这个工具做什么。
        
        返回给模型看的能力说明，帮助模型判断何时调用本工具。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具参数的 JSON Schema。
        
        返回本工具的参数 JSON Schema，供模型按结构生成调用参数。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        ...

    @property
    def read_only(self) -> bool:
        """该工具是否无副作用，并适合并行执行。
        
        声明工具是否只读，供调度器判断并发和安全策略。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """该工具是否可以与其他并发安全工具一起执行。
        
        声明工具是否适合并发执行，供 Runner 对工具调用分批。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return self.read_only and not self.exclusive

    @property
    def exclusive(self) -> bool:
        """即便启用了并发，这个工具是否也必须独占执行。
        
        声明工具是否需要独占执行，避免和其他工具并发造成状态冲突。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
        return False

    # --- 插件元数据：供 ToolLoader / 插件系统使用 ---

    config_key: str = ""
    _plugin_discoverable: bool = True
    _scopes: set[str] = {"core"}

    @classmethod
    def config_cls(cls) -> type[BaseModel] | None:
        """返回该工具对应的配置模型类；没有则返回 ``None``。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return None

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        """判断当前上下文下该工具是否应启用。
        
        实现方法：从输入值和当前配置中提取关键标志，按布尔条件组合判断，并把异常或空值按保守结果处理。"""
        return True

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        """按上下文创建工具实例。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return cls()

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具主体。
        
        实现方法：先校验参数和运行时上下文，再调用具体工具逻辑；异常会被包装成模型可继续修正的文本结果。"""
        ...

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """cast object。
        
        实现方法：根据 JSON Schema 递归转换参数类型，使模型给出的字符串数字或布尔值能落到正确类型。"""
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {k: self._cast_value(v, props[k]) if k in props else v for k, v in obj.items()}

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """在校验前做一层安全的、基于 schema 的类型转换。
        
        实现方法：根据 JSON Schema 递归转换参数类型，使模型给出的字符串数字或布尔值能落到正确类型。"""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        """cast value。
        
        实现方法：根据 JSON Schema 递归转换参数类型，使模型给出的字符串数字或布尔值能落到正确类型。"""
        t = self._resolve_type(schema.get("type"))

        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[t]
            if isinstance(val, expected):
                return val

        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val

        if t == "string":
            return val if val is None else str(val)

        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                return True
            if low in self._BOOL_FALSE:
                return False
            return val

        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            return [self._cast_value(x, items) for x in val] if items else val

        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """按 JSON Schema 校验参数。
        
        实现方法：先做格式与安全边界检查，再把失败原因转换成调用方可读的错误信息。"""
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        """导出成 OpenAI 风格函数调用 schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """类装饰器：把 JSON Schema 挂到 ``Tool`` 子类上。

    工具作者可以不用手写 ``@property def parameters``，直接用这个装饰器即可。

    示例::

        @tool_parameters({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        class ReadFileTool(Tool):
            ...

    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。
    """

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        """decorator。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            """parameters。
            
            返回本工具的参数 JSON Schema，供模型按结构生成调用参数。 实现方法：直接从实例状态或常量配置中取值，必要时委托已有切换逻辑保持状态一致。"""
            return deepcopy(frozen)

        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]

        return cls

    return decorator
