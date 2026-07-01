"""常用 JSON Schema 片段类型。

这些类是工具参数定义层的“积木”：
- ``StringSchema``：字符串字段
- ``IntegerSchema``：整数字段
- ``NumberSchema``：数字字段
- ``BooleanSchema``：布尔字段
- ``ArraySchema``：数组字段
- ``ObjectSchema``：对象字段

工具作者可以用它们组合出参数结构，再交给 ``Tool.parameters`` 或
``tool_parameters_schema`` 使用。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanobot.agent.tools.base import Schema


class StringSchema(Schema):
    """字符串参数定义。"""

    def __init__(
        self,
        description: str = "",
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        enum: tuple[Any, ...] | list[Any] | None = None,
        nullable: bool = False,
    ) -> None:
        """init。
        
        初始化 StringSchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._description = description
        self._min_length = min_length
        self._max_length = max_length
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "string"
        if self._nullable:
            t = ["string", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._min_length is not None:
            d["minLength"] = self._min_length
        if self._max_length is not None:
            d["maxLength"] = self._max_length
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class IntegerSchema(Schema):
    """整数参数定义。"""

    def __init__(
        self,
        value: int = 0,
        *,
        description: str = "",
        minimum: int | None = None,
        maximum: int | None = None,
        enum: tuple[int, ...] | list[int] | None = None,
        nullable: bool = False,
    ) -> None:
        """init。
        
        初始化 IntegerSchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "integer"
        if self._nullable:
            t = ["integer", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._minimum is not None:
            d["minimum"] = self._minimum
        if self._maximum is not None:
            d["maximum"] = self._maximum
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class NumberSchema(Schema):
    """数字参数定义。"""

    def __init__(
        self,
        value: float = 0.0,
        *,
        description: str = "",
        minimum: float | None = None,
        maximum: float | None = None,
        enum: tuple[float, ...] | list[float] | None = None,
        nullable: bool = False,
    ) -> None:
        """init。
        
        初始化 NumberSchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "number"
        if self._nullable:
            t = ["number", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._minimum is not None:
            d["minimum"] = self._minimum
        if self._maximum is not None:
            d["maximum"] = self._maximum
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class BooleanSchema(Schema):
    """布尔参数定义。

    Python 不允许继承 ``bool``，所以这里单独做一个 Schema 类。
    """

    def __init__(
        self,
        *,
        description: str = "",
        default: bool | None = None,
        nullable: bool = False,
    ) -> None:
        """init。
        
        初始化 BooleanSchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._description = description
        self._default = default
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "boolean"
        if self._nullable:
            t = ["boolean", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._default is not None:
            d["default"] = self._default
        return d


class ArraySchema(Schema):
    """数组参数定义，由 ``items`` 指定元素类型。"""

    def __init__(
        self,
        items: Any | None = None,
        *,
        description: str = "",
        min_items: int | None = None,
        max_items: int | None = None,
        nullable: bool = False,
    ) -> None:
        """init。
        
        初始化 ArraySchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._items_schema: Any = items if items is not None else StringSchema("")
        self._description = description
        self._min_items = min_items
        self._max_items = max_items
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "array"
        if self._nullable:
            t = ["array", "null"]
        d: dict[str, Any] = {
            "type": t,
            "items": Schema.fragment(self._items_schema),
        }
        if self._description:
            d["description"] = self._description
        if self._min_items is not None:
            d["minItems"] = self._min_items
        if self._max_items is not None:
            d["maxItems"] = self._max_items
        return d


class ObjectSchema(Schema):
    """对象参数定义。

    ``properties`` 或关键字参数中的键是字段名，值可以是：
    - 子 Schema
    - 原始 JSON Schema dict
    """

    def __init__(
        self,
        properties: Mapping[str, Any] | None = None,
        *,
        required: list[str] | None = None,
        description: str = "",
        additional_properties: bool | dict[str, Any] | None = None,
        nullable: bool = False,
        **kwargs: Any,
    ) -> None:
        """init。
        
        初始化 ObjectSchema 实例。实现方法：把构造参数保存到实例字段，创建后续调用需要复用的缓存、状态容器或运行时依赖。"""
        self._properties = dict(properties or {}, **kwargs)
        self._required = list(required or [])
        self._root_description = description
        self._additional_properties = additional_properties
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        """to json schema。
        
        实现方法：把内部对象字段映射到目标格式，递归转换嵌套结构，并过滤目标协议不需要的空字段。"""
        t: Any = "object"
        if self._nullable:
            t = ["object", "null"]
        props = {k: Schema.fragment(v) for k, v in self._properties.items()}
        out: dict[str, Any] = {"type": t, "properties": props}
        if self._required:
            out["required"] = self._required
        if self._root_description:
            out["description"] = self._root_description
        if self._additional_properties is not None:
            out["additionalProperties"] = self._additional_properties
        return out


def tool_parameters_schema(
    *,
    required: list[str] | None = None,
    description: str = "",
    **properties: Any,
) -> dict[str, Any]:
    """快速构建工具根参数对象 schema。
    
    实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
    return ObjectSchema(
        required=required,
        description=description,
        **properties,
    ).to_json_schema()
