"""Provider 注册表：LLM 服务商元数据的单一事实来源。

如果你想新增一个 Provider，通常只需要两步：
1. 在下面的 ``PROVIDERS`` 里加一条 ``ProviderSpec``
2. 在 ``config/schema.py`` 的 ``ProvidersConfig`` 里加对应字段

之后大部分自动匹配、环境变量推断、状态展示都会跟着这里走。

特别注意：``PROVIDERS`` 的顺序有意义，它会影响匹配优先级和 fallback 顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic.alias_generators import to_snake


@dataclass(frozen=True)
class ProviderSpec:
    """单个 Provider 的元数据描述。

    你可以把它理解成“Provider 的配置说明书 + 匹配说明书”。
    Factory、Config 自动匹配等逻辑都会参考这里。
    """

    # 基础身份信息
    name: str  # 配置字段名，例如 "dashscope"
    keywords: tuple[str, ...]  # 模型名关键词，用于自动匹配
    env_key: str  # API key 对应环境变量名
    display_name: str = ""  # 给用户展示的名字

    # 选择哪种底层 Provider 实现
    backend: str = "openai_compat"

    # 额外需要注入的环境变量
    env_extras: tuple[tuple[str, str], ...] = ()

    # 网关 / 本地部署识别
    is_gateway: bool = False  # 是否是可转发任意模型的网关型 Provider
    is_local: bool = False  # 是否是本地部署 Provider
    detect_by_key_prefix: str = ""  # 通过 api_key 前缀识别
    detect_by_base_keyword: str = ""  # 通过 api_base 中的关键字识别
    default_api_base: str = ""  # 默认 API Base

    # 网关行为
    strip_model_prefix: bool = False  # 发送给网关前是否移除 "provider/" 前缀
    supports_max_completion_tokens: bool = False

    # 针对特定模型的参数覆盖
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # OAuth 类 Provider（如 OpenAI Codex）不依赖传统 API key
    is_oauth: bool = False

    # Direct Provider 跳过 API key 校验
    is_direct: bool = False

    # 这个 Provider 只用于转写等能力，不能承载普通 chat completion
    is_transcription_only: bool = False

    # 是否支持 content block 级别的 prompt caching
    supports_prompt_caching: bool = False

    # 如何向 extra_body 注入 thinking 开关
    thinking_style: str = ""

    # 网关原生 reasoning 控制方式
    gateway_reasoning_style: str = ""

    # 某些 Provider 把真正回答放在 reasoning 字段而不是 content 字段
    reasoning_as_content: bool = False

    @property
    def label(self) -> str:
        """返回更适合展示给用户的 Provider 名称。
        
        实现方法：围绕当前模块的运行时状态组织输入、执行核心判断或数据转换，并把结果返回给上层流程继续使用。"""
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS 注册表本体。顺序 = 匹配优先级。
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (
    # === Custom：用户手填的直连 OpenAI-compatible 端点 ======================
    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="",
        display_name="Custom",
        backend="openai_compat",
        is_direct=True,
    ),

    # === Azure OpenAI：直连 Azure OpenAI API ===============================
    ProviderSpec(
        name="azure_openai",
        keywords=("azure", "azure-openai"),
        env_key="",
        display_name="Azure OpenAI",
        backend="azure_openai",
        is_direct=True,
    ),
    # === AWS Bedrock：通过原生 Converse API 接入 ============================
    ProviderSpec(
        name="bedrock",
        keywords=(
            "bedrock",
            "anthropic.claude",
            "amazon.nova",
            "meta.",
            "mistral.",
            "cohere.",
            "qwen.",
            "deepseek.",
            "openai.gpt-oss",
            "ai21.",
            "moonshot.",
            "writer.",
            "zai.",
        ),
        env_key="AWS_BEARER_TOKEN_BEDROCK",
        display_name="AWS Bedrock",
        backend="bedrock",
        is_direct=True,
    ),
    # === 网关型 Provider：主要通过 api_key / api_base 识别，而不是模型名 =====
    # 网关通常可以转发任意模型，因此 fallback 阶段优先级更高。
    # OpenRouter：全局模型网关，key 通常以 ``sk-or-`` 开头。
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        backend="openai_compat",
        is_gateway=True,
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        default_api_base="https://openrouter.ai/api/v1",
        supports_prompt_caching=True,
        gateway_reasoning_style="reasoning_effort",
    ),
    # Hugging Face Inference Providers：聊天模型的 OpenAI-compatible 路由层。
    ProviderSpec(
        name="huggingface",
        keywords=("huggingface", "hugging-face"),
        env_key="HF_TOKEN",
        display_name="Hugging Face",
        backend="openai_compat",
        is_gateway=True,
        detect_by_key_prefix="hf_",
        detect_by_base_keyword="huggingface",
        default_api_base="https://router.huggingface.co/v1",
    ),
    # Skywork / APIFree：OpenAI-compatible 的 MaaS 网关。
    ProviderSpec(
        name="skywork",
        keywords=("skywork", "skyclaw", "apifree"),
        env_key="SKYWORK_API_KEY",
        display_name="Skywork",
        backend="openai_compat",
        env_extras=(("APIFREE_API_KEY", "{api_key}"),),
        is_gateway=True,
        detect_by_base_keyword="apifree.ai",
        default_api_base="https://api.apifree.ai/agent/v1",
    ),
    # AiHubMix：全局网关，接口兼容 OpenAI。
    # ``strip_model_prefix=True`` 表示它不理解 ``anthropic/claude-3`` 这种前缀模型名，
    # 发送前要裁成裸模型名 ``claude-3``。
    ProviderSpec(
        name="aihubmix",
        keywords=("aihubmix",),
        env_key="OPENAI_API_KEY",
        display_name="AiHubMix",
        backend="openai_compat",
        is_gateway=True,
        detect_by_base_keyword="aihubmix",
        default_api_base="https://aihubmix.com/v1",
        strip_model_prefix=True,
    ),
    # SiliconFlow（硅基流动）：OpenAI-compatible 网关，模型名保留组织前缀。
    ProviderSpec(
        name="siliconflow",
        keywords=("siliconflow",),
        env_key="OPENAI_API_KEY",
        display_name="SiliconFlow",
        backend="openai_compat",
        is_gateway=True,
        detect_by_base_keyword="siliconflow",
        default_api_base="https://api.siliconflow.cn/v1",
    ),

    # Novita AI：托管模型 API 的 OpenAI-compatible 网关。
    ProviderSpec(
        name="novita",
        keywords=("novita",),
        env_key="NOVITA_API_KEY",
        display_name="Novita AI",
        backend="openai_compat",
        is_gateway=True,
        detect_by_base_keyword="novita",
        default_api_base="https://api.novita.ai/openai",
    ),

    # VolcEngine（火山引擎）：OpenAI-compatible 网关，按量计费模型。
    ProviderSpec(
        name="volcengine",
        keywords=("volcengine", "volces", "ark"),
        env_key="OPENAI_API_KEY",
        display_name="VolcEngine",
        backend="openai_compat",
        is_gateway=True,
        detect_by_base_keyword="volces",
        default_api_base="https://ark.cn-beijing.volces.com/api/v3",
        thinking_style="thinking_type",
        supports_max_completion_tokens=True,
    ),

    # VolcEngine Coding Plan（火山引擎 Coding Plan）：与 volcengine 共用同一套 key。
    ProviderSpec(
        name="volcengine_coding_plan",
        keywords=("volcengine-plan",),
        env_key="OPENAI_API_KEY",
        display_name="VolcEngine Coding Plan",
        backend="openai_compat",
        is_gateway=True,
        default_api_base="https://ark.cn-beijing.volces.com/api/coding/v3",
        strip_model_prefix=True,
        thinking_style="thinking_type",
        supports_max_completion_tokens=True,
    ),

    # BytePlus：火山引擎国际版，按量计费模型。
    ProviderSpec(
        name="byteplus",
        keywords=("byteplus",),
        env_key="OPENAI_API_KEY",
        display_name="BytePlus",
        backend="openai_compat",
        is_gateway=True,
        detect_by_base_keyword="bytepluses",
        default_api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
        strip_model_prefix=True,
        thinking_style="thinking_type",
    ),

    # BytePlus Coding Plan：与 byteplus 共用同一套 key。
    ProviderSpec(
        name="byteplus_coding_plan",
        keywords=("byteplus-plan",),
        env_key="OPENAI_API_KEY",
        display_name="BytePlus Coding Plan",
        backend="openai_compat",
        is_gateway=True,
        default_api_base="https://ark.ap-southeast.bytepluses.com/api/coding/v3",
        strip_model_prefix=True,
        thinking_style="thinking_type",
    ),


    # === 标准 Provider：主要靠模型名关键词匹配 ===============================
    # Anthropic：走原生 Anthropic SDK
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        backend="anthropic",
        supports_prompt_caching=True,
    ),
    # OpenAI：使用 SDK 默认 base URL，无需额外覆盖
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        backend="openai_compat",
        supports_max_completion_tokens=True,
    ),
    # OpenAI Codex：基于 OAuth 的专用 Provider
    ProviderSpec(
        name="openai_codex",
        keywords=("openai-codex",),
        env_key="",
        display_name="OpenAI Codex",
        backend="openai_codex",
        detect_by_base_keyword="codex",
        default_api_base="https://chatgpt.com/backend-api",
        is_oauth=True,
    ),
    # GitHub Copilot：基于 OAuth
    ProviderSpec(
        name="github_copilot",
        keywords=("github_copilot", "copilot"),
        env_key="",
        display_name="Github Copilot",
        backend="github_copilot",
        default_api_base="https://api.githubcopilot.com",
        strip_model_prefix=True,
        is_oauth=True,
        supports_max_completion_tokens=True,
    ),
    # DeepSeek：``api.deepseek.com`` 上的 OpenAI-compatible 接口
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        backend="openai_compat",
        default_api_base="https://api.deepseek.com",
        thinking_style="thinking_type",
    ),
    # Gemini：Google 提供的 OpenAI-compatible 入口
    ProviderSpec(
        name="gemini",
        keywords=("gemini", "gemma"),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        backend="openai_compat",
        default_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    # Zhipu（智谱）：``open.bigmodel.cn`` 上的 OpenAI-compatible 接口
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        display_name="Zhipu AI",
        backend="openai_compat",
        env_extras=(("ZHIPUAI_API_KEY", "{api_key}"),),
        default_api_base="https://open.bigmodel.cn/api/paas/v4",
    ),
    # DashScope（通义）：Qwen 模型的 OpenAI-compatible 接口
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        env_key="DASHSCOPE_API_KEY",
        display_name="DashScope",
        backend="openai_compat",
        default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        thinking_style="enable_thinking",
    ),
    # Moonshot（月之暗面）：Kimi K2.5 / K2.6 强制要求 temperature >= 1.0
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        display_name="Moonshot",
        backend="openai_compat",
        default_api_base="https://api.moonshot.ai/v1",
        model_overrides=(
            ("kimi-k2.5", {"temperature": 1.0}),
            ("kimi-k2.6", {"temperature": 1.0}),
        ),
    ),
    # MiniMax：OpenAI-compatible API
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        env_key="MINIMAX_API_KEY",
        display_name="MiniMax",
        backend="openai_compat",
        default_api_base="https://api.minimax.io/v1",
        thinking_style="reasoning_split",
    ),
    # MiniMax Anthropic-compatible 端点：支持 thinking 模式
    ProviderSpec(
        name="minimax_anthropic",
        keywords=("minimax_anthropic",),
        env_key="MINIMAX_API_KEY",
        display_name="MiniMax (Anthropic)",
        backend="anthropic",
        default_api_base="https://api.minimax.io/anthropic",
    ),
    # Mistral AI：OpenAI-compatible API
    ProviderSpec(
        name="mistral",
        keywords=("mistral",),
        env_key="MISTRAL_API_KEY",
        display_name="Mistral",
        backend="openai_compat",
        default_api_base="https://api.mistral.ai/v1",
    ),
    # Step Fun（阶跃星辰）：OpenAI-compatible API
    ProviderSpec(
        name="stepfun",
        keywords=("stepfun", "step"),
        env_key="STEPFUN_API_KEY",
        display_name="Step Fun",
        backend="openai_compat",
        default_api_base="https://api.stepfun.com/v1",
        reasoning_as_content=True,
    ),
    # Xiaomi MIMO（小米）：OpenAI-compatible API
    # 托管接口 ``api.xiaomimimo.com`` 接受
    # ``{"thinking": {"type": "enabled"|"disabled"}}`` 来切换 reasoning，
    # 与现有的 ``thinking_type`` 风格保持一致。
    ProviderSpec(
        name="xiaomi_mimo",
        keywords=("xiaomi_mimo", "mimo"),
        env_key="XIAOMIMIMO_API_KEY",
        display_name="Xiaomi MIMO",
        backend="openai_compat",
        default_api_base="https://api.xiaomimimo.com/v1",
        thinking_style="thinking_type",
    ),
    # LongCat：OpenAI-compatible API
    ProviderSpec(
        name="longcat",
        keywords=("longcat",),
        env_key="LONGCAT_API_KEY",
        display_name="LongCat",
        backend="openai_compat",
        default_api_base="https://api.longcat.chat/openai/v1",
    ),
    # Ant Ling：面向 Ling / Ring 模型族的 OpenAI-compatible API
    ProviderSpec(
        name="ant_ling",
        keywords=("ant_ling", "ant-ling", "ling-", "ring-"),
        env_key="ANT_LING_API_KEY",
        display_name="Ant Ling",
        backend="openai_compat",
        detect_by_base_keyword="ant-ling.com",
        default_api_base="https://api.ant-ling.com/v1",
    ),
    # === 本地部署 Provider：按配置字段匹配，而不是按 api_base ================
    # vLLM / 任意 OpenAI-compatible 本地服务
    ProviderSpec(
        name="vllm",
        keywords=("vllm",),
        env_key="HOSTED_VLLM_API_KEY",
        display_name="vLLM",
        backend="openai_compat",
        is_local=True,
    ),
    # Ollama（本地，OpenAI-compatible）
    ProviderSpec(
        name="ollama",
        keywords=("ollama", "nemotron"),
        env_key="OLLAMA_API_KEY",
        display_name="Ollama",
        backend="openai_compat",
        is_local=True,
        detect_by_base_keyword="11434",
        default_api_base="http://localhost:11434/v1",
    ),
    # LM Studio（本地，OpenAI-compatible）
    ProviderSpec(
        name="lm_studio",
        keywords=("lm-studio", "lmstudio", "lm_studio"),
        env_key="LM_STUDIO_API_KEY",
        display_name="LM Studio",
        backend="openai_compat",
        is_local=True,
        detect_by_base_keyword="1234",
        default_api_base="http://localhost:1234/v1",
    ),
    # Atomic Chat（本地，OpenAI-compatible）— https://atomic.chat/
    ProviderSpec(
        name="atomic_chat",
        keywords=("atomic-chat", "atomic_chat", "atomicchat"),
        env_key="ATOMIC_CHAT_API_KEY",
        display_name="Atomic Chat",
        backend="openai_compat",
        is_local=True,
        detect_by_base_keyword="1337",
        default_api_base="http://localhost:1337/v1",
    ),
    # === OpenVINO Model Server：本地直连，/v3 兼容 OpenAI ====================
    ProviderSpec(
        name="ovms",
        keywords=("openvino", "ovms"),
        env_key="",
        display_name="OpenVINO Model Server",
        backend="openai_compat",
        is_direct=True,
        is_local=True,
        default_api_base="http://localhost:8000/v3",
    ),
    # === NVIDIA NIM（NVIDIA Inference Microservices）=======================
    # key 通常以 "nvapi-" 开头，base URL 位于 integrate.api.nvidia.com。
    ProviderSpec(
        name="nvidia",
        keywords=("nvidia", "nemotron", "nvapi"),
        env_key="NVIDIA_NIM_API_KEY",
        display_name="NVIDIA NIM",
        backend="openai_compat",
        is_gateway=False,
        detect_by_key_prefix="nvapi-",
        detect_by_base_keyword="nvidia.com",
        default_api_base="https://integrate.api.nvidia.com/v1",
    ),
    # === 辅助型 Provider（不是主聊天 LLM 的首选）===========================
    # Groq：常用于 Whisper 语音转写，也可以直接承担 LLM 调用。
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq",
        backend="openai_compat",
        default_api_base="https://api.groq.com/openai/v1",
    ),
    # AssemblyAI：仅用于语音转写。
    # 它会出现在 Provider 配置里，方便用户管理凭证；
    # 但 WebUI 不会把它放进聊天模型下拉列表。
    ProviderSpec(
        name="assemblyai",
        keywords=("assemblyai",),
        env_key="ASSEMBLYAI_API_KEY",
        display_name="AssemblyAI",
        backend="openai_compat",
        default_api_base="https://api.assemblyai.com/v2",
        is_transcription_only=True,
    ),
    # Qianfan（百度千帆）：OpenAI-compatible API
    ProviderSpec(
        name="qianfan",
        keywords=("qianfan", "ernie"),
        env_key="QIANFAN_API_KEY",
        display_name="Qianfan",
        backend="openai_compat",
        default_api_base="https://qianfan.baidubce.com/v2"
    ),
)


# ---------------------------------------------------------------------------
# 查找辅助函数
# ---------------------------------------------------------------------------


def find_by_name(name: str) -> ProviderSpec | None:
    """按配置字段名查找 ProviderSpec。
    
    实现方法：遍历候选集合并应用过滤条件，返回符合条件的项或最接近的候选。"""
    normalized = to_snake(name.replace("-", "_"))
    for spec in PROVIDERS:
        if spec.name == normalized:
            return spec
    return None
