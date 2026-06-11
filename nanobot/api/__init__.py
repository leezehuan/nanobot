"""nanobot 的 OpenAI 兼容 HTTP API 包。

【你可以把它理解成什么】

这个包的作用，是把 nanobot 包装成一个“长得像 OpenAI API 的服务”。
这样外部程序就可以像调用 `/v1/chat/completions` 一样来调用 nanobot，
而不需要直接理解内部的 AgentLoop、消息总线、会话管理等实现细节。

P2 阶段这里本身代码很少，真正的请求解析、流式输出和路由注册逻辑
都在 `nanobot.api.server` 里。
"""
