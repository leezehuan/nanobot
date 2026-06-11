"""Shell 命令执行的沙箱后端。

这个模块的职责很单一：把原始命令包装成“在某种沙箱里运行”的命令字符串。

如果以后要新增沙箱后端，约定是：

- 实现 ``_wrap_<name>(command, workspace, cwd) -> str``
- 再把它注册到下面的 ``_BACKENDS`` 里
"""

import shlex
from pathlib import Path

from nanobot.config.paths import get_media_dir


def _bwrap(command: str, workspace: str, cwd: str) -> str:
    """用 bubblewrap 包装命令，使其在受限沙箱里执行。

    关键思路：

    - 只有 workspace 会被以可读写方式挂进去
    - workspace 的父目录会被一个新的 tmpfs 遮住，避免顺手看到配置目录
    - media 目录只读挂载，让命令可以读取上传附件，但不能随意改
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent", "--setenv", "HOME", str(ws)]
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(ws.parent),        # 遮住配置目录所在父目录
        "--dir", str(ws),                 # 重新创建 workspace 挂载点
        "--bind", str(ws), str(ws),
        "--ro-bind-try", str(media), str(media),  # 只读访问媒体目录
        "--chdir", sandbox_cwd,
        "--", "sh", "-c", command,
    ]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap}


def wrap_command(sandbox: str, command: str, workspace: str, cwd: str) -> str:
    """按名字选择对应沙箱后端来包装命令。"""
    if backend := _BACKENDS.get(sandbox):
        return backend(command, workspace, cwd)
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")
