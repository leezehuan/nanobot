#!/bin/sh

# 这个容器入口脚本做的事情很简单：
# 在真正启动 nanobot 之前，先检查 ~/.nanobot 是否可写。
# 如果目录权限不对，后续配置、会话、缓存都会写失败，所以要尽早报错。

dir="$HOME/.nanobot"
if [ -d "$dir" ] && [ ! -w "$dir" ]; then
    owner_uid=$(stat -c %u "$dir" 2>/dev/null || stat -f %u "$dir" 2>/dev/null)
    cat >&2 <<EOF
Error: $dir is not writable (owned by UID $owner_uid, running as UID $(id -u)).

Fix (pick one):
  Host:   sudo chown -R 1000:1000 ~/.nanobot
  Docker: docker run --user \$(id -u):\$(id -g) ...
  Podman: podman run --userns=keep-id ...
EOF
    exit 1
fi
exec nanobot "$@"
