#!/bin/sh
set -eu

# 这个脚本的目标，是让用户在类 Unix 环境里尽量“一条命令装好 nanobot”。
# 它会负责：
# 1. 找到满足要求的 Python 3.11+
# 2. 确认 pip 可用，不可用时尝试 ensurepip
# 3. 从 PyPI 或 GitHub main 安装/升级 nanobot
# 4. 安装完成后自动启动 onboarding 向导

package="nanobot-ai"
main_source="https://github.com/HKUDS/nanobot/archive/refs/heads/main.zip"
install_target="$package"
install_source="PyPI"
dry_run="0"

info() {
  # 统一普通信息输出，便于后续替换日志形式。
  printf '%s\n' "$*"
}

fail() {
  # 统一错误退出路径，避免每个分支重复写 stderr + exit。
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

install_failure_hint() {
  # 当 pip 安装失败时，给用户一段“下一步该怎么做”的可执行提示。
  printf '%s\n' "Error: pip could not install nanobot from $install_source." >&2
  printf '%s\n' "If pip mentioned externally-managed-environment, install in a virtual environment or use uv/pipx." >&2
  printf '%s\n' "You can also run manually:" >&2
  printf '  %s\n' "$python_bin -m pip install --upgrade $install_target" >&2
  printf '%s\n' "Then start setup with:" >&2
  printf '  %s\n' "$python_bin -m nanobot onboard --wizard" >&2
  exit 1
}

usage() {
  # 打印命令帮助。这里不执行任何安装动作。
  cat <<'EOF'
Usage: install.sh [--dev] [--dry-run]

By default this installs or upgrades nanobot-ai from PyPI.
Use --dev to install from the current main branch on GitHub.
Use --dry-run to print what would happen without installing or starting the wizard.
EOF
}

find_python() {
  # 依次尝试 python3 / python，筛选出版本 >= 3.11 的解释器。
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
      then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dev)
      install_target="$main_source"
      install_source="GitHub main"
      ;;
    --dry-run)
      dry_run="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

python_bin="${PYTHON:-}"

if [ -n "$python_bin" ]; then
  # 如果用户显式指定了 PYTHON，就优先信任这个解释器，但仍然要校验版本。
  command -v "$python_bin" >/dev/null 2>&1 || fail "PYTHON=$python_bin was not found"
  "$python_bin" - <<'PY' >/dev/null 2>&1 || fail "nanobot requires Python 3.11 or newer"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
else
  # 没显式指定时，再自动探测系统里可用的 Python。
  python_bin="$(find_python)" || fail "Python 3.11 or newer was not found. Install Python first, then rerun this command."
fi

info "Using Python: $("$python_bin" --version 2>&1)"

if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
  if [ "$dry_run" = "1" ]; then
    info "Dry run: pip was not found. Install would try: $python_bin -m ensurepip --upgrade"
  else
    info "pip was not found for this Python. Trying ensurepip..."
    "$python_bin" -m ensurepip --upgrade >/dev/null 2>&1 || fail "pip is not available. Install pip for $python_bin, then rerun this command."
  fi
fi

if [ "$dry_run" = "1" ]; then
  # Dry run 只打印计划动作，不真正改环境。
  info "Dry run: would install or upgrade nanobot from $install_source."
  info "Dry run: would run: $python_bin -m pip install --upgrade $install_target"
  info "Dry run: if that fails because system site-packages are not writable, would retry: $python_bin -m pip install --user --upgrade $install_target"
  if [ "${NANOBOT_SKIP_WIZARD:-}" = "1" ]; then
    info "Dry run: would skip setup wizard because NANOBOT_SKIP_WIZARD=1."
  else
    info "Dry run: would run: $python_bin -m nanobot onboard --wizard"
  fi
  info "Dry run: no changes made."
  exit 0
fi

info "Installing or upgrading nanobot from $install_source..."
if ! "$python_bin" -m pip install --upgrade "$install_target"; then
  # 第一次安装失败时，退回到 --user 模式，兼容没有全局写权限的环境。
  info "Install failed. Retrying as a user install..."
  "$python_bin" -m pip install --user --upgrade "$install_target" || install_failure_hint
fi

info "Installed nanobot:"
"$python_bin" -m nanobot --version

if [ "${NANOBOT_SKIP_WIZARD:-}" = "1" ]; then
  # 某些自动化环境只想装包，不想进入交互式向导。
  info "Skipping setup wizard because NANOBOT_SKIP_WIZARD=1."
  info "Run this later: $python_bin -m nanobot onboard --wizard"
  exit 0
fi

info "Starting setup wizard..."
"$python_bin" -m nanobot onboard --wizard

info "Done. Try: $python_bin -m nanobot agent -m \"Hello!\""
