param(
    [switch]$Dev,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

# 这个 PowerShell 安装脚本和 install.sh 目标一致：
# 让 Windows 用户尽量通过一条命令完成安装、升级和首次向导启动。

$Package = "nanobot-ai"
$MainSource = "https://github.com/HKUDS/nanobot/archive/refs/heads/main.zip"
$InstallTarget = $Package
$InstallSource = "PyPI"

function Write-Info {
    param([string]$Message)
    # 统一普通输出入口，后续如果想切日志样式，只改这里即可。
    Write-Host $Message
}

function Fail {
    param([string]$Message)
    # 统一错误抛出路径，让调用栈保持清晰。
    throw "Error: $Message"
}

function Show-InstallFailureHint {
    # pip 安装失败时，输出一段用户可以直接照着操作的提示。
    [Console]::Error.WriteLine("Error: pip could not install nanobot from $InstallSource.")
    [Console]::Error.WriteLine("If pip mentioned externally-managed-environment, install in a virtual environment or use uv/pipx.")
    [Console]::Error.WriteLine("You can also run manually:")
    [Console]::Error.WriteLine("  $Python -m pip install --upgrade $InstallTarget")
    [Console]::Error.WriteLine("Then start setup with:")
    [Console]::Error.WriteLine("  $Python -m nanobot onboard --wizard")
    throw "pip could not install nanobot from $InstallSource"
}

function Show-Usage {
    # 打印帮助说明，不执行安装。
    Write-Host "Usage: install.ps1 [-Dev|--dev] [-DryRun|--dry-run]"
    Write-Host ""
    Write-Host "By default this installs or upgrades nanobot-ai from PyPI."
    Write-Host "Use --dev to install from the current main branch on GitHub."
    Write-Host "Use --dry-run to print what would happen without installing or starting the wizard."
}

function Test-Python {
    param([string]$Command)
    # 直接调用候选解释器，要求版本至少为 3.11。
    try {
        & $Command -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Find-Python {
    # 查找可用 Python 的优先级：
    # 1. 用户显式设置的 PYTHON
    # 2. 系统 PATH 里的 python / py
    if ($env:PYTHON) {
        if (Get-Command $env:PYTHON -ErrorAction SilentlyContinue) {
            if (Test-Python $env:PYTHON) {
                return $env:PYTHON
            }
            Fail "PYTHON=$env:PYTHON is not Python 3.11 or newer."
        }
        Fail "PYTHON=$env:PYTHON was not found."
    }

    foreach ($Candidate in @("python", "py")) {
        if (Get-Command $Candidate -ErrorAction SilentlyContinue) {
            if (Test-Python $Candidate) {
                return $Candidate
            }
        }
    }

    Fail "Python 3.11 or newer was not found. Install Python first, then rerun this command."
}

foreach ($Arg in $RemainingArgs) {
    switch ($Arg) {
        "--dev" {
            $Dev = $true
        }
        "--dry-run" {
            $DryRun = $true
        }
        "-h" {
            Show-Usage
            return
        }
        "--help" {
            Show-Usage
            return
        }
        default {
            Fail "Unknown option: $Arg"
        }
    }
}

if ($Dev) {
    # Dev 模式改为从 GitHub main 安装，方便测试最新代码。
    $InstallTarget = $MainSource
    $InstallSource = "GitHub main"
}

$Python = Find-Python
$Version = & $Python --version
Write-Info "Using Python: $Version"

try {
    & $Python -m pip --version *> $null
} catch {}

if ($LASTEXITCODE -ne 0) {
    if ($DryRun) {
        Write-Info "Dry run: pip was not found. Install would try: $Python -m ensurepip --upgrade"
    } else {
        # 某些 Python 环境默认不带 pip，这里尝试自动补齐。
        Write-Info "pip was not found for this Python. Trying ensurepip..."
        & $Python -m ensurepip --upgrade *> $null
        if ($LASTEXITCODE -ne 0) {
            Fail "pip is not available. Install pip for $Python, then rerun this command."
        }
    }
}

if ($DryRun) {
    # Dry run 只报告计划，不对本机环境做写入。
    Write-Info "Dry run: would install or upgrade nanobot from $InstallSource."
    Write-Info "Dry run: would run: $Python -m pip install --upgrade $InstallTarget"
    Write-Info "Dry run: if that fails because system site-packages are not writable, would retry: $Python -m pip install --user --upgrade $InstallTarget"
    if ($env:NANOBOT_SKIP_WIZARD -eq "1") {
        Write-Info "Dry run: would skip setup wizard because NANOBOT_SKIP_WIZARD=1."
    } else {
        Write-Info "Dry run: would run: $Python -m nanobot onboard --wizard"
    }
    Write-Info "Dry run: no changes made."
    return
}

Write-Info "Installing or upgrading nanobot from $InstallSource..."
& $Python -m pip install --upgrade $InstallTarget
if ($LASTEXITCODE -ne 0) {
    # 全局安装失败时，退回到用户目录安装，兼容无管理员权限场景。
    Write-Info "Install failed. Retrying as a user install..."
    & $Python -m pip install --user --upgrade $InstallTarget
    if ($LASTEXITCODE -ne 0) {
        Show-InstallFailureHint
    }
}

Write-Info "Installed nanobot:"
& $Python -m nanobot --version
if ($LASTEXITCODE -ne 0) {
    Fail "nanobot was installed, but the command could not be started."
}

if ($env:NANOBOT_SKIP_WIZARD -eq "1") {
    # 自动化或 CI 场景可以跳过首次交互向导。
    Write-Info "Skipping setup wizard because NANOBOT_SKIP_WIZARD=1."
    Write-Info "Run this later: $Python -m nanobot onboard --wizard"
    return
}

Write-Info "Starting setup wizard..."
& $Python -m nanobot onboard --wizard
if ($LASTEXITCODE -ne 0) {
    Fail "Setup wizard did not complete."
}

Write-Info "Done. Try: $Python -m nanobot agent -m `"Hello!`""
