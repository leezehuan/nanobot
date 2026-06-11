"""Git-backed version control for memory files, using dulwich.

【中文名称】Git 版本存储

【功能说明】
以 Git 为后端，为 nanobot 的记忆文件（SOUL.md、USER.md、MEMORY.md）提供：
- 自动提交（auto_commit）：每次记忆变更后自动落一个 commit
- 历史浏览（log）：查看最近 N 次变更记录
- 行级追溯（line_ages）：通过 blame 查看每行代码的最近修改时间
- diff 对比（diff_commits）：查看两次提交之间的差异
- 版本还原（revert）：回滚到指定 commit 的父版本

【为什么需要它】
agent 的记忆文件是长期积累的，如果某次自动编辑写坏了，用户可以通过
git 历史回滚到健康版本。它本质上是一个"记忆安全保障"。

【关键设计决策】
- 使用 dulwich（纯 Python Git 实现），无需系统安装 git 命令
- .gitignore 采用白名单策略：先忽略一切，再逐项放行 tracked_files
- 首次 init 时会自动 touch 缺失的 tracked_files，确保初始 commit 有内容
- 如果 workspace 本身已经在另一个 git 仓库内，则跳过嵌套仓库初始化

【存储结构示例】
workspace/
  .git/                  # dulwich 初始化的 git 仓库
  .gitignore             # 白名单策略：忽略全部，只放行 tracked_files
  SOUL.md                # (tracked) agent 人格定义
  USER.md                # (tracked) 用户偏好
  memory/
    MEMORY.md            # (tracked) 长期记忆
  other_project_files/   # (NOT tracked) 不在白名单内，不会被提交
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class CommitInfo:
    """一次 Git 提交的摘要信息。

    【字段说明】
    - sha: 短哈希值（前 8 位），用于展示和用户输入查找
    - message: 提交信息（例如 "Dream consolidation" 或 "revert: undo abc12345"）
    - timestamp: 人类可读的时间字符串（例如 "2026-06-11 14:30"）
    """

    sha: str  # Short SHA (8 chars)
    message: str
    timestamp: str  # Formatted datetime

    def format(self, diff: str = "") -> str:
        """格式化当前 commit 为 Markdown 展示文本。

        【参数说明】
        - diff: 可选的 diff 文本（由 diff_commits 生成）

        【返回值】
        - 带 diff 时：三部分 —— 标题头、SHA+时间、```diff``` 代码块
        - 不带 diff 时：仅标题头 + SHA+时间 + "(no file changes)"

        【示例输出（有 diff）】
        ## Dream consolidation
        `abc12345` — 2026-06-11 14:30
        ```diff
        - old content
        + new content
        ```

        【示例输出（无 diff）】
        ## Dream consolidation
        `abc12345` — 2026-06-11 14:30
        (no file changes)
        """
        header = f"## {self.message.splitlines()[0]}\n`{self.sha}` — {self.timestamp}\n"
        if diff:
            return f"{header}\n```diff\n{diff}\n```"
        return f"{header}\n(no file changes)"


@dataclass
class LineAge:
    """某一代码行距最后一次修改的天数。

    【字段说明】
    - age_days: 距今多少天（例如 0 表示今天刚改过，365 表示一年未动）

    【使用场景】
    - 记忆系统通过行级年龄判断哪些内容已经"冻结"、哪些还是"热的"
    - Dream consolidation 时根据行龄决定是否触发记忆合并
    """

    age_days: int  # days since last modification


def _compute_line_ages(annotated) -> list[LineAge]:
    """把 dulwich annotate 的结果转换成 LineAge 列表。

    【中文名称】计算行年龄

    【输入】
    - annotated: dulwich porcelain.annotate() 的返回值，格式为
      [((commit, tree_entry), line_bytes), ...]

    【处理逻辑】
    对每一行：
    1. 从 commit 对象的 commit_time 字段读取修改时间戳
    2. 计算 now - commit_time 的天数差
    3. 包装成 LineAge(age_days=N)

    【返回值】
    与输入行数等长的 LineAge 列表，按原始行顺序排列。
    """
    now = datetime.now(tz=timezone.utc).date()
    ages: list[LineAge] = []
    for (commit, _tree_entry), _line_bytes in annotated:
        dt = datetime.fromtimestamp(commit.commit_time, tz=timezone.utc).date()
        ages.append(LineAge(age_days=(now - dt).days))
    return ages


class GitStore:
    """Git-backed version control for memory files.

    【中文名称】Git 版本存储（主类）

    【使用方式】
    ```python
    gs = GitStore(workspace, tracked_files=["SOUL.md", "USER.md", "memory/MEMORY.md"])
    gs.init()                           # 初始化仓库（仅首次）
    sha = gs.auto_commit("edit SOUL")   # 日常自动提交
    history = gs.log(max_entries=20)    # 查看历史
    gs.revert("abc12345")               # 回滚某次提交
    ```

    【关键方法一览】
    初始化: init()
    自动提交: auto_commit(message) -> sha | None
    历史浏览: log(max_entries) -> [CommitInfo]
    行级追溯: line_ages(file_path) -> [LineAge]
    Diff: diff_commits(sha1, sha2) -> str
    版本还原: revert(commit_sha) -> new_sha | None
    """

    def __init__(self, workspace: Path, tracked_files: list[str]):
        """初始化 GitStore。

        【参数说明】
        - workspace: 工作区根目录，.git 仓库会创建在此目录下
        - tracked_files: 需要 Git 跟踪的文件清单（相对于 workspace 的相对路径）
          例如 ["SOUL.md", "USER.md", "memory/MEMORY.md"]
        """
        self._workspace = workspace
        self._tracked_files = tracked_files

    def is_initialized(self) -> bool:
        """检查 .git 目录是否已存在。"""
        return (self._workspace / ".git").is_dir()

    # -- init ------------------------------------------------------------------

    def init(self) -> bool:
        """初始化 Git 仓库。

        【中文名称】初始化 Git 仓库

        【完整流程（6 步）】

        Step 1: 检查是否已初始化 → 已存在则直接返回 False

        Step 2: 检查是否被嵌套在另一个 Git 仓库中
           - 向上遍历父目录，找到 .git 就视为"已在仓库内"
           - 支持 .git 文件和 .git 目录（兼容 worktree/submodule）
           - 嵌套场景跳过初始化，并打 warning 日志

        Step 3: dulwich.porcelain.init() 创建 .git

        Step 4: 生成 .gitignore
           - 采用白名单策略：第一行 "/*" 忽略全部
           - 然后逐项用 "!路径/" 放行父目录
           - 再逐项用 "!路径" 放行 tracked file
           - 最后放行 "!.gitignore" 自身
           - 如果 .gitignore 已存在，则合并而非覆盖

        Step 5: 确保 tracked_files 全部存在
           - 缺失的文件会创建空文件（touch）
           - 这样首次 commit 有内容可提交

        Step 6: 执行初始提交
           - git add .gitignore + 所有 tracked_files
           - git commit 消息为 "init: nanobot memory store"

        【返回值】
        - True: 新仓库创建成功
        - False: 仓库已存在 / 在嵌套仓库中 / 初始化失败
        """
        if self.is_initialized():
            return False

        if self._is_inside_git_repo():
            logger.warning(
                "Workspace {} is already inside a git repo; "
                "skipping nested repo initialization",
                self._workspace,
            )
            return False

        try:
            from dulwich import porcelain

            porcelain.init(str(self._workspace))

            # Write .gitignore (merge with existing if present)
            gitignore = self._workspace / ".gitignore"
            dream_entries = self._build_gitignore()
            if gitignore.exists():
                existing = gitignore.read_text(encoding="utf-8")
                existing_lines = set(existing.splitlines())
                new_lines = [
                    line
                    for line in dream_entries.splitlines()
                    if line not in existing_lines
                ]
                if new_lines:
                    merged = existing.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
                    gitignore.write_text(merged, encoding="utf-8")
            else:
                gitignore.write_text(dream_entries, encoding="utf-8")

            # 确保 tracked_files 作为空文件存在，这样首次 commit 有内容可提交
            for rel in self._tracked_files:
                p = self._workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text("", encoding="utf-8")

            # 初始提交
            porcelain.add(str(self._workspace), paths=[".gitignore"] + self._tracked_files)
            porcelain.commit(
                str(self._workspace),
                message=b"init: nanobot memory store",
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            logger.info("Git store initialized at {}", self._workspace)
            return True
        except Exception:
            logger.exception("Git store init failed for {}", self._workspace)
            return False

    # -- daily operations ------------------------------------------------------

    def auto_commit(self, message: str) -> str | None:
        """自动提交：stage tracked files 并在有变更时 commit。

        【中文名称】自动提交

        【功能说明】
        这是 GitStore 中最常用的操作。每次记忆文件变更后（Dream consolidation、
        手动编辑等），由调用方自动触发一次提交，保证历史可以追溯。

        【完整流程】
        1. 检查仓库是否初始化（未初始化跳过）
        2. git status 查看是否有变更
        3. 没有变更 → 返回 None（不创建空提交）
        4. 有变更 → git add 所有 tracked_files
        5. git commit 写入 "nanobot <nanobot@dream>" 身份
        6. 返回短 SHA（hex[:8]）

        【参数说明】
        - message: 提交信息，例如 "Dream consolidation" 或 "edit SOUL.md"

        【返回值】
        - str: 新提交的短 SHA（8 位十六进制）
        - None: 无变更可提交 / 仓库未初始化 / 执行异常
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich import porcelain

            # .gitignore 采用白名单策略，所以任何变更都来自 tracked_files
            st = porcelain.status(str(self._workspace))
            if not st.unstaged and not any(st.staged.values()):
                return None

            msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
            porcelain.add(str(self._workspace), paths=self._tracked_files)
            sha_bytes = porcelain.commit(
                str(self._workspace),
                message=msg_bytes,
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            if sha_bytes is None:
                return None
            sha = sha_bytes.hex()[:8]
            logger.debug("Git auto-commit: {} ({})", sha, message)
            return sha
        except Exception:
            logger.exception("Git auto-commit failed: {}", message)
            return None

    # -- internal helpers ------------------------------------------------------

    def _resolve_sha(self, short_sha: str) -> bytes | None:
        """把短 SHA（如 "abc12345"）解析为完整的 40 位 SHA bytes。

        【中文名称】解析短 SHA

        【查找策略】
        从 HEAD 开始沿 parent 链向下 walk，直到找到 sha.hex() 以 short_sha 开头的 commit。
        这是一个 O(N) 的线性查找，N 受限于仓库 commit 数量。

        【参数说明】
        - short_sha: 至少 1 位的短 SHA 前缀（通常来自用户输入或前端展示）

        【返回值】
        - bytes: 完整的 40 位 SHA
        - None: 短 SHA 没匹配到任何 commit
        """
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                try:
                    sha = repo.refs[b"HEAD"]
                except KeyError:
                    return None

                while sha:
                    if sha.hex().startswith(short_sha):
                        return sha
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    sha = commit.parents[0] if commit.parents else None
            return None
        except Exception:
            return None

    def _is_inside_git_repo(self) -> bool:
        """检查 workspace 是否已经嵌套在另一个 Git 仓库中。

        【中文名称】检查嵌套 Git 仓库

        【检查逻辑】
        从 self._workspace 向上遍历到文件系统根目录，
        如果任何父目录包含 .git（文件或目录），则认为已处于 Git 仓库内。

        【为什么支持 .git 文件】
        Git worktree 和 submodule 使用 .git 文件（内容为 "gitdir: /path/to/.git"），
        而非 .git 目录。两者都应被视为"已在仓库内"的标志。
        """
        current = self._workspace.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _build_gitignore(self) -> str:
        """生成白名单策略的 .gitignore 内容。

        【中文名称】生成 .gitignore

        【策略说明】
        采用"先全部忽略，再逐项放行"的白名单策略：
        - 第一行: "/*" — 忽略 workspace 下所有内容
        - 中间行: "!父目录/" — 放行 tracked_files 所在的目录
        - 核心行: "!tracked_file" — 逐项放行被跟踪的文件
        - 最后: "!.gitignore" — 放行 .gitignore 自身（否则无法被提交）

        【为什么用白名单而非黑名单】
        如果采用黑名单（.gitignore 只写忽略项），用户新增文件会被自动纳入版本控制，
        可能导致敏感数据意外提交。白名单确保"只有明确声明的文件才被跟踪"。

        【示例输出（tracked_files = ["SOUL.md", "memory/MEMORY.md"]）】
        /*
        !memory/
        !.gitignore
        !SOUL.md
        !memory/MEMORY.md
        """
        dirs: set[str] = set()
        for f in self._tracked_files:
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)
        lines = ["/*"]
        for d in sorted(dirs):
            lines.append(f"!{d}/")
        for f in self._tracked_files:
            lines.append(f"!{f}")
        lines.append("!.gitignore")
        return "\n".join(lines) + "\n"

    # -- query -----------------------------------------------------------------

    def log(self, max_entries: int = 20) -> list[CommitInfo]:
        """返回最近 N 条提交记录摘要。

        【中文名称】查看提交历史

        【实现方式】
        从 HEAD 开始沿 parent 链 walk，最多返回 max_entries 条记录。

        【参数说明】
        - max_entries: 最多返回多少条记录（默认 20）

        【返回值】
        - [CommitInfo, ...]: 最近提交列表，按时间降序（最新在前）
        - []: 仓库未初始化 / 无 HEAD / 读取异常
        """
        if not self.is_initialized():
            return []

        try:
            from dulwich.repo import Repo

            entries: list[CommitInfo] = []
            with Repo(str(self._workspace)) as repo:
                try:
                    head = repo.refs[b"HEAD"]
                except KeyError:
                    return []

                sha = head
                while sha and len(entries) < max_entries:
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(commit.commit_time),
                    )
                    msg = commit.message.decode("utf-8", errors="replace").strip()
                    entries.append(CommitInfo(
                        sha=sha.hex()[:8],
                        message=msg,
                        timestamp=ts,
                    ))
                    sha = commit.parents[0] if commit.parents else None

            return entries
        except Exception:
            logger.exception("Git log failed")
            return []

    def line_ages(self, file_path: str) -> list[LineAge]:
        """通过 git blame 计算指定文件每行代码的修改年龄。

        【中文名称】行年龄计算

        【功能说明】
        使用 git blame 追溯 tracked file 每一行的最后修改时间，
        并将时间差转换为"距今多少天"。

        【使用场景】
        记忆系统（Dream consolidation）通过行龄判断：
        - 年龄较大的行 → "冻结状态"，可以考虑压缩/归档
        - 年龄较小的行 → "活跃状态"，保留详情

        【参数说明】
        - file_path: 相对于 workspace 的文件路径（如 "memory/MEMORY.md"）

        【返回值】
        - [LineAge, ...]: 与文件行数等长的年龄列表
        - []: 仓库未初始化 / 文件不存在或为空 / annotate 失败
        """

        if not self.is_initialized():
            return []

        target = self._workspace / file_path
        if not target.exists() or target.stat().st_size == 0:
            return []

        try:
            from dulwich import porcelain

            annotated = porcelain.annotate(str(self._workspace), file_path)
        except Exception:
            logger.exception("Git line_ages annotate failed for {}", file_path)
            return []

        if not annotated:
            return []

        return _compute_line_ages(annotated)

    def diff_commits(self, sha1: str, sha2: str) -> str:
        """对比两次提交之间的 diff。

        【中文名称】提交对比

        【参数说明】
        - sha1: 旧 commit 的短 SHA（作为 diff 基准）
        - sha2: 新 commit 的短 SHA（与旧版本对比）

        【返回值】
        - str: 标准 diff 文本
        - "": 仓库未初始化 / SHA 无法解析 / 读取异常

        【使用场景】
        - 用户想查看某次提交具体改了哪些行
        - show_commit_diff() 内部调用，展示某次提交相对于父提交的变更
        """
        if not self.is_initialized():
            return ""

        try:
            from dulwich import porcelain

            full1 = self._resolve_sha(sha1)
            full2 = self._resolve_sha(sha2)
            if not full1 or not full2:
                return ""

            out = io.BytesIO()
            porcelain.diff(
                str(self._workspace),
                commit=full1,
                commit2=full2,
                outstream=out,
            )
            return out.getvalue().decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Git diff_commits failed")
            return ""

    def find_commit(self, short_sha: str, max_entries: int = 20) -> CommitInfo | None:
        """按短 SHA 前缀查找提交。

        【中文名称】按 SHA 查找提交

        【查找方式】
        调用 log() 遍历最近 N 条 commit，逐一对比 sha 是否以 short_sha 开头。

        【参数说明】
        - short_sha: 短 SHA 前缀（如 "abc" 可匹配 abc12345）
        - max_entries: 查找范围（默认最近 20 条）

        【返回值】
        - CommitInfo: 匹配成功
        - None: 未找到
        """
        for c in self.log(max_entries=max_entries):
            if c.sha.startswith(short_sha):
                return c
        return None

    def show_commit_diff(self, short_sha: str, max_entries: int = 20) -> tuple[CommitInfo, str] | None:
        """查找提交并返回它相对于父提交的 diff。

        【中文名称】显示提交变更

        【功能说明】
        这一步实际上做了两件事：
        1. 在历史中找到目标 commit
        2. 对该 commit 与它的父 commit 做 diff

        【参数说明】
        - short_sha: 目标提交的短 SHA
        - max_entries: 查找范围

        【返回值】
        - (CommitInfo, diff_str): 成功找到并有父提交，diff_str 非空
        - (CommitInfo, ""): 成功找到但无父提交（即这是根 commit）
        - None: 未找到目标 commit
        """
        commits = self.log(max_entries=max_entries)
        for i, c in enumerate(commits):
            if c.sha.startswith(short_sha):
                if i + 1 < len(commits):
                    diff = self.diff_commits(commits[i + 1].sha, c.sha)
                else:
                    diff = ""
                return c, diff
        return None

    # -- restore ---------------------------------------------------------------

    def revert(self, commit: str) -> str | None:
        """回滚（撤销）指定 commit 引入的变更。

        【中文名称】版本还原

        【功能说明】
        将 tracked_files 恢复到目标 commit 的**父 commit** 状态，
        然后创建一条新的 revert commit 记录这次回滚。

        【完整流程（5 步）】

        Step 1: 从短 SHA 解析完整 SHA → 解析失败返回 None

        Step 2: 读取 commit 对象 → 不是 commit 类型返回 None

        Step 3: 检查是否有父 commit
           - 没有父 commit（即根 commit）→ 无法 revert，返回 None

        Step 4: 遍历 tracked_files，读取父 commit tree 中对应 blob 的内容
           - 把内容写回 workspace 对应文件
           - 记录成功恢复的文件列表

        Step 5: 自动提交 revert 记录
           - commit 消息格式: "revert: undo <短SHA>"
           - 返回新 commit 的短 SHA

        【参数说明】
        - commit: 要撤销的那个 commit 的短 SHA

        【返回值】
        - str: revert 操作创建的新 commit 的短 SHA
        - None: commit 不存在 / 无可 revert 的父提交 / 读取或写入失败

        【注意】
        这个 revert 不是 `git revert`（反向应用 diff），而是直接恢复为父版本文件内容。
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich.repo import Repo

            full_sha = self._resolve_sha(commit)
            if not full_sha:
                logger.warning("Git revert: SHA not found: {}", commit)
                return None

            with Repo(str(self._workspace)) as repo:
                commit_obj = repo[full_sha]
                if commit_obj.type_name != b"commit":
                    return None

                if not commit_obj.parents:
                    logger.warning("Git revert: cannot revert root commit {}", commit)
                    return None

                # 使用父 commit 的 tree，也就是"撤销"目标 commit 的变更
                parent_obj = repo[commit_obj.parents[0]]
                tree = repo[parent_obj.tree]

                restored: list[str] = []
                for filepath in self._tracked_files:
                    content = self._read_blob_from_tree(repo, tree, filepath)
                    if content is not None:
                        dest = self._workspace / filepath
                        dest.write_text(content, encoding="utf-8")
                        restored.append(filepath)

            if not restored:
                return None

            # 自动提交 revert 记录
            msg = f"revert: undo {commit}"
            return self.auto_commit(msg)
        except Exception:
            logger.exception("Git revert failed for {}", commit)
            return None

    @staticmethod
    def _read_blob_from_tree(repo, tree, filepath: str) -> str | None:
        """从 tree 对象中读取指定文件路径的 blob 内容。

        【中文名称】从 Git Tree 读取文件

        【实现方式】
        按路径段逐级 walk：
        1. 从 tree 中查找第一段路径的 entry
        2. 如果路径有多段，解析 entry 为 subtree 并递归
        3. 最后一层找到 blob 后读取并解码为 UTF-8

        【参数说明】
        - repo: dulwich Repo 对象
        - tree: dulwich Tree 对象（当前要查找的子目录）
        - filepath: 相对于 repo 根目录的文件路径，如 "memory/MEMORY.md"

        【返回值】
        - str: 文件的文本内容
        - None: 路径不存在 / 中间节点不是 tree / 最终节点不是 blob
        """
        parts = Path(filepath).parts
        current = tree
        for part in parts:
            try:
                entry = current[part.encode()]
            except KeyError:
                return None
            obj = repo[entry[1]]
            if obj.type_name == b"blob":
                return obj.data.decode("utf-8", errors="replace")
            if obj.type_name == b"tree":
                current = obj
            else:
                return None
        return None
