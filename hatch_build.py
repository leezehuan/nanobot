"""把 WebUI 构建产物打包进 Python wheel 的 Hatch 构建钩子。

【中文名称】WebUI 构建钩子

当执行 `python -m build` 这类基于 Hatch 的构建命令时，
这个文件会自动决定是否需要先构建前端，再把产物放进
`nanobot/web/dist`。

它存在的意义是：

1. 发布 wheel/sdist 时，确保前端静态文件是现成可用的；
2. 避免开发者每次打包都手工记住 `cd webui && bun run build`；
3. 对可编辑安装、预构建产物、显式跳过构建等场景做智能判断。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class WebUIBuildHook(BuildHookInterface):
    """Hatch 在打包阶段调用的 WebUI 构建钩子实现。"""
    PLUGIN_NAME = "webui-build"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: D401
        """在构建开始前决定是否执行 WebUI 安装与打包。

        这里的控制逻辑比较关键，因为前端构建通常比 Python 打包慢得多。
        所以它会优先判断“能不能跳过”，只有确实需要时才执行安装和 build。
        """
        root = Path(self.root)
        webui_dir = root / "webui"
        package_json = webui_dir / "package.json"
        dist_dir = root / "nanobot" / "web" / "dist"
        index_html = dist_dir / "index.html"

        # `pip install -e .` 面向本地 Python 开发，通常不需要额外打包静态前端。
        # WebUI 开发者会直接跑 `bun run dev`，所以这里主动跳过耗时构建。
        if self.target_name == "wheel" and version == "editable":
            self.app.display_info(
                "[webui-build] skipped for editable install "
                "(use `cd webui && bun run build` to bundle webui manually)"
            )
            return

        if os.environ.get("NANOBOT_SKIP_WEBUI_BUILD") == "1":
            self.app.display_info("[webui-build] skipped via NANOBOT_SKIP_WEBUI_BUILD=1")
            return

        if not package_json.is_file():
            self.app.display_info(
                "[webui-build] no webui/ source tree, assuming prebuilt nanobot/web/dist/"
            )
            return

        force = os.environ.get("NANOBOT_FORCE_WEBUI_BUILD") == "1"
        if index_html.is_file() and not force:
            self.app.display_info(
                f"[webui-build] reusing existing build at {dist_dir} "
                "(set NANOBOT_FORCE_WEBUI_BUILD=1 to rebuild)"
            )
            return

        runner = self._pick_runner()
        if runner is None:
            raise RuntimeError(
                "[webui-build] neither `bun` nor `npm` is available on PATH; "
                "install one or set NANOBOT_SKIP_WEBUI_BUILD=1 to bypass."
            )

        self.app.display_info(f"[webui-build] using {runner} to build webui")
        self._run([runner, "install"], cwd=webui_dir)
        self._run([runner, "run", "build"], cwd=webui_dir)

        if not index_html.is_file():
            raise RuntimeError(
                f"[webui-build] build finished but {index_html} is missing; "
                "check webui/vite.config.ts outDir."
            )
        self.app.display_info(f"[webui-build] webui ready at {dist_dir}")

    @staticmethod
    def _pick_runner() -> str | None:
        """优先选择 bun，没有时退回 npm。"""
        for candidate in ("bun", "npm"):
            if shutil.which(candidate):
                return candidate
        return None

    def _run(self, cmd: list[str], *, cwd: Path) -> None:
        """执行一条构建命令，并把失败包装成更清晰的构建错误。"""
        self.app.display_info(f"[webui-build] $ {' '.join(cmd)} (cwd={cwd})")
        try:
            subprocess.run(cmd, cwd=cwd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"[webui-build] command failed ({exc.returncode}): {' '.join(cmd)}"
            ) from exc
