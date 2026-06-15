"""CLI entry for pass-corridor explain assets (pass line + intercept threat).

Thin wrapper around ``pass_explain_run`` with ``--lane-only --lane-video``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))


def main() -> None:
    if "--lane-only" not in sys.argv:
        sys.argv.append("--lane-only")
    if "--lane-corridor" not in sys.argv:
        sys.argv.append("--lane-corridor")
    if "--lane-video" not in sys.argv and "--explain-video" not in sys.argv:
        sys.argv.append("--lane-video")

    from world_cup_projects.player_stats.pass_explain_run import main as run

    run()


if __name__ == "__main__":
    main()
