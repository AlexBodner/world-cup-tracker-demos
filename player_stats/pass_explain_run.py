"""Backward-compat shim — use ``world_cup_projects.explain.pass_explain_run``."""

from world_cup_projects.explain.pass_explain_run import *  # noqa: F403
from world_cup_projects.explain.pass_explain_run import main

if __name__ == "__main__":
    main()
