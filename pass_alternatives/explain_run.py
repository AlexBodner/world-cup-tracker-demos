"""Backward-compat shim — use ``world_cup_projects.explain.pass_alternatives_run``."""

from world_cup_projects.explain.pass_alternatives_run import *  # noqa: F403
from world_cup_projects.explain.pass_alternatives_run import main

if __name__ == "__main__":
    main()
