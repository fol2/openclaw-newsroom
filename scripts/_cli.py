"""Zero-arg CLI wrappers for console_scripts entry points.

Each script's ``main(argv)`` accepts ``sys.argv[1:]``.  The wrappers here
adapt that signature to the zero-arg callable that setuptools/hatchling
console_scripts expects.
"""
from __future__ import annotations

import sys


def gdelt_pool_update() -> None:
    from scripts.gdelt_pool_update import main
    raise SystemExit(main(sys.argv[1:]))


def news_pool_status() -> None:
    from scripts.news_pool_status import main
    raise SystemExit(main(sys.argv[1:]))


def news_pool_update() -> None:
    from scripts.news_pool_update import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_daily_inputs() -> None:
    from scripts.newsroom_daily_inputs import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_hourly_inputs() -> None:
    from scripts.newsroom_hourly_inputs import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_runner() -> None:
    from scripts.newsroom_runner import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_write_run_job() -> None:
    from scripts.newsroom_write_run_job import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_clustering_decisions() -> None:
    from scripts.newsroom_clustering_decisions import main
    raise SystemExit(main(sys.argv[1:]))


def newsroom_decision_log_inspector() -> None:
    from scripts.newsroom_clustering_decisions import main
    raise SystemExit(main(sys.argv[1:]))


def build_clustering_eval_dataset() -> None:
    from scripts.build_clustering_eval_dataset import main
    raise SystemExit(main(sys.argv[1:]))


def replay_clustering_eval_dataset() -> None:
    from scripts.replay_clustering_eval_dataset import main
    raise SystemExit(main(sys.argv[1:]))


def rss_pool_update() -> None:
    from scripts.rss_pool_update import main
    raise SystemExit(main(sys.argv[1:]))
