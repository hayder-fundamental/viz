"""Run Tracker"""

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

import core

MAX_DOWNLOAD_THREADS = 10


def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dashboard")
    parser.add_argument(
        "name",
        type=str,
        help=(
            "Name of dashboard, must be a subclass of `core.DashboardConfig`."
            " Define your class like MyConfig(core.DashboardConfig, name='my-name')"
            " and it will become available as an argument here."
        ),
        choices=list(core._REGISTRY.keys()),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = core.get_config(args.name)

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=cfg.read_timeout,
        run_filter=cfg.run_filter,
    )
    logger.info("Download run data.")
    # downloader.clear_cache(*runs)
    run_data = downloader.fetch_histories(
        runs,
        max_threads=MAX_DOWNLOAD_THREADS,
    )

    logger.info("Basic checks.")

    # Selecting columns because some could be missing from some runs.
    data = []
    non_empty_runs = []
    for run, df in zip(runs, run_data):
        if df.empty:
            continue
        present = [s for s in cfg.select_metrics if s in df.columns]
        selected = df.set_index("_step")[present]
        data.append(selected)
        non_empty_runs.append(run)

    assert not all(d.empty for d in data), "All dataframes empty"

    logger.info("Normalise indices of DataFrames.")
    data_df = pd.concat(
        {r.name: d for r, d in zip(non_empty_runs, data)}, axis=1
    ).sort_index()
    line_gen = core.LineGenerator(non_empty_runs, data_df)

    for title, kwds in cfg.line_configs().items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
