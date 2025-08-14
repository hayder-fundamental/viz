"""Run Tracker"""

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

import core
import configs


def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dashboard")
    parser.add_argument(
        "name",
        type=str,
        help="Name of dashboard, must be configured in `configs.py`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = configs.get_config(args.name)

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=cfg.read_timeout,
        run_filter=cfg.run_filter,
    )
    logger.info("Download run data.")
    run_data = downloader.fetch_history(runs, n_samples=cfg.n_samples)

    logger.info("Basic checks.")

    # Selecting columns because some could be missing from some runs.
    data = []
    for df in run_data:
        present = [s for s in cfg.select_metrics if s in df.columns]
        selected = df.set_index("_step")[present]
        data.append(selected)

    assert not all(d.empty for d in data), "All dataframes empty"

    logger.info("Normalise indices of DataFrames.")
    data_df = pd.concat({r.name: d for r, d in zip(runs, data)}, axis=1).sort_index()
    line_gen = core.LineGenerator(runs, data_df)

    for title, kwds in cfg.line_configs.items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
