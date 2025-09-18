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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear cache and re-download. Overrides --cache-only.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Don't download data, only read from the cache.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help="Number of rows to download per wandb query in `run.scan_history`",
        default=10_000,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to make charts.",
    )
    parser.add_argument(
        "--log-level",
        type=lambda s: s.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level of the root logger (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    cfg = core.get_config(args.name)

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=cfg.read_timeout,
        query_filter=cfg.query_filter,
        run_filter=cfg.run_filter,
    )

    import pdb; pdb.set_trace()
    if args.refresh:
        print("Clearing Cache... Need to fix this!")
        downloader.clear_cache(runs)
        logger.info("Downloading run data.")
        run_data = downloader.fetch_histories(
            runs,
            max_threads=MAX_DOWNLOAD_THREADS,
            page_size=args.page_size,
        )
    elif args.cache_only:
        logger.info("Reading run data from cache.")
        run_data = [downloader.read_cache(run) for run in runs]
    else:
        raise NotImplementedError("Appending to cache not implemented.")

    if not args.plot:
        import sys

        sys.exit(0)
    logger.info("Basic checks.")

    # Selecting columns because some could be missing from some runs.
    data = []
    non_empty_runs = []
    for run, df in zip(runs, run_data):
        if df.empty:
            continue
        present = [s for s in cfg.select_metrics if s in df.columns]
        selected = df.set_index("_step")[present]
        duplicated_steps = selected.index.duplicated()
        if n_dup_steps := duplicated_steps.sum():
            print(f"{run.name}: {n_dup_steps} duplicated step indices.")
            # print("Dropping duplicated steps")
            # selected.loc[~selected.index.duplicated(keep="first"), :]

        data.append(selected)
        non_empty_runs.append(run)

    assert not all(d.empty for d in data), "All dataframes empty"

    logger.info("Normalise indices of DataFrames.")
    data_df = pd.concat(
        {r.name: d.drop_duplicates() for r, d in zip(non_empty_runs, data)}, axis=1
    ).sort_index()
    line_gen = core.LineGenerator(non_empty_runs, data_df)

    for title, kwds in cfg.line_configs().items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
