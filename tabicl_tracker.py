"""Tabicl Run Tracker"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

import core
import constants


def select_tabicl_runs(run):
    return run.id in constants.RunIDs.tabicl_run_ids


def running_or_baseline(run):
    return run.state == "running" or run.id in constants.RunIDs.woj_tabicl_run_ids


def wojtek_params(run):
    return (
        "woj params" in run.name.lower()
        or run.id in constants.RunIDs.woj_tabicl_run_ids
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configs
    class cfg:
        download_path = "research/training_setup"
        n_samples = 10_000
        select_metrics = [constants.MetricNames.t_cross_entropy, *constants.MetricNames.eips]

    # -------------------------

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=30,
        run_filter=select_tabicl_runs,
    )
    logger.info("Download run data.")
    run_data = downloader.fetch_history(runs, n_samples=cfg.n_samples)

    logger.info("Basic checks.")
    # -------------------------
    # Selecting columns because some could be missing from some runs.
    data = []
    for run, df in zip(runs, run_data):
        present = [s for s in cfg.select_metrics if s in df.columns]
        if missing := set(cfg.select_metrics) - set(df.columns):
            print(f"Missing columns for {run.name}")
            print(missing)
        selected = df.set_index("_step")[present]
        data.append(selected)

        entity = "research"
        project = "training_setup"
    assert not all(d.empty for d in data), "All dataframes empty"

    # Checking all nan columns
    # -------------------------
    nan_col_runs = []
    for r, d in zip(runs, data):
        nan_cols = d.columns[d.isna().all(axis=0)]
        if not nan_cols.empty:
            nan_col_runs.append(r.name, nan_cols.tolist())

    if nan_col_runs:
        print("Runs with all NaN cloumns:")
    for name, cols in nan_col_runs:
        print(name, cols)

    # -------------------------

    logger.info("Normalise indices of DataFrames.")
    data_df = pd.concat({r.name: d for r, d in zip(runs, data)}, axis=1).sort_index()
    line_gen = core.LineGenerator(runs, data_df)

    line_configs = {
        f"Running: {constants.MetricNames.t_cross_entropy}": dict(
            plot_metric=constants.MetricNames.t_cross_entropy,
            window=1000,
            run_filter=running_or_baseline,
            min_periods=1,
        ),
        f"Running: {constants.MetricNames.eip_acc}": dict(
            plot_metric=constants.MetricNames.eip_acc,
            window=5000,
            run_filter=running_or_baseline,
            min_periods=1,
        ),
        f"Woj Params: {constants.MetricNames.t_cross_entropy}": dict(
            plot_metric=constants.MetricNames.t_cross_entropy,
            window=1000,
            run_filter=wojtek_params,
            min_periods=1,
        ),
        f"Woj Params: {constants.MetricNames.eip_acc}": dict(
            plot_metric=constants.MetricNames.eip_acc,
            window=5000,
            run_filter=wojtek_params,
            min_periods=1,
        ),
    }

    for title, kwds in line_configs.items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
