# -*- coding: utf-8 -*-
"""Tabicl Run Tracker"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

import core


class MetricNames:
    # works
    t_cross_entropy = "training/CrossEntropyLoss"

    eip_acc = (
        "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy"
    )
    eip_acc_32 = "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy with 32 num_context"
    eip_acc_128 = "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy with 128 num_context"
    eip_acc_1024 = "evaluation_Improvement probability over TabPFNv2fixed/score with at  - accuracy 1024 num_context"
    eip_acc_7500 = "evaluation_Improvement probability over TabPFNv2fixed/score with at  - accuracy 7500 num_context"

    eips = [
        eip_acc,
        eip_acc_32,
        eip_acc_128,
        eip_acc_1024,
        eip_acc_7500,
    ]


class select:
    woj_tabicl_run_ids = [
        # Simplified Tabicl V0
        "li9vmts8",
        # SimpleTabICL model ctd @lr=1e-4 + rng fix, smaller LR restart
        "4ij90pn6",
    ]
    tabicl_run_ids = [
        "1pmir581",
        "fguyfgu3",
        "91xwo8vq",
        "bk6hy1u8",
        "fipt4khi",
        "fnemi4d4",
        "4qlafc3f",
        "3kc8angw",
        "pkq7w59y",
        "l3vdkvcc",
        "o8dxdwnj",
        "vzjjudsp",
        "f8ly6enm",
        *woj_tabicl_run_ids,
    ]

    def my_runs(run):
        return run.user.name == "Hayder Elesedy"

    def tabicl_runs(run):
        return run.id in select.tabicl_run_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def running_or_baseline(run):
        return run.state == "running" or run.id in select.woj_tabicl_run_ids

    def wojtek_params(run):
        return "woj params" in run.name.lower() or run.id in select.woj_tabicl_run_ids

    # Configs
    class cfg:
        download_path = "research/training_setup"
        n_samples = 10_000
        window_size = 168  # smoothing
        WOJ_RUN = "Simplified TabICL v0"
        select_run = select.tabicl_runs
        select_metrics = [MetricNames.t_cross_entropy, *MetricNames.eips]

    # -------------------------

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=30,
        run_filter=cfg.select_run,
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
        f"Running: {MetricNames.t_cross_entropy}": dict(
            plot_metric=MetricNames.t_cross_entropy,
            window=1000,
            run_filter=running_or_baseline,
            min_periods=1,
        ),
        f"Running: {MetricNames.eip_acc}": dict(
            plot_metric=MetricNames.eip_acc,
            window=5000,
            run_filter=running_or_baseline,
            min_periods=1,
        ),
        f"Woj Params: {MetricNames.t_cross_entropy}": dict(
            plot_metric=MetricNames.t_cross_entropy,
            window=1000,
            run_filter=wojtek_params,
            min_periods=1,
        ),
        f"Woj Params: {MetricNames.eip_acc}": dict(
            plot_metric=MetricNames.eip_acc,
            window=5000,
            run_filter=wojtek_params,
            min_periods=1,
        ),
    }

    for title, kwds in line_configs.items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
