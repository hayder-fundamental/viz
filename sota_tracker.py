"""MS4 SOTA Push Run Tracker"""

import logging

import wandb
import matplotlib.pyplot as plt
import pandas as pd

import core
import constants

# TODO maybe don't need to do select_metrics thing

# https://fundamental.wandb.io/research/training_setup/runs/t53dts72/overview
BASELINE_CLF_RUN_ID = "t53dts72"
# https://fundamental.wandb.io/research/training_setup/runs/0fxigprp/overview
# This one should be working for both clf and reg, I think it's best overall.
BASELINE_REG_RUN_ID = "0fxigprp"


def select_sota_runs(run: wandb.apis.public.Run) -> bool:
    my_sota_run = "Hayder::MS4-SOTA" in run.tags and run.user.name == "Hayder Elesedy"
    baseline_run = run.id in [BASELINE_CLF_RUN_ID, BASELINE_REG_RUN_ID]
    return my_sota_run or baseline_run


def clf_runs(run: wandb.apis.public.Run) -> bool:
    return run.name.startswith("MS4 SOTA") or run.id == BASELINE_CLF_RUN_ID


def reg_runs(run: wandb.apis.public.Run) -> bool:
    exclude_ids = [
        # https://fundamental.wandb.io/research/training_setup/runs/vm3uynfk/overview
        # This was the run with min_values_std not set, so doesn't learn.
        "vm3uynfk",
    ]
    yes = "regression" in run.name or run.id == BASELINE_REG_RUN_ID
    no = run.id in exclude_ids
    return yes and not no


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configs
    class cfg:
        download_path = "research/training_setup"
        n_samples = 1_000_000
        select_metrics = [
            constants.MetricNames.t_cross_entropy,
            constants.MetricNames.t_huber,
            constants.MetricNames.e_huber,
            *constants.MetricNames.eips,
        ]

    # -------------------------

    logger.info("Login.")
    downloader = core.HistoryDownloader()
    logger.info("Download runs.")
    runs = downloader.fetch_runs(
        path=cfg.download_path, timeout=30, run_filter=select_sota_runs
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

    # -------------------------

    logger.info("Normalise indices of DataFrames.")
    data_df = pd.concat({r.name: d for r, d in zip(runs, data)}, axis=1).sort_index()
    line_gen = core.LineGenerator(runs, data_df)

    line_configs = {
        f"Regression: {constants.MetricNames.t_huber}": dict(
            plot_metric=constants.MetricNames.t_huber,
            window=1000,
            run_filter=reg_runs,
            min_periods=1,
        ),
        f"Regression: {constants.MetricNames.e_huber}": dict(
            plot_metric=constants.MetricNames.e_huber,
            window=5000,
            run_filter=reg_runs,
            min_periods=1,
        ),
        f"Classification: {constants.MetricNames.t_cross_entropy}": dict(
            plot_metric=constants.MetricNames.t_cross_entropy,
            window=1000,
            run_filter=clf_runs,
            min_periods=1,
        ),
        f"Classification: {constants.MetricNames.eip_acc}": dict(
            plot_metric=constants.MetricNames.eip_acc,
            window=5000,
            run_filter=clf_runs,
            min_periods=1,
        ),
    }

    for title, kwds in line_configs.items():
        lines = line_gen(**kwds)
        core.plot_lines(lines, title=title)

    plt.show()
