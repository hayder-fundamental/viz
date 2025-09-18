import constants

import wandb

import core


class TabICLEval(core.DownloadConfig, name="tabicl-eval"):
    download_path = "research/evaluating_our_models"
    read_timeout = 120
    run_ids = [
        # 19999
        "9xwqr7yf",
        "edd713aq",  ## martas, not finished
        "eq8eovmk",
        "eak0sdnb",
        "swgqhu8z",
        # 89999
        # "qcb1qlru",
        # "2rtjgeb0",
        # "uuv8g3q4",
    ]

    def run_filter(self, run):
        return run.id in self.run_ids


class TabICLReproduction(core.DownloadConfig, name="tabicl-reproduction"):
    download_path = "research/training_setup"
    read_timeout = 120

    baseline_tabicl_run_ids = [
        # Simplified Tabicl V0
        "li9vmts8",
        # # SimpleTabICL model ctd @lr=1e-4 + rng fix, smaller LR restart
        "4ij90pn6",
        "mpbdd3vp",  # FTM tabicl class only (old data, etc.)
        "xdmbkknm",  # simplified tabicl rerun
    ]

    query_filter = {"id": {"$regex": "|".join(baseline_tabicl_run_ids)}}

    def run_filter(self, run):
        return True


class TabICL(core.DownloadConfig, name="tabicl"):
    tags = ["Hayder::tabicl-model", constants.Tags.relevant]

    download_path = "research/training_setup"
    read_timeout = 120

    def query_filter(self) -> core.QueryFilterType:
        return {"tags": {"$all": self.tags}}

    # def line_configs(self):
    # return {
    # f"Running: {constants.MetricNames.t_cross_entropy}": dict(
    # plot_metric=constants.MetricNames.t_cross_entropy,
    # window=1000,
    # min_periods=1,
    # ),
    # f"Running: {constants.MetricNames.eip_acc}": dict(
    # plot_metric=constants.MetricNames.eip_acc,
    # window=15000,
    # min_periods=1,
    # ),
    # }


class SOTA(core.DownloadConfig, name="sota"):
    tags = {"Hayder::MS4-SOTA"}
    download_path = "research/training_setup"
    read_timeout = 120

    # https://fundamental.wandb.io/research/training_setup/runs/t53dts72/overview
    BASELINE_CLF_RUN_ID = "t53dts72"
    # https://fundamental.wandb.io/research/training_setup/runs/0fxigprp/overview
    # This one should be working for both clf and reg, I think it's best overall.
    BASELINE_REG_RUN_ID = "0fxigprp"

    def run_filter(self, run: wandb.apis.public.Run) -> bool:
        my_sota_run = self.tags.issubset(run.tags) and run.user.name == "Hayder Elesedy"
        baseline_run = run.id in [self.BASELINE_CLF_RUN_ID, self.BASELINE_REG_RUN_ID]
        return my_sota_run or baseline_run

    def clf_runs(self, run: wandb.apis.public.Run) -> bool:
        return (
            constants.Tags.classification in run.tags
            or run.id == self.BASELINE_CLF_RUN_ID
        )

    def reg_runs(self, run: wandb.apis.public.Run) -> bool:
        exclude_ids = [
            # https://fundamental.wandb.io/research/training_setup/runs/vm3uynfk/overview
            # This was the run with min_values_std not set, so doesn't learn.
            "vm3uynfk",
        ]
        yes = (
            constants.Tags.regression in run.tags or run.id == self.BASELINE_REG_RUN_ID
        )
        no = run.id in exclude_ids
        return yes and not no

    def line_configs(self):
        return {
            f"Regression: {constants.MetricNames.t_huber}": dict(
                plot_metric=constants.MetricNames.t_huber,
                window=1000,
                run_filter=self.reg_runs,
                min_periods=1,
            ),
            f"Regression: {constants.MetricNames.e_huber}": dict(
                plot_metric=constants.MetricNames.e_huber,
                window=10000,
                run_filter=self.reg_runs,
                min_periods=1,
            ),
            f"Regression: {constants.MetricNames.eip_mse}": dict(
                plot_metric=constants.MetricNames.eip_mse,
                window=10000,
                run_filter=self.reg_runs,
                min_periods=1,
            ),
            f"Classification: {constants.MetricNames.t_cross_entropy}": dict(
                plot_metric=constants.MetricNames.t_cross_entropy,
                window=1000,
                run_filter=self.clf_runs,
                min_periods=1,
            ),
            f"Classification: {constants.MetricNames.eip_acc}": dict(
                plot_metric=constants.MetricNames.eip_acc,
                window=10000,
                run_filter=self.clf_runs,
                min_periods=1,
            ),
        }
