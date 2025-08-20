import constants

import wandb

import core


class TabICL(core.DashboardConfig, name="tabicl"):
    tags = {"Hayder::tabicl-model", constants.Tags.relevant}

    download_path = "research/training_setup"
    n_samples = 10_000_000
    read_timeout = 120
    select_metrics = [
        constants.MetricNames.t_cross_entropy,
        *constants.MetricNames.eip_accs,
    ]

    baseline_tabicl_run_ids = [
        # Simplified Tabicl V0
        "li9vmts8",
        # SimpleTabICL model ctd @lr=1e-4 + rng fix, smaller LR restart
        "4ij90pn6",
    ]

    def run_filter(self, run):
        return (
            set(run.tags).issuperset(self.tags)
            or run.id in self.baseline_tabicl_run_ids
        )

    def line_configs(self):
        return {
            f"Running: {constants.MetricNames.t_cross_entropy}": dict(
                plot_metric=constants.MetricNames.t_cross_entropy,
                window=1000,
                min_periods=1,
            ),
            f"Running: {constants.MetricNames.eip_acc}": dict(
                plot_metric=constants.MetricNames.eip_acc,
                window=5000,
                min_periods=1,
            ),
            # f"Woj Params: {constants.MetricNames.t_cross_entropy}": dict(
            # plot_metric=constants.MetricNames.t_cross_entropy,
            # window=1000,
            # run_filter=self.wojtek_params,
            # min_periods=1,
            # ),
            # f"Woj Params: {constants.MetricNames.eip_acc}": dict(
            # plot_metric=constants.MetricNames.eip_acc,
            # window=5000,
            # run_filter=self.wojtek_params,
            # min_periods=1,
            # ),
        }


class SOTA(core.DashboardConfig, name="sota"):
    tags = {"Hayder::MS4-SOTA"}
    download_path = "research/training_setup"
    n_samples = 1_000_000
    read_timeout = 120
    select_metrics = [
        constants.MetricNames.t_cross_entropy,
        constants.MetricNames.t_huber,
        constants.MetricNames.e_huber,
        *constants.MetricNames.eip_accs,
        constants.MetricNames.eip_mse,
    ]

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
