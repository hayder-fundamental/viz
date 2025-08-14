import constants


_REGISTRY = {}


# This pattern of populating the registry is dumb bc you can't see all the keys
# in one place, but it's a bit of fun.
class DashboardConfig:
    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[name] = cls


def get_config(name: str) -> DashboardConfig:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Config with name {name} not found.\nHave configs:\n{sorted(_REGISTRY.keys())}."
        )


class TabICL(DashboardConfig, name="tabicl"):
    download_path = "research/training_setup"
    n_samples = 10_000
    read_timeout = 60
    select_metrics = [
        constants.MetricNames.t_cross_entropy,
        *constants.MetricNames.eip_accs,
    ]

    @staticmethod
    def select_tabicl_runs(run):
        return run.id in constants.RunIDs.tabicl_run_ids

    @staticmethod
    def running_or_baseline(run):
        return run.state == "running" or run.id in constants.RunIDs.woj_tabicl_run_ids

    @staticmethod
    def wojtek_params(run):
        return (
            "woj params" in run.name.lower()
            or run.id in constants.RunIDs.woj_tabicl_run_ids
        )

    def line_configs(self):
        return {
            f"Running: {constants.MetricNames.t_cross_entropy}": dict(
                plot_metric=constants.MetricNames.t_cross_entropy,
                window=1000,
                run_filter=self.running_or_baseline,
                min_periods=1,
            ),
            f"Running: {constants.MetricNames.eip_acc}": dict(
                plot_metric=constants.MetricNames.eip_acc,
                window=5000,
                run_filter=self.running_or_baseline,
                min_periods=1,
            ),
            f"Woj Params: {constants.MetricNames.t_cross_entropy}": dict(
                plot_metric=constants.MetricNames.t_cross_entropy,
                window=1000,
                run_filter=self.wojtek_params,
                min_periods=1,
            ),
            f"Woj Params: {constants.MetricNames.eip_acc}": dict(
                plot_metric=constants.MetricNames.eip_acc,
                window=5000,
                run_filter=self.wojtek_params,
                min_periods=1,
            ),
        }


class SOTA(DashboardConfig, name="sota"):
    pass
