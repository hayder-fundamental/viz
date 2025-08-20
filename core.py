import abc
import asyncio
import concurrent.futures
import os
import typing
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import platformdirs
import tqdm.asyncio
import wandb

_LineGeneratorYieldType = tuple[
    wandb.apis.public.Run,
    tuple[pd.Index, pd.Series],
]
_RunFilterType = typing.Callable[wandb.apis.public.Run, bool]

_REGISTRY = {}


# This pattern of populating the registry is just a bit of fun.
# On the one hand you can't see all the keys in one place, but
# on the other you don't need to modify core code to be able to run
# your own config.
class DashboardConfig(abc.ABC):
    download_path: str
    read_timeout: int

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[name] = cls

    @abc.abstractmethod
    def run_filter(self, run: wandb.apis.public.Run) -> bool:
        pass


def get_config(name: str) -> DashboardConfig:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Config with name {name} not found.\nHave configs:\n{sorted(_REGISTRY.keys())}."
        )


class HistoryDownloader:
    """Download and filter runs from wandb api."""

    def __init__(
        self,
        api_key: str | None = None,
    ):
        wandb.login(host="https://fundamental.wandb.io", key=api_key)
        self.cache_dir = os.path.join(platformdirs.user_cache_dir(), "viz", "run_data")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_runs(
        self,
        path: str,
        timeout: int,
        run_filter: _RunFilterType = lambda run: True,
    ) -> list[wandb.apis.public.Run]:
        api = wandb.Api(timeout=timeout)
        # TODO(HE): This currently fetches all runs, can we filter?
        # I couldn't get filter by username to work earlier.
        # TODO(HE): Possible we run into annoyances with the pagination at
        # some point? Left this at default value.
        all_runs = api.runs(path, filters=None, per_page=50)
        return [run for run in all_runs if run_filter(run)]

    def get_cache_path(self, run: wandb.apis.public.Run) -> str:
        return os.path.join(self.cache_dir, f"{run.id}.csv")

    def clear_cache(self, *runs: wandb.apis.public.Run):
        for run in runs:
            try:
                os.remove(self.get_cache_path(run))
            except FileNotFoundError:
                pass

    def fetch_history(self, run: wandb.apis.public.Run) -> pd.DataFrame:
        """Fetch entire history for single run and cache results.

        Loads data from cache if exists.

        Args:
            run (wandb.apis.public.Run): The run.

        Returns:
            pd.DataFrame: The history.
        """
        run_data_path = self.get_cache_path(run)
        cached = (
            pd.read_csv(run_data_path, index_col=0)
            if os.path.exists(run_data_path)
            else pd.DataFrame()
        )
        try:
            start_step = cached["_step"].max() + 1
        except KeyError:
            start_step = 0

        new_history = pd.DataFrame(
            list(run.scan_history(min_step=start_step, max_step=None))
        ).map(lambda x: float("nan") if x is None else x)
        with warnings.catch_warnings():
            # Pandas gives a warning about behaviour on empty DF concat,
            # but it seems fine. See this issue:
            # https://github.com/pandas-dev/pandas/issues/55928.
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = pd.concat([cached, new_history], axis=0).reset_index(drop=True)
        if not data.empty:
            data.to_csv(run_data_path)
        return data

    def fetch_histories(
        self,
        runs: typing.Iterable[wandb.apis.public.Run],
        max_threads: int | None = None,
    ) -> list[pd.DataFrame]:
        # TODO(HE): Enforce this.
        # Runs must not contain duplicates! Could make thread issue in cache read.
        async def download(
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> list[pd.DataFrame]:
            loop = asyncio.get_running_loop()
            async_futures = [
                loop.run_in_executor(executor, lambda: self.fetch_history(run))
                for run in runs
            ]
            return await tqdm.asyncio.tqdm.gather(
                *async_futures, desc="Fetching run data."
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            return asyncio.run(download(executor))


class LineGenerator:
    def __init__(self, runs: list[wandb.apis.public.Run], data_df: pd.DataFrame):
        self.runs = runs
        self.data_df = data_df

    def smooth(
        self,
        srs: pd.Series,
        window: int,
        min_periods: int = 1,
        **kwds,
    ) -> pd.Series:
        trailing_nan_mask = srs.bfill().notnull()
        return (
            srs.loc[trailing_nan_mask]
            .rolling(window, min_periods=min_periods, **kwds)
            .mean()
            .ffill()
        )

    def __call__(
        self,
        plot_metric: str,
        window: int,
        run_filter: _RunFilterType = lambda x: True,
        *,
        min_periods=1,
        **smooth_kwds,
    ) -> typing.Generator[_LineGeneratorYieldType, None, None]:
        for run in self.runs:
            if run_filter(run):
                raw = self.data_df.get((run.name, plot_metric))
                if raw is None:
                    continue
                to_plot = self.smooth(
                    raw, window=window, min_periods=min_periods, **smooth_kwds
                )
                yield run, (to_plot.index, to_plot)


def plot_lines(lines: typing.Iterable[_LineGeneratorYieldType], title: str):
    fig, ax = plt.subplots(1)
    for run, args in lines:
        stub = "(*) " if run.state == "running" else ""
        ax.plot(*args, label=stub + run.name)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


# TODO(HE): What's a better way of doing this?
# Need to import configs here to compile the classes
# and populate the registry.
import configs  # noqa: F401, E402, E501  # pylint: disable=unused-import, wrong-import-position
