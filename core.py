import asyncio
import concurrent.futures
import typing

import matplotlib.pyplot as plt
import pandas as pd
import tqdm.asyncio
import wandb

_LineGeneratorYieldType = tuple[
    wandb.apis.public.Run,
    tuple[pd.Series, pd.Series],
]
_RunFilterType = typing.Callable[wandb.apis.public.Run, bool]


class HistoryDownloader:
    """Download and filter runs from wandb api."""

    def __init__(
        self,
        api_key: str | None = None,
    ):
        wandb.login(host="https://fundamental.wandb.io", key=api_key)

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

    # TODO(HE): It's possible to add in a switch to get full history
    # instead of sampling but
    # it requires a little further work and we aren't using it atm
    # it would use run.scan_history
    def fetch_history(
        self,
        runs: typing.Iterable[wandb.apis.public.Run],
        n_samples: int,
        keys: list[str] | None = None,
    ) -> list[pd.DataFrame]:
        def fetch_single(run: wandb.apis.public.Run) -> pd.DataFrame:
            return run.history(samples=n_samples, keys=keys)

        async def download(
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> list[pd.DataFrame]:
            loop = asyncio.get_running_loop()
            async_futures = [
                loop.run_in_executor(executor, lambda: fetch_single(run))
                for run in runs
            ]
            return await tqdm.asyncio.tqdm.gather(
                *async_futures, desc="Fetching run data."
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
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
        ax.plot(*args, label=run.name)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax
