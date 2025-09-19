import abc
import concurrent.futures
import logging
import os
import typing

import matplotlib.pyplot as plt
import pandas as pd
import platformdirs
import tqdm.asyncio
import wandb

logger = logging.getLogger(__name__)

_LineGeneratorYieldType = tuple[
    wandb.apis.public.Run,
    tuple[pd.Index, pd.Series],
]
# TODO(HE): Check this type hint.
QueryFilterType = dict[str, "list[QueryFilterType] | QueryFilterType | str"]
RunFilterType = typing.Callable[[wandb.apis.public.Run], bool]

_REGISTRY = {}

# TODO(HE): Update LineConfigs and plotting functionality
# TODO(HE): clean up exec async function (maybe use in fetch histories...?)


# This pattern of populating the registry is just a bit of fun.
# On the one hand you can't see all the keys in one place, but
# on the other you don't need to modify core code to be able to run
# your own config.
class DownloadConfig(abc.ABC):
    """Base class for download configs."""

    download_path: str
    read_timeout: int | None

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[name] = cls

    def query_filter(self) -> QueryFilterType | None:
        """MongoDB query to give to weights and biases API to filter runs to download.

        See here for usage: https://docs.wandb.ai/ref/python/public-api/api/#method-apiruns.
        (You may also need to look online for more detail on how to use this syntax.)
        """
        return None

    def run_filter(self) -> RunFilterType | None:
        """Optional callable to filter runs after download, quicker to use a query_filter"""
        return None


def get_config(name: str) -> DownloadConfig:
    """Retrieve config by name from registry.

    Args:
        name (str): Name you've given to the config, see usage.

    Returns:
        DownloadConfig: Retrieved config, subclass of DownloadConfig.

    Raises:
        ValueError: If the config with `name` not found.

    Usage:
        When defining the config, one should make the class as
        ```
        class MyConfig(core.DownloadConfig, name="my-name"):
            ...
        ```
        Then this class can be retrieved by `get_config("my-name")`.
    """
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Config with name {name} not found.\nHave configs:\n{sorted(_REGISTRY.keys())}."
        )


class HistoryDownloader:
    def __init__(
        self,
        api_key: str | None = None,
        _login: bool = True,  # Set to False for testing, so tests don't access wandb.
    ):
        """Download runs and their history from Weights and Biases API.

        This class implements local caching as CSV so we don't need to
        re-download old data.

        Initialisation creates a cache directory if it doesn't exist already
        and logs in to weights and biases.

        Args:
            api_key (str | None): WandB API key. If None it is detected automatically
                by `wandb.login`. Default None.
        """
        if _login:
            wandb.login(host="https://fundamental.wandb.io", key=api_key)
        self.cache_dir = os.path.join(platformdirs.user_cache_dir(), "viz", "run_data")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_runs(
        self,
        path: str,
        timeout: int,
        query_filter: QueryFilterType | None = None,
        run_filter: RunFilterType | None = None,
        per_page: int = 50,
    ) -> list[wandb.apis.public.Run]:
        """Download and filter wanbd runs.

        Thin wrapper around `wandb.apis.public.Api.runs`.

        Args:
            path (str): Query runs from this path.
            timeout (int): timeout
            query_filter (QueryFilterType | None): MongoDB query to filter
                runs in `wandb.apis.public.Api.runs` call.
            run_filter (RunFilterType | None): Optional callable to filter
                runs after download. Faster to use `query_filter` if possible.
            per_page (int): per_page

        Returns:
            list[wandb.apis.public.Run]: Downloaded and filtered runs.
        """
        api = wandb.Api(timeout=timeout)
        all_runs = api.runs(
            path,
            filters=query_filter,
            per_page=per_page,
        )
        return (
            list(all_runs)
            if run_filter is not None
            else list(filter(run_filter, all_runs))
        )

    def get_cache_path(self, run: wandb.apis.public.Run) -> str:
        """Path to cache location for history of `run`.

        Args:
            run (wandb.apis.public.Run): The run who's cache you want.

        Returns:
            str: File name by `run.id` in a platform specific local cache directory.
        """
        return os.path.join(self.cache_dir, f"{run.id}.csv")

    def clear_cache(self, runs: wandb.apis.public.Run | list[wandb.apis.public.Run]):
        """Delete cached history data for `runs`.

        Args:
            runs (wandb.apis.public.Run | list[wandb.apis.public.Run]): A run or a list of runs.
        """
        if not isinstance(runs, list):
            runs = [runs]
        for run in runs:
            try:
                os.remove(self.get_cache_path(run))
            except FileNotFoundError:
                pass

    def read_cache(self, run: wandb.apis.public.Run) -> pd.DataFrame:
        """Read cached history data for `run`.

        Args:
            run (wandb.apis.public.Run): Read history for this run from local cache.

        Returns:
            pd.DataFrame: pandas DataFrame of run history.

        Raises:
            ValueError: No cache data found for `run`.
        """
        run_data_path = self.get_cache_path(run)
        if not os.path.exists(run_data_path):
            raise ValueError(f"No cached data found at path {run_data_path}.")
        # TODO(HE): Fix bad cache lines
        logger.debug("Reading cache from %s.", run_data_path)
        return pd.read_csv(run_data_path, on_bad_lines="warn")

    def write_cache(self, run: wandb.apis.public.Run, df: pd.DataFrame) -> None:
        """Write to CSV cache for `run`. This overwrites existing cache data.

        We do not write the index of `df`.

        We assume that the data in the cache does not have columns of mixed dtype.
        For instance, pandas will read ints in a column as strings if the first
        value of the column is string.

        Args:
            run (wandb.apis.public.Run): run who's cache you want to write.
                Cached data lives at `.get_cache_path(run)` locally.
            df (pd.DataFrame): The data to write. No checks are performed.
        """
        cache_path = self.get_cache_path(run)
        logging.debug("Writing cache at %s.", cache_path)
        return df.to_csv(cache_path, index=False)

    def fetch_history(
        self,
        run: wandb.apis.public.Run,
        page_size: int = 50,
        update_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch entire history for single run and cache results.

        Loads data from cache if it exists, then appends more recent
        data and saves the cache again.

        Args:
            run (wandb.apis.public.Run): The run.
            page_size (int): Number of rows of history to collect per
                internal query in `run.scan_history`.
            update_cache (bool): Whether to update local cached run
                data with additional data downloaded. Default True.

        Returns:
            pd.DataFrame: History of the run.
        """
        last_history_step = run.lastHistoryStep
        cache_path = self.get_cache_path(run)
        if os.path.exists(cache_path):
            cached = self.read_cache(run)
            start_step = cached["_step"].max() + 1
            if start_step > last_history_step:
                logger.debug(
                    "Cached data has max step %d and run.lastHistoryStep=%d. "
                    "Nothing to update returning cached data.",
                    start_step - 1,
                    last_history_step,
                )
                return cached
        else:
            logger.debug("No cached data found at %s", cache_path)
            cached = None
            start_step = 0

        expected_rows = last_history_step - start_step + 1

        logger.debug(
            "Scan history from step %d with page size %d. Expecting %d new rows.",
            start_step,
            page_size,
            expected_rows,
        )
        new_history = pd.DataFrame(
            list(
                run.scan_history(
                    min_step=start_step,
                    max_step=None,
                    page_size=page_size,
                )
            )
        ).map(lambda x: float("nan") if x is None else x)

        # Account for wandb sometimes returning too many rows.
        logger.debug(
            "Scan history returned %d new rows, expected "
            "%d (could change slightly if a step happens "
            "between logging calls).",
            new_history.shape[0],
            expected_rows,
        )
        logger.debug("Defensively selecting rows in requested range from result.")
        new_history = new_history.query("_step>=@start_step")

        data = (
            new_history
            if cached is None
            else pd.concat([cached, new_history], axis=0).reset_index(drop=True)
        )
        if update_cache:
            self.write_cache(run, data)
        return data

    def fetch_histories(
        self,
        runs: typing.Sequence[wandb.apis.public.Run],
        max_threads: int | None = None,
        page_size: int = 50,
        update_cache: bool = True,
    ) -> list[pd.DataFrame]:
        # Runs must not contain duplicates. Could cause issue in cache read/write.
        unique_run_ids = {run.id for run in runs}
        if len(unique_run_ids) < len(runs):
            raise ValueError("Detected duplicate runs.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            return list(
                tqdm.tqdm(
                    executor.map(
                        lambda run: self.fetch_history(
                            run,
                            page_size=page_size,
                            update_cache=update_cache,
                        ),
                        runs,
                    ),
                    total=len(runs),
                    desc="Fetching run histories",
                )
            )


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
        run_filter: RunFilterType = lambda x: True,
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


# TODO(HE): Is there a better way of doing this?
# Need to import configs here to compile the classes
# and populate the registry.
import configs  # noqa: F401, E402, E501  # pylint: disable=unused-import, wrong-import-position
