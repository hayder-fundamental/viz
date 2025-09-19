"""Download and locally cache data from wandb runs."""

import argparse
import logging
import typing

import tqdm

import core
import utils


def cmd_args() -> argparse.Namespace:
    def validator_int_strict_positive(argname: str) -> typing.Callable[[str], int]:
        def validator(value: str) -> int:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError(f"`{argname}` must be strictly positive, got {value}.")
            return int_value

        return validator

    parser = argparse.ArgumentParser("Download data from wandb.")
    parser.add_argument(
        "name",
        type=str,
        help=(
            "Name of dashboard, must be a subclass of `core.DownloadConfig`."
            " Define your class like MyConfig(core.DownloadConfig, name='my-name')"
            " and it will become available as an argument here."
        ),
        choices=list(core._REGISTRY.keys()),
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache data before downloading.",
    )
    parser.add_argument(
        "--page-size",
        type=validator_int_strict_positive("--page-size"),
        help="Number of rows to download per wandb query in `run.scan_history`",
        default=100,
    )
    parser.add_argument(
        "--max-threads",
        type=validator_int_strict_positive("--max-threads"),
        default=1,
        help="Maximum number of concurrent threads for download.",
    )
    utils.add_log_level_arg(parser)
    return parser.parse_args()



if __name__ == "__main__":
    args = cmd_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    cfg = core.get_config(args.name)

    downloader = core.HistoryDownloader()
    logging.info("Downloading runs for config %s.", args.name)
    runs = downloader.fetch_runs(
        path=cfg.download_path,
        timeout=cfg.read_timeout,
        query_filter=cfg.query_filter(),
        run_filter=cfg.run_filter(),
    )
    logging.info("Collected runs with ids %s", [r.id for r in runs])
    if args.clear_cache:
        logging.info("Clearing cached data for selected runs.")
        downloader.clear_cache(runs)

    if args.max_threads == 1:
        logging.info("Downloading run data.")
        for run in tqdm.tqdm(runs, desc="Downloading data"):
            downloader.fetch_history(
                run,
                page_size=args.page_size,
                update_cache=True,
            )
    else:
        downloader.fetch_histories(
            runs,
            max_threads=args.max_threads,
            page_size=args.page_size,
            update_cache=True,
        )
