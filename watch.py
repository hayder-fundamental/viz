import argparse
import logging
import time

import tqdm

import core
import utils


def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Watch running training jobs and update cache.")
    parser.add_argument(
        "username",
        type=str,
        help="Search only eval runs for this user."
        " If not given we will print them all.",
    )
    parser.add_argument(
        "--wait",
        type=utils.validator_int_strict_positive("wait"),
        help="How long to wait between checking for new data.",
        default=5 * 60,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for wandb `Api.runs` call. "
        "Wandb uses a default value if not specified.",
        default=None,
    )
    parser.add_argument(
        "--page-size",
        type=utils.validator_int_strict_positive("--page-size"),
        help="Number of rows to download per wandb query in `run.scan_history`",
        default=100,
    )
    parser.add_argument(
        "--max-threads",
        type=utils.validator_int_strict_positive("--max-threads"),
        default=1,
        help="Maximum number of concurrent threads for download.",
    )
    utils.add_log_level_arg(parser, default="info")
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_args()
    logger = logging.getLogger()
    logging.basicConfig(level=args.log_level)

    downloader = core.HistoryManager()

    while True:
        logger.info("Finding ongoing training runs.")
        runs = utils.get_train_running(
            username=args.username,
            timeout=args.timeout,
        )
        message = ["Found ongoing runs"]
        message.append("{:<75} {:>10}".format("Run Name", "Run ID"))
        for run in runs:
            message.append("{:<75} {:>10}".format(run.name, run.id))
        logger.info("\n".join(message))

        if args.max_threads == 1:
            logging.info("Downloading run data serially.")
            for run in tqdm.tqdm(runs, desc="Updating run data"):
                downloader.fetch_history(
                    run,
                    page_size=args.page_size,
                    update_cache=True,
                )
        else:
            logging.info("Downloading run data on %d threads.", args.max_threads)
            downloader.fetch_histories(
                runs,
                max_threads=args.max_threads,
                page_size=args.page_size,
                update_cache=True,
            )

        logging.info("Waiting for %d seconds.", args.wait)
        time.sleep(args.wait)
