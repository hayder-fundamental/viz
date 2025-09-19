import argparse

import wandb

import core
import constants


def add_log_level_arg(parser: argparse.ArgumentParser, default: str) -> None:
    """Add log level argument to argument parser.

    Args:
        parser (argparse.ArgumentParser): The parser you're using. Modifies
            in-place.
        default (str): The default value for this argument.
    """
    parser.add_argument(
        "--log-level",
        type=lambda s: s.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=default.upper(),
        help="Set the logging level of the root logger (default: %(default)s)."
        " Argument is case insensitive.",
    )


def get_train_running(
    username: str | None = None,
    timeout: int | None = None,
) -> list[wandb.apis.public.Run]:
    """Get ongoing training runs.

    Args:
        username (str | None): Filter for this username, if
            None we return ongoing training runs for all users.
            Default None.
        timeout (int | None): Timeout for wandb `Api.runs` call.
            Wandb uses a default value if not specified. Default None.

    Returns:
        list[wandb.apis.public.Run]: Ongoing training runs, filtered by
            username if given.
    """
    query_filter = {
        "$and": [
            {"state": "running"},
        ]
    }
    if username is not None:
        query_filter["$and"].append({"username": username})
    return core.fetch_runs(
        path=constants.Paths.TRAIN,
        timeout=timeout,
        query_filter=query_filter,
    )
