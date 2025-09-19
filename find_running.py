import argparse
import logging

import core
import utils

EVAL_REGEX = "^[a-z0-9]*_step_[0-9]*_.*"
EVAL_PATH = "research/evaluating_our_models"
TRAIN_PATH = "research/training_setup"


def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Find running jobs.")
    parser.add_argument(
        "--username",
        type=str,
        help="Search only eval runs for this user."
        " If not given we will print them all.",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for wandb `Api.runs` call. "
        "Wandb uses a default value if not specified.",
        default=None,
    )
    utils.add_log_level_arg(parser, default="info")
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_args()
    logger = logging.getLogger()
    logging.basicConfig(level=args.log_level)

    eval_query_filter = {
        "$and": [
            {"displayName": {"$regex": EVAL_REGEX}},
            {"state": "running"},
        ]
    }
    train_query_filter = {
        "$and": [
            {"state": "running"},
        ]
    }
    if args.username is not None:
        eval_query_filter["$and"].append({"username": args.username})
        train_query_filter["$and"].append({"username": args.username})

    logging.info("Fetching ongoing eval runs.")
    eval_running = core.fetch_runs(
        path=EVAL_PATH,
        timeout=args.timeout,
        query_filter=eval_query_filter,
    )

    logging.info("Fetching ongoing training runs.")
    train_running = core.fetch_runs(
        path=TRAIN_PATH,
        timeout=args.timeout,
        query_filter=train_query_filter,
    )

    eval_names = [run.name for run in eval_running]
    train_running_ids = {run.id for run in train_running}
    trains_with_eval = {s.split("_")[0] for s in eval_names}

    def print_row(run):
        print(f"{run.name:<75} {run.id:>10}")

    print("Train running:")
    for run in train_running:
        print_row(run)
    print()
    print("Eval running:")
    for run in eval_running:
        print_row(run)
    print()
    print("Training runs without running evals:")
    for run in train_running:
        if run.id in trains_with_eval:
            continue
        print_row(run)
