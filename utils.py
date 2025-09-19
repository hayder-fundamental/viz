import argparse


def add_log_level_arg(parser: argparse.ArgumentParser, default: str):
    parser.add_argument(
        "--log-level",
        type=lambda s: s.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=default.upper(),
        help="Set the logging level of the root logger (default: %(default)s)."
        " Argument is case insensitive.",
    )
