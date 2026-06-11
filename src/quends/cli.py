"""Minimal command-line interface for QUENDS.

Currently supports reporting the version and loading + summarizing a single
variable from a data file. Extend the subcommands as the package grows.

Usage
-----
    python -m quends --version
    python -m quends summary <file> <variable>
"""

import argparse
import sys

from . import __version__


def _cmd_summary(args: argparse.Namespace) -> int:
    from .preprocessing.csv import from_csv
    from .preprocessing.netcdf import from_netcdf

    loader = from_netcdf if args.file.endswith(".nc") else from_csv
    ds = loader(args.file, args.variable)
    df = ds.data
    print(f"file      : {args.file}")
    print(f"variable  : {args.variable}")
    print(f"columns   : {list(df.columns)}")
    print(f"n_samples : {len(df)}")
    if "time" in df.columns and len(df):
        print(f"time range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quends", description="QUENDS command-line interface.")
    parser.add_argument("--version", action="version", version=f"quends {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_summary = sub.add_parser("summary", help="Load one variable and print a summary.")
    p_summary.add_argument("file", help="Path to a .csv or .nc data file.")
    p_summary.add_argument("variable", help="Variable/column name to load.")
    p_summary.set_defaults(func=_cmd_summary)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
