"""Tests for the command-line interface (``quends.cli`` / ``python -m quends``)."""

import pandas as pd
import pytest

from quends import cli


def test_build_parser_parses_summary_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["summary", "data.csv", "Q"])
    assert args.command == "summary"
    assert args.file == "data.csv"
    assert args.variable == "Q"
    assert args.func is cli._cmd_summary


def test_main_no_command_prints_help(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage" in out.lower()


def test_main_version_exits_zero():
    # argparse's ``version`` action prints and raises SystemExit(0).
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_main_summary_csv(tmp_path, capsys):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"time": [0.0, 1.0, 2.0], "Q": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)

    rc = cli.main(["summary", str(csv), "Q"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "n_samples : 3" in out
    assert "time range: 0.0 -> 2.0" in out
    assert "Q" in out


def test_main_module_is_importable():
    # Importing the module exercises ``python -m quends``'s entry module.
    import quends.__main__  # noqa: F401
