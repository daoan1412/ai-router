from pathlib import Path

import tomllib
from typer.testing import CliRunner

from ai_router.cli.main import app

runner = CliRunner(mix_stderr=False)


def _expected_version() -> str:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_version_flag():
    expected = _expected_version()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == expected
    assert result.stdout.endswith("\n") or result.stdout == expected
