from pathlib import Path

import tomllib


def _project() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def _dependency(name: str) -> str:
    for spec in _project()["dependencies"]:
        if spec.split(">=")[0].split("[")[0].strip() == name:
            return spec
    raise AssertionError(f"{name} is not a declared dependency")


def test_distribution_name_is_mcp_ai_router():
    assert _project()["name"] == "mcp-ai-router"


def test_console_script_is_ai_router_only():
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert "ai-router" in scripts
    assert scripts["ai-router"] == "ai_router.cli.main:app"
    assert "ai" not in scripts


def test_mcp_dependency_has_upper_bound():
    assert _dependency("mcp") == "mcp>=1.13,<2"


def test_python_requires_311_to_4():
    assert _project()["requires-python"] == ">=3.11,<4"


def test_build_backend_is_uv():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["build-system"]["build-backend"] == "uv_build"
    assert data["tool"]["uv"]["build-backend"]["module-name"] == "ai_router"
