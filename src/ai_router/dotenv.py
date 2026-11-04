"""Minimal .env support — no external dependency.

load_dotenv() reads KEY=VALUE pairs into os.environ (without overriding values
already set in the real environment). set_env_vars() upserts keys back into the
file so provisioned credentials survive across runs.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_PATH = Path.cwd() / ".env"


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export ") :].strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a .env file into os.environ. Returns the parsed mapping.

    Real environment variables win unless override=True.
    """
    env_path = path or DEFAULT_ENV_PATH
    parsed: dict[str, str] = {}
    if not env_path.exists():
        return parsed
    for line in env_path.read_text(encoding="utf-8").splitlines():
        item = _parse_line(line)
        if item is None:
            continue
        key, value = item
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def set_env_vars(mapping: dict[str, str], path: Path | None = None) -> Path:
    """Upsert KEY=VALUE pairs into the .env file, preserving other lines."""
    env_path = path or DEFAULT_ENV_PATH
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(mapping)

    out: list[str] = []
    for line in lines:
        item = _parse_line(line)
        if item is not None and item[0] in remaining:
            key = item[0]
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    # Reflect immediately in the current process.
    for key, value in mapping.items():
        os.environ[key] = value
    return env_path
