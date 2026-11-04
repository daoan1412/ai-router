"""Launch the persistent browser context: CloakBrowser (Playwright) or real Chrome (Patchright)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from cloakbrowser.config import get_default_stealth_args
from cloakbrowser.download import ensure_binary
from playwright.async_api import BrowserContext, async_playwright

SEED_FILE = "ai-router-fingerprint-seed"


def stable_stealth_args(user_data_dir: str) -> list[str]:
    """CloakBrowser's stealth args with the fingerprint seed pinned per profile.

    CloakBrowser draws a random ``--fingerprint`` seed on every launch, so a
    persistent profile (same cookies) would show up as a different device each
    time — sites like Google then distrust the session. The first seed is saved
    next to the profile and reused.
    """
    args = list(get_default_stealth_args())
    seed_path = Path(user_data_dir) / SEED_FILE
    try:
        seed = seed_path.read_text(encoding="utf-8").strip()
    except OSError:
        seed = ""
    if not seed.isdigit():
        seed = next((a.split("=", 1)[1] for a in args if a.startswith("--fingerprint=")), "")
        if seed:
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_text(seed, encoding="utf-8")
    return [f"--fingerprint={seed}" if a.startswith("--fingerprint=") else a for a in args]


async def launch_persistent_context_async(
    user_data_dir: str,
    *,
    headless: bool = False,
    **kwargs: Any,
) -> BrowserContext:
    """Open a persistent browser profile.

    Backend from ``AI_ROUTER_BROWSER``: ``cloak`` (default) = CloakBrowser's
    stealth Chromium via Playwright; ``chrome`` = the installed Google Chrome
    via Patchright (real, stable fingerprint; automation leaks patched), on a
    sibling ``<profile>-chrome`` dir seeded from the CloakBrowser profile.
    """
    if browser_backend() == "chrome":
        return await _launch_real_chrome(user_data_dir, headless=headless, **kwargs)
    pw = await async_playwright().start()
    binary_path = ensure_binary()
    chrome_args = stable_stealth_args(user_data_dir)

    ctx = await pw.chromium.launch_persistent_context(
        user_data_dir,
        executable_path=binary_path,
        headless=headless,
        args=chrome_args,
        ignore_default_args=["--enable-automation"],
        **kwargs,
    )
    return _stop_driver_on_close(ctx, pw)


def browser_backend() -> str:
    return (os.getenv("AI_ROUTER_BROWSER") or "cloak").strip().lower()


# Caches are not worth copying (and can be big); sessions live elsewhere.
_SKIP_ON_COPY = ("Cache", "Code Cache", "GPUCache", "GrShaderCache", "ShaderCache",
                 "DawnGraphiteCache", "DawnWebGPUCache", "CacheStorage", "Crashpad",
                 "SingletonLock", "SingletonCookie", "SingletonSocket", SEED_FILE)


def chrome_profile_dir(user_data_dir: str) -> Path:
    """The real-Chrome profile next to ``user_data_dir``, seeded from it once.

    Kept separate so the CloakBrowser profile is never upgraded by a newer Chrome
    (and stays usable to switch back).
    """
    src = Path(user_data_dir)
    if src.name.endswith("-chrome"):
        return src
    dst = src.with_name(src.name + "-chrome")
    if not dst.exists() and src.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*_SKIP_ON_COPY))
    return dst


async def _launch_real_chrome(user_data_dir: str, *, headless: bool, **kwargs: Any) -> BrowserContext:
    from patchright.async_api import async_playwright as patchright_playwright

    profile = chrome_profile_dir(user_data_dir)
    pw = await patchright_playwright().start()
    # Patchright's recommended setup: real Chrome, no viewport emulation, no
    # custom UA/headers, so the fingerprint is the machine's own.
    ctx = await pw.chromium.launch_persistent_context(
        str(profile), channel="chrome", headless=headless, no_viewport=True, **kwargs
    )
    return _stop_driver_on_close(ctx, pw)


def _stop_driver_on_close(ctx, pw):
    """Make ``ctx.close()`` also stop the Playwright/Patchright driver."""
    original_close = ctx.close

    async def _close_with_cleanup() -> None:
        await original_close()
        await pw.stop()

    ctx.close = _close_with_cleanup  # type: ignore[method-assign]
    return ctx
