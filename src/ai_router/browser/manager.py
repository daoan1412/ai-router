from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ai_router.browser.cloak import launch_persistent_context_async
from playwright.async_api import BrowserContext, Page

from ai_router.browser.pwcompat import PlaywrightError

from ai_router.config import AppConfig


class BrowserManager:
    def __init__(
        self,
        config: AppConfig,
        *,
        on_context_reset: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_context_reset = on_context_reset
        self._ctx: BrowserContext | None = None
        self._lock = asyncio.Lock()

    async def ensure_context(self) -> BrowserContext:
        if not await self._context_alive():
            await self._reset_context()
        if self._ctx is None:
            self._config.profile_dir.mkdir(parents=True, exist_ok=True)
            self._ctx = await launch_persistent_context_async(
                str(self._config.profile_dir),
                headless=False,
            )
        return self._ctx

    async def _context_alive(self) -> bool:
        if self._ctx is None:
            return False
        try:
            return not self._ctx.is_closed()
        except PlaywrightError:
            return False

    async def _reset_context(self) -> None:
        if self._ctx is not None:
            try:
                await self._ctx.close()
            except PlaywrightError:
                pass
        self._ctx = None
        if self._on_context_reset is not None:
            self._on_context_reset()

    async def new_page(self) -> Page:
        ctx = await self.ensure_context()
        try:
            if ctx.pages:
                return ctx.pages[0]
            return await ctx.new_page()
        except PlaywrightError:
            await self._reset_context()
            ctx = await self.ensure_context()
            return await ctx.new_page()

    async def new_tab(self) -> Page:
        """Always create a NEW tab (new_page() reuses pages[0])."""
        ctx = await self.ensure_context()
        try:
            return await ctx.new_page()
        except PlaywrightError:
            await self._reset_context()
            ctx = await self.ensure_context()
            return await ctx.new_page()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[BrowserContext]:
        """Serialize browser automation — concurrent asks wait in FIFO order."""
        async with self._lock:
            yield await self.ensure_context()

    async def close(self) -> None:
        await self._reset_context()
