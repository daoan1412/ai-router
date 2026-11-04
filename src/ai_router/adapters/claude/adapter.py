from __future__ import annotations

from playwright.async_api import Page

from ai_router.browser.pwcompat import PlaywrightTimeout

from ai_router.adapters.base import SessionStatus
from ai_router.adapters.claude.planner import ClaudePlanner
from ai_router.adapters.claude.selectors import (
    CLAUDE_COMPLETION_RE,
    CLAUDE_ERROR_MARKERS,
    CLAUDE_URL,
    SEL_LOGIN,
    SEL_PROMPT_INPUT,
    SEL_SUBMIT_BUTTON,
)
from ai_router.adapters.claude.stream import parse_stream_done
from ai_router.adapters.claude.wait import (
    is_rate_limited,
    is_stop_visible,
    read_response_snapshot,
    submit_ready,
)
from ai_router.browser.profile import ProviderProfile, ProviderSelectors
from ai_router.config import AppConfig


class ClaudeAdapter:
    id = "claude"
    name = "Claude"
    keywords: list[str] = ["claude", "@claude", "anthropic"]
    status = "available"

    async def check_session(self, page: Page) -> SessionStatus:
        return await self.ensure_page_ready(page)

    async def ensure_page_ready(self, page: Page) -> SessionStatus:
        """Open a fresh chat and verify login."""
        await page.goto(CLAUDE_URL, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=15_000)
            return SessionStatus.LOGGED_IN
        except PlaywrightTimeout:
            if await page.locator(SEL_LOGIN).count() > 0:
                return SessionStatus.LOGGED_OUT
            return SessionStatus.UNKNOWN

    async def open_new_chat(self, page: Page) -> None:
        await page.goto(CLAUDE_URL, wait_until="domcontentloaded")

    def build_profile(self, cfg: AppConfig) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.id,
            stream_url_re=CLAUDE_COMPLETION_RE,
            parse_stream_done=parse_stream_done,
            is_stop_visible=is_stop_visible,
            read_response_snapshot=read_response_snapshot,
            is_rate_limited=is_rate_limited,
            submit_ready=submit_ready,
            planner=ClaudePlanner(),
            selectors=ProviderSelectors(
                prompt_input=SEL_PROMPT_INPUT,
                submit_button=SEL_SUBMIT_BUTTON,
            ),
            error_markers=CLAUDE_ERROR_MARKERS,
            recoverable_codes=("CLAUDE_ERROR",),
            answer_timeout_s=cfg.claude_answer_timeout_s,
            parse_ws_frame=None,
        )
