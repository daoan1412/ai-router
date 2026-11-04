from __future__ import annotations

from playwright.async_api import Page

from ai_router.browser.pwcompat import PlaywrightTimeout

from ai_router.adapters.base import SessionStatus
from ai_router.adapters.gemini.planner import GeminiPlanner
from ai_router.adapters.gemini.selectors import (
    GEMINI_ERROR_MARKERS,
    GEMINI_URL,
    SEL_PROMPT_INPUT,
    SEL_SIGN_IN,
    SEL_SUBMIT_BUTTON,
    STREAM_GENERATE_RE,
)
from ai_router.adapters.gemini.wait import (
    clear_prompt,
    is_rate_limited,
    is_stop_visible,
    parse_stream_done,
    read_response_snapshot,
    send_button_ready,
    type_prompt,
)
from ai_router.browser.profile import ProviderProfile, ProviderSelectors
from ai_router.config import AppConfig


class GeminiAdapter:
    id = "gemini"
    name = "Gemini"
    keywords = ["gemini", "@gemini", "google gemini"]
    status = "available"

    async def check_session(self, page: Page) -> SessionStatus:
        return await self.ensure_page_ready(page)

    async def ensure_page_ready(self, page: Page) -> SessionStatus:
        """Open a fresh chat and verify login."""
        await page.goto(GEMINI_URL, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=15_000)
            return SessionStatus.LOGGED_IN
        except PlaywrightTimeout:
            if await page.locator(SEL_SIGN_IN).count() > 0:
                return SessionStatus.LOGGED_OUT
            return SessionStatus.UNKNOWN

    async def open_new_chat(self, page: Page) -> None:
        await page.goto(GEMINI_URL, wait_until="domcontentloaded")

    def build_profile(self, cfg: AppConfig) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.id,
            stream_url_re=STREAM_GENERATE_RE,
            parse_stream_done=parse_stream_done,
            is_stop_visible=is_stop_visible,
            read_response_snapshot=read_response_snapshot,
            is_rate_limited=is_rate_limited,
            submit_ready=send_button_ready,
            planner=GeminiPlanner(),
            selectors=ProviderSelectors(
                prompt_input=SEL_PROMPT_INPUT,
                submit_button=SEL_SUBMIT_BUTTON,
            ),
            error_markers=GEMINI_ERROR_MARKERS,
            recoverable_codes=("GEMINI_ERROR",),
            answer_timeout_s=None,
            type_prompt=type_prompt,
            clear_prompt=clear_prompt,
        )
