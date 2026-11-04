from __future__ import annotations

from playwright.async_api import Page

from ai_router.browser.pwcompat import PlaywrightTimeout

from ai_router.adapters.base import SessionStatus
from ai_router.adapters.deepseek.planner import DeepSeekPlanner
from ai_router.adapters.deepseek.selectors import (
    DEEPSEEK_COMPLETION_RE,
    DEEPSEEK_ERROR_MARKERS,
    DEEPSEEK_URL,
    SEL_LOGIN,
    SEL_PROMPT_INPUT,
    SEL_SUBMIT_BUTTON,
)
from ai_router.adapters.deepseek.stream import parse_stream_done
from ai_router.adapters.deepseek.wait import (
    ensure_active_chat_view,
    ensure_new_chat,
    is_challenge_visible,
    is_generating_started,
    is_rate_limited,
    is_stop_visible,
    read_response_snapshot,
    submit_ready,
)
from ai_router.browser.profile import ProviderProfile, ProviderSelectors
from ai_router.config import AppConfig


class DeepSeekAdapter:
    id = "deepseek"
    name = "DeepSeek"
    keywords: list[str] = ["deepseek", "@deepseek"]
    status = "available"

    async def check_session(self, page: Page) -> SessionStatus:
        return await self.ensure_page_ready(page)

    async def ensure_page_ready(self, page: Page) -> SessionStatus:
        await page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
        if await is_challenge_visible(page):
            return SessionStatus.UNKNOWN
        try:
            await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=15_000)
            return SessionStatus.LOGGED_IN
        except PlaywrightTimeout:
            if await page.locator(SEL_LOGIN).count() > 0:
                return SessionStatus.LOGGED_OUT
            return SessionStatus.UNKNOWN

    async def open_new_chat(self, page: Page) -> None:
        await ensure_new_chat(page)

    def build_profile(self, cfg: AppConfig) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.id,
            stream_url_re=DEEPSEEK_COMPLETION_RE,
            parse_stream_done=parse_stream_done,
            is_stop_visible=is_stop_visible,
            read_response_snapshot=read_response_snapshot,
            is_rate_limited=is_rate_limited,
            submit_ready=submit_ready,
            planner=DeepSeekPlanner(),
            selectors=ProviderSelectors(
                prompt_input=SEL_PROMPT_INPUT,
                submit_button=SEL_SUBMIT_BUTTON,
            ),
            error_markers=DEEPSEEK_ERROR_MARKERS,
            recoverable_codes=("DEEPSEEK_ERROR",),
            answer_timeout_s=cfg.deepseek_answer_timeout_s,
            generating_start_timeout_s=120.0,
            parse_ws_frame=None,
            on_new_chat=ensure_new_chat,
            after_submit=ensure_active_chat_view,
            is_generating_started=is_generating_started,
            is_challenge_visible=is_challenge_visible,
        )
