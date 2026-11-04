from __future__ import annotations

from playwright.async_api import Page

from ai_router.browser.pwcompat import PlaywrightTimeout

from ai_router.adapters.base import SessionStatus
from ai_router.adapters.chatgpt.planner import ChatGPTPlanner
from ai_router.adapters.chatgpt.selectors import (
    CHATGPT_CONVERSATION_RE,
    CHATGPT_ERROR_MARKERS,
    CHATGPT_URL,
    SEL_LOGIN,
    SEL_PROMPT_INPUT,
    SEL_SUBMIT_BUTTON,
)
from ai_router.adapters.chatgpt.stream import parse_stream_done, parse_ws_frame
from ai_router.adapters.chatgpt.wait import (
    is_rate_limited,
    is_stop_visible,
    read_response_snapshot,
    submit_ready,
)
from ai_router.browser.profile import ProviderProfile, ProviderSelectors
from ai_router.config import AppConfig


class ChatGPTAdapter:
    id = "chatgpt"
    name = "ChatGPT"
    keywords: list[str] = ["chatgpt", "gpt", "@chatgpt"]
    status = "available"

    async def check_session(self, page: Page) -> SessionStatus:
        return await self.ensure_page_ready(page)

    async def ensure_page_ready(self, page: Page) -> SessionStatus:
        """Open a fresh chat and verify login."""
        await page.goto(CHATGPT_URL, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=15_000)
            return SessionStatus.LOGGED_IN
        except PlaywrightTimeout:
            if await page.locator(SEL_LOGIN).count() > 0:
                return SessionStatus.LOGGED_OUT
            return SessionStatus.UNKNOWN

    async def open_new_chat(self, page: Page) -> None:
        await page.goto(CHATGPT_URL, wait_until="domcontentloaded")

    def build_profile(self, cfg: AppConfig) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.id,
            stream_url_re=CHATGPT_CONVERSATION_RE,
            parse_stream_done=parse_stream_done,
            is_stop_visible=is_stop_visible,
            read_response_snapshot=read_response_snapshot,
            is_rate_limited=is_rate_limited,
            submit_ready=submit_ready,
            planner=ChatGPTPlanner(),
            selectors=ProviderSelectors(
                prompt_input=SEL_PROMPT_INPUT,
                submit_button=SEL_SUBMIT_BUTTON,
            ),
            error_markers=CHATGPT_ERROR_MARKERS,
            recoverable_codes=("CHATGPT_ERROR", "CHATGPT_INCOMPLETE"),
            answer_timeout_s=cfg.chatgpt_answer_timeout_s,
            parse_ws_frame=parse_ws_frame,
        )
