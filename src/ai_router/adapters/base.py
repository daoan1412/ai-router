from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol

from playwright.async_api import Page

if TYPE_CHECKING:
    from ai_router.browser.profile import ProviderProfile
    from ai_router.config import AppConfig


class SessionStatus(str, Enum):
    LOGGED_IN = "logged_in"
    LOGGED_OUT = "logged_out"
    UNKNOWN = "unknown"


ProviderStatus = Literal["available", "coming_soon"]


class ModelLockedError(RuntimeError):
    """The requested model is disabled in the provider's picker (needs an account)."""


class ProviderAdapter(Protocol):
    id: str
    name: str
    keywords: list[str]
    status: ProviderStatus

    async def check_session(self, page: Page) -> SessionStatus: ...
    async def open_new_chat(self, page: Page) -> None: ...
    def build_profile(self, cfg: AppConfig) -> ProviderProfile: ...
