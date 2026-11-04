"""Automated DeepSeek email sign-up for ai-router.

Selectors below were captured from the live chat.deepseek.com/sign_in page
(region: "Only email registration is supported"). The flow runs *headed* so the
user can solve a Cloudflare Turnstile if one appears; the verification code is
supplied by ``get_code`` (mailtemp website or API) so this module stays
independent of where the mailbox lives.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from ai_router.adapters.deepseek.selectors import DEEPSEEK_URL, SEL_PROMPT_INPUT
from ai_router.adapters.deepseek.wait import is_challenge_visible, read_response_snapshot

SIGN_IN_URL = "https://chat.deepseek.com/sign_in"

# --- Verified sign-up form selectors -----------------------------------------
SEL_EMAIL_INPUT = (
    'input[placeholder="Email address"], '
    'input[placeholder*="email" i], '
    'input[placeholder*="Phone number / email" i]'
)
SEL_PASSWORD_INPUT = 'input[placeholder="Password"], input[type="password"]'
SEL_CONFIRM_INPUT = 'input[placeholder="Confirm password"]'
SEL_CODE_INPUT = (
    'input[placeholder="Code"], input[autocomplete="one-time-code"], input[type="tel"]'
)
SEL_TERMS_CHECKBOX = 'input[type="checkbox"]'

CodeProvider = Callable[[], Awaitable[str]]
Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


async def _logged_in(page: Page) -> bool:
    try:
        await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=2_000)
        return True
    except PlaywrightTimeout:
        return False


async def _fill(page: Page, selector: str, value: str, log: Logger, label: str) -> bool:
    loc = page.locator(selector).first
    try:
        if await loc.count() == 0:
            log(f"  [skip] {label}: field not found")
            return False
        await loc.click(timeout=3_000)
        await loc.fill(value)
        log(f"  filled {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] {label}: {exc}")
        return False


async def _click_text(page: Page, labels: list[str], log: Logger, *, button: bool = False) -> bool:
    for label in labels:
        finders = []
        if button:
            finders.append(lambda ln=label: page.get_by_role("button", name=ln, exact=True))
        finders.append(lambda ln=label: page.get_by_text(ln, exact=False))
        for finder in finders:
            loc = finder().first
            try:
                if await loc.count() == 0:
                    continue
                await loc.click(timeout=3_000)
                log(f"  clicked {label!r}")
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


async def _wait_logged_in(page: Page, log: Logger, timeout: float) -> bool:
    log(
        f"Waiting up to {timeout:.0f}s for login. If a Cloudflare check or extra "
        "step appears in the window, complete it by hand..."
    )
    deadline = asyncio.get_event_loop().time() + timeout
    warned = False
    while asyncio.get_event_loop().time() < deadline:
        if await _logged_in(page):
            return True
        if not warned and await is_challenge_visible(page):
            log("  Cloudflare challenge detected — please solve it in the window.")
            warned = True
        await asyncio.sleep(1.5)
    return False


async def deepseek_sign_up(
    page: Page,
    *,
    email: str,
    password: str,
    get_code: CodeProvider,
    log: Logger = _noop,
    login_timeout: float = 300.0,
) -> str:
    """Sign up a DeepSeek account by email. Returns the outcome string.

    "already_logged_in" | "signed_up" | "needs_manual"
    """
    await page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
    if await _logged_in(page):
        log("Already logged in — skipping sign-up.")
        return "already_logged_in"

    log("Opening sign-in page...")
    try:
        await page.goto(SIGN_IN_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
    except PlaywrightTimeout:
        pass

    # Switch to the sign-up form (reveals Confirm password + Code fields).
    await _click_text(page, ["Sign up"], log)
    await page.wait_for_timeout(1000)

    log(f"Entering email {email} ...")
    await _fill(page, SEL_EMAIL_INPUT, email, log, "email")

    if await _click_text(page, ["Send code", "获取验证码", "Gửi mã"], log):
        log("Requested code — waiting for it to arrive in the mailbox...")
        try:
            code = await get_code()
            await _fill(page, SEL_CODE_INPUT, code, log, "code")
        except Exception as exc:  # noqa: BLE001
            log(f"  [warn] code retrieval failed: {exc} (type it manually if you have it)")
    else:
        log("  [skip] 'Send code' not found — complete the email step manually.")

    log("Setting password...")
    await _fill(page, SEL_PASSWORD_INPUT, password, log, "password")
    await _fill(page, SEL_CONFIRM_INPUT, password, log, "confirm password")

    # Accept terms if a checkbox is present.
    try:
        cb = page.locator(SEL_TERMS_CHECKBOX).first
        if await cb.count() > 0 and not await cb.is_checked():
            await cb.check(timeout=2_000)
            log("  accepted terms")
    except Exception:  # noqa: BLE001
        pass

    # Submit: the in-form "Sign up" button (exact, to avoid the tab link).
    await _click_text(page, ["Sign up"], log, button=True)

    if await _wait_logged_in(page, log, login_timeout):
        log("Logged in successfully.")
        return "signed_up"
    log("Not logged in after automated attempt.")
    return "needs_manual"


async def smoke_chat(page: Page, prompt: str, log: Logger = _noop, timeout: float = 180.0) -> str:
    """Send one prompt via the DOM and return the assistant's reply text.

    A lightweight smoke test (not the full StateReducer gate): submit, wait for
    a new assistant block, then read it once the text stops growing.
    """
    from ai_router.adapters.deepseek.selectors import SEL_SUBMIT_BUTTON
    from ai_router.adapters.deepseek.wait import ensure_new_chat, is_stop_visible

    await ensure_new_chat(page)
    count_before, _ = await read_response_snapshot(page)

    inp = page.locator(SEL_PROMPT_INPUT).first
    await inp.click()
    await inp.fill("")
    await inp.type(prompt, delay=15)
    log(f"Submitting prompt: {prompt!r}")

    submit = page.locator(SEL_SUBMIT_BUTTON).first
    if await submit.count() > 0 and not await submit.is_disabled():
        await submit.click()
    else:
        await inp.press("Enter")

    deadline = asyncio.get_event_loop().time() + timeout
    last_text = ""
    stable_ticks = 0
    while asyncio.get_event_loop().time() < deadline:
        count, text = await read_response_snapshot(page)
        generating = await is_stop_visible(page)
        if count > count_before and text and not generating:
            if text == last_text:
                stable_ticks += 1
                if stable_ticks >= 4:
                    return text
            else:
                stable_ticks = 0
                last_text = text
        else:
            last_text = text
        await asyncio.sleep(0.5)
    return last_text
