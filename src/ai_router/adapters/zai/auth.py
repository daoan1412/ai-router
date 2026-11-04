"""Z.ai (chat.z.ai, GLM models) chat for ai-router.

chat.z.ai is an Open WebUI fork. Guests can chat without an account, but only
with GLM-5.3-Flash — GLM-5.3 / GLM-5.2 are disabled in the picker until you
sign in. Answers are read from the page's own completion stream
(``/api/v2/chat/completions`` SSE, via CDP): ``phase: "thinking"`` deltas are
the reasoning, ``phase: "answer"`` the raw markdown/LaTeX reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from playwright.async_api import Page

from ai_router.adapters.base import ModelLockedError  # noqa: F401 - raised by select_model
from ai_router.streaming import Delta, collect

ZAI_URL = "https://chat.z.ai/"
ZAI_AUTH_URL = "https://chat.z.ai/auth"
ZAI_COMPLETION_PATH = "/api/v2/chat/completions"

SEL_CHAT_INPUT = "#chat-input"
SEL_MODEL_BUTTON = "button.modelSelectorButton"

# Picker labels (exact text of the menu items).
MODEL_FLASH = "GLM-5.3-Flash"
MODEL_53 = "GLM-5.3"
MODEL_52 = "GLM-5.2"

Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


class ZaiSSEParser:
    """Turn chat.z.ai completion SSE payloads into ("reasoning"|"content", delta).

    Observed schema (2026-11): ``{"type": "chat:completion", "data": {...}}``
    with ``data.phase`` "thinking" | "answer" carrying ``delta_content`` token
    deltas, then "other" (usage) and "done" (``done: true``).
    """

    def __init__(self) -> None:
        self.done = False
        self.error: str | None = None
        self.got_any = False
        self.other_phases: set[str] = set()

    def feed(self, obj: Any) -> list[Delta]:
        if not isinstance(obj, dict):
            return []
        err = obj.get("error") or obj.get("detail")
        data = obj.get("data") if isinstance(obj.get("data"), dict) else None
        if data is not None:
            err = err or data.get("error")
        if err:
            self.error = str(err)[:300]
            return []
        if data is None or obj.get("type") not in (None, "chat:completion"):
            return []
        self.got_any = True
        phase = data.get("phase")
        text = data.get("delta_content") or ""
        out: list[Delta] = []
        if phase == "thinking":
            if text:
                out.append(("reasoning", text))
        elif phase == "answer":
            if text:
                out.append(("content", text))
        elif phase in ("other", "done", None):
            pass
        else:
            self.other_phases.add(phase)  # e.g. a tool / search phase
        if data.get("done") or phase == "done":
            self.done = True
        return out


async def guest_or_signed_in(page: Page) -> bool:
    """Whether the chat composer is usable (guests can chat too)."""
    try:
        await page.wait_for_selector(SEL_CHAT_INPUT, timeout=10_000)
        return True
    except Exception:  # noqa: BLE001
        return False


async def is_signed_in(page: Page) -> bool:
    """True when the header has no visible "Sign in" button (i.e. not a guest)."""
    try:
        return await page.evaluate(
            r"""() => ![...document.querySelectorAll('button,a')].some(b =>
                b.offsetParent !== null && (b.innerText || '').trim() === 'Sign in')"""
        )
    except Exception:  # noqa: BLE001
        return False


async def dismiss_overlays(page: Page, log: Logger = _noop) -> None:
    """Close onboarding overlays that swallow clicks on a new account: the
    driver.js feature tour (its × button) and promo popups ("Maybe Later")."""
    for _ in range(6):
        clicked = await page.evaluate(
            r"""() => {
                const vis = e => e && e.offsetParent !== null;
                const pop = document.querySelector('.driver-popover');
                if (pop) {
                    const x = pop.querySelector('.driver-popover-close-btn');
                    if (x) { x.click(); return 'tour'; }
                }
                const want = ['Maybe Later', 'Not now', 'Skip', 'Got it'];
                const b = [...document.querySelectorAll('button')]
                    .find(b => vis(b) && want.includes((b.innerText || '').trim()));
                if (b) { b.click(); return (b.innerText || '').trim(); }
                return document.querySelector('.driver-overlay') ? 'overlay' : null;
            }"""
        )
        if clicked is None:
            return
        if clicked == "overlay":
            await page.keyboard.press("Escape")
        log(f"  dismissed: {clicked}")
        await page.wait_for_timeout(700)


async def select_model(page: Page, label: str, log: Logger = _noop) -> None:
    """Pick ``label`` in the model menu; ModelLockedError if it's disabled."""
    button = page.locator(SEL_MODEL_BUTTON).first
    if await button.count() == 0:
        log("  [warn] model picker not found; staying on current model")
        return
    if (await button.inner_text()).strip() == label:
        return
    # DOM click: an onboarding overlay would swallow a pointer click.
    await page.evaluate("(s) => document.querySelector(s).click()", SEL_MODEL_BUTTON)
    await page.wait_for_timeout(600)
    state = await page.evaluate(
        r"""(label) => {
            const b = [...document.querySelectorAll('button')].find(x =>
                x.offsetParent !== null && !x.classList.contains('modelSelectorButton')
                && (x.innerText || '').trim().split('\n')[0].trim() === label);
            if (!b) return 'missing';
            if (b.disabled || b.getAttribute('aria-disabled') === 'true') return 'disabled';
            b.click();
            return 'clicked';
        }""",
        label,
    )
    await page.wait_for_timeout(600)
    if state != "clicked":
        await page.keyboard.press("Escape")
        if state == "disabled":
            raise ModelLockedError(
                f"Z.ai model {label!r} needs a signed-in account "
                "(guests only get GLM-5.3-Flash). Run `ai-router browser provision-zai`."
            )
        log(f"  [warn] model {label!r} not in picker; staying on current model")
        return
    now = (await button.inner_text()).strip()
    log(f"  model -> {now}")


async def chat_stream(
    page: Page,
    prompt: str,
    *,
    log: Logger = _noop,
    timeout: float = 300.0,
    model_label: str | None = None,
    open_timeout: float = 30.0,
) -> AsyncIterator[Delta]:
    """Send one prompt on a fresh chat; yield ("reasoning"|"content", delta).

    The caller should have loaded ZAI_URL (a fresh page is a new chat).
    """
    from ai_router.browser.sse_tap import CDPStreamTap, SSELineBuffer

    if not await guest_or_signed_in(page):
        raise RuntimeError("Z.ai chat input not found")
    if model_label:
        await select_model(page, model_label, log)

    tap = CDPStreamTap(page, ZAI_COMPLETION_PATH)
    await tap.start()
    try:
        ta = page.locator(SEL_CHAT_INPUT).first
        await ta.click()
        await ta.fill(prompt)
        log(f"Submitting prompt: {prompt!r}")
        await ta.press("Enter")

        opened = await tap.next(open_timeout)
        if opened is None or opened[0] != "open":
            raise RuntimeError("Z.ai completion stream not seen after sending the prompt")
        status = opened[1]
        lines = SSELineBuffer()
        parser = ZaiSSEParser()
        raw_head = ""
        deadline = asyncio.get_event_loop().time() + timeout
        while not parser.done:
            remaining = deadline - asyncio.get_event_loop().time()
            item = await tap.next(remaining) if remaining > 0 else None
            if item is None:
                log("  [warn] Z.ai stream timed out")
                break
            kind, value = item
            if kind == "text":
                if len(raw_head) < 2000:
                    raw_head += value
                for obj in lines.feed(value):
                    for d in parser.feed(obj):
                        yield d
                if parser.error:
                    raise RuntimeError(f"Z.ai error: {parser.error}")
            elif kind == "end":
                for obj in lines.flush():
                    for d in parser.feed(obj):
                        yield d
                break
            elif kind == "error":
                log(f"  [warn] stream tap: {value}")

        if status >= 400 or (not parser.got_any and raw_head.strip()):
            raise RuntimeError(f"Z.ai HTTP {status}: {raw_head.strip()[:300]}")
        if parser.error:
            raise RuntimeError(f"Z.ai error: {parser.error}")
        if parser.other_phases:
            log(f"  [warn] Z.ai used non-answer phases: {sorted(parser.other_phases)}")
    finally:
        await tap.stop()


async def chat(
    page: Page, prompt: str, *, log: Logger = _noop, timeout: float = 300.0,
    model_label: str | None = None,
) -> tuple[str, str | None]:
    """Send one prompt; return (answer, reasoning-or-None)."""
    return await collect(chat_stream(page, prompt, log=log, timeout=timeout, model_label=model_label))


# --- Account: sign-up / sign-in (selectors captured live, 2026-11) -----------
#
# /auth → "Continue with Email" → (Sign up: Name/Email/Password | Sign in:
# Email/Password) → Aliyun drag-puzzle ("Click to start verification", the
# user drags it — headed) → "Create Account" → /auth/verify ("we sent a
# verification link") → link in the email → "Complete Registration" (set the
# password again) → signed in. Buttons ignore pointer clicks right after the
# form swaps, so they're clicked through the DOM.

SEL_NAME = 'input[placeholder="Enter Your Full Name"]'
SEL_EMAIL = 'input[type="email"]'
SEL_PASSWORD = 'input[type="password"]'
VERIFY_TEXT = "Click to start verification"
PUZZLE_TEXT = "Please drag the slider"
VERIFY_LINK_PART = "/auth/verify_email?token="

LinkProvider = Callable[[], Awaitable[str]]

_JS_CLICK_BUTTON = r"""(t) => {
    const b = [...document.querySelectorAll('button')].find(x =>
        x.offsetParent !== null && (x.innerText || '').replace(/\s+/g, ' ').trim() === t);
    if (b) { b.click(); return true; }
    return false;
}"""


async def _click_button(page: Page, text: str) -> bool:
    return bool(await page.evaluate(_JS_CLICK_BUTTON, text))


async def _body(page: Page) -> str:
    try:
        return await page.evaluate("() => document.body.innerText")
    except Exception:  # noqa: BLE001
        return ""


async def signed_in_now(page: Page) -> bool:
    """Signed in = off the /auth pages, composer present, no "Sign in" button."""
    if "/auth" in page.url:
        return False
    try:
        if await page.locator(SEL_CHAT_INPUT).count() == 0:
            return False
    except Exception:  # noqa: BLE001
        return False
    return await is_signed_in(page)


async def _open_email_form(page: Page, log: Logger) -> None:
    await page.goto(ZAI_AUTH_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await _click_button(page, "Continue with Email")
    await page.wait_for_selector(SEL_EMAIL, timeout=15_000)
    log("  email form open")


async def _solve_captcha(page: Page, log: Logger, timeout: float) -> bool:
    """Open the Aliyun puzzle and wait for the user to drag it (headed)."""
    widget = page.get_by_text(VERIFY_TEXT, exact=True).locator("visible=true").first
    try:
        # The Aliyun widget renders a moment after the form; count() alone
        # would miss it and submit without the captcha.
        await widget.wait_for(state="visible", timeout=15_000)
    except Exception:  # noqa: BLE001
        log("  [warn] captcha widget did not appear; submitting without it")
        return True
    await page.bring_to_front()  # the mailbox tab may be on top
    box = await widget.bounding_box()
    if box:
        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    log("  Solve the Aliyun puzzle (drag the slider) in the Z.ai window...")
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await page.wait_for_timeout(1000)
        body = await _body(page)
        if VERIFY_TEXT not in body and PUZZLE_TEXT not in body:
            log("  puzzle solved")
            return True
    log("  [warn] puzzle not solved in time")
    return False


async def _wait_signed_in(page: Page, timeout: float) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await signed_in_now(page):
            return True
        await page.wait_for_timeout(1000)
    return False


async def zai_sign_up(
    page: Page,
    *,
    email: str,
    password: str,
    name: str,
    get_link: LinkProvider,
    log: Logger = _noop,
    captcha_timeout: float = 300.0,
) -> str:
    """Create a Z.ai account. Returns "already_signed_in" | "signed_up" | "needs_manual"."""
    await page.goto(ZAI_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await dismiss_overlays(page, log)
    if await signed_in_now(page):
        log("Already signed in — skipping sign-up.")
        return "already_signed_in"

    await _open_email_form(page, log)
    await _click_button(page, "Sign up")
    await page.wait_for_selector(SEL_NAME, timeout=15_000)
    await page.locator(SEL_NAME).fill(name)
    await page.locator(SEL_EMAIL).fill(email)
    await page.locator(SEL_PASSWORD).first.fill(password)
    log(f"  sign-up form filled for {email}")
    if not await _solve_captcha(page, log, captcha_timeout):
        return "needs_manual"
    await _click_button(page, "Create Account")
    try:
        await page.wait_for_url("**/auth/verify**", timeout=20_000)
    except Exception:  # noqa: BLE001
        log("  [warn] no verification page: " + (await _body(page))[:200].replace("\n", " | "))
        return "needs_manual"
    log("  verification email sent — waiting for the link...")

    link = await get_link()
    await page.goto(link, wait_until="domcontentloaded")
    await page.wait_for_selector(SEL_PASSWORD, timeout=20_000)
    pw = page.locator(SEL_PASSWORD)
    for i in range(await pw.count()):  # "Password" + "Confirm Password"
        await pw.nth(i).fill(password)
    user = page.locator('input[type="text"]').first
    if await user.count() and not await user.input_value():
        await user.fill(name)
    await _click_button(page, "Complete Registration")
    if not await _wait_signed_in(page, 40):
        log("  [warn] not signed in after registration")
        return "needs_manual"
    await dismiss_overlays(page, log)
    log("Signed up and signed in.")
    return "signed_up"


async def zai_sign_in(
    page: Page,
    *,
    email: str,
    password: str,
    log: Logger = _noop,
    captcha_timeout: float = 300.0,
) -> str:
    """Sign in with a saved email/password. Returns "already_signed_in" |
    "signed_in" | "needs_manual"."""
    await page.goto(ZAI_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await dismiss_overlays(page, log)
    if await signed_in_now(page):
        return "already_signed_in"
    await _open_email_form(page, log)
    await page.locator(SEL_EMAIL).fill(email)
    await page.locator(SEL_PASSWORD).first.fill(password)
    if not await _solve_captcha(page, log, captcha_timeout):
        return "needs_manual"
    await _click_button(page, "Sign in")
    if not await _wait_signed_in(page, 30):
        log("  [warn] not signed in: " + (await _body(page))[:200].replace("\n", " | "))
        return "needs_manual"
    await dismiss_overlays(page, log)
    return "signed_in"
