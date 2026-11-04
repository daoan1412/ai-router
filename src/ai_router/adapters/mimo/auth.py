"""Xiaomi MiMo Studio (aistudio.xiaomimimo.com, MiMo models) chat for ai-router.

Chat needs a Xiaomi Account (guests are sent to the login page as soon as they
touch the composer). Answers are read from the page's own completion stream
(``/open-apis/react/chat`` SSE, via CDP): ``reasoningContent`` deltas are the
reasoning, ``content`` deltas the raw markdown/LaTeX reply. The request body
carries ``modelConfig.webSearchStatus``; it is forced to "disabled" on the way
out so answers never use web search.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from playwright.async_api import Page

from ai_router.streaming import Delta, collect

MIMO_URL = "https://aistudio.xiaomimimo.com/"
MIMO_CHAT_URL = "https://aistudio.xiaomimimo.com/#/c"  # a fresh load is a new chat
MIMO_COMPLETION_PATH = "/open-apis/react/chat"

SEL_CHAT_INPUT = "textarea"
SEL_MODEL_BUTTON = '[aria-label^="MiMo-V"][role="button"]'
SIGNED_OUT_PLACEHOLDER = "Sign in to continue"

# Picker labels (first line of the menu items).
MODEL_PRO = "MiMo-V2.6-Pro"
MODEL_FLASH = "MiMo-V2.6-Flash"

Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


class MimoSSEParser:
    """Turn MiMo Studio chat SSE payloads into ("reasoning"|"content", delta).

    Observed schema (2026-11), one JSON per ``data:`` line:
    ``dialogId`` ``{"content": "<id>"}``, then ``message`` events
    ``{"messageType": "assistant", "reasoningContent"|"content": delta}``,
    ``usage`` (token counts) and ``finish`` ``{"content": "[DONE]"}``.
    """

    def __init__(self) -> None:
        self.done = False
        self.error: str | None = None
        self.got_any = False
        self.other_types: set[str] = set()

    def feed(self, obj: Any) -> list[Delta]:
        if not isinstance(obj, dict):
            return []
        kind = obj.get("messageType")
        if kind is None:
            if obj.get("content") == "[DONE]":
                self.done = True
                return []
            code = obj.get("code")
            err = obj.get("error") or (obj.get("msg") if code not in (None, 0, 200) else None)
            if err:
                self.error = str(err)[:300]
            return []  # dialogId / usage
        self.got_any = True
        if kind != "assistant":
            self.other_types.add(str(kind))  # e.g. a tool / search message
            return []
        out: list[Delta] = []
        if obj.get("reasoningContent"):
            out.append(("reasoning", obj["reasoningContent"]))
        if obj.get("content"):
            out.append(("content", obj["content"]))
        return out


def force_no_search(body: str) -> str:
    """Return the chat request body with web search switched off."""
    try:
        data = json.loads(body)
    except ValueError:
        return body
    if not isinstance(data, dict):
        return body
    cfg = data.get("modelConfig")
    if isinstance(cfg, dict) and cfg.get("webSearchStatus") != "disabled":
        cfg["webSearchStatus"] = "disabled"
        return json.dumps(data, ensure_ascii=False)
    return body


async def _install_no_search(page: Page) -> None:
    if getattr(page, "_mimo_no_search", False):
        return

    async def handler(route) -> None:
        body = route.request.post_data
        if route.request.method == "POST" and body:
            await route.continue_(post_data=force_no_search(body))
        else:
            await route.continue_()

    await page.route(f"**{MIMO_COMPLETION_PATH}*", handler)
    page._mimo_no_search = True  # type: ignore[attr-defined]


async def accept_agreement(page: Page, log: Logger = _noop) -> None:
    """Accept the one-time "User Agreement & Privacy Policy" modal (first login)."""
    modal = page.locator("div.modal").filter(has_text="User Agreement")
    try:
        if not await modal.count() or not await modal.first.is_visible():
            return
        box = modal.locator("#agreement-checkbox")
        if await box.count() and not await box.is_checked():
            await box.click()
        await modal.get_by_text("OK", exact=True).first.click()
        await page.wait_for_timeout(800)
        log("  accepted the MiMo user agreement")
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] agreement modal: {exc}")


async def is_signed_in(page: Page, timeout: float = 10.0) -> bool:
    """Wait for the composer to settle: signed in once its placeholder stops
    saying "Sign in to continue chatting" (a cold load renders signed-out first)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            ph = await page.evaluate(
                "(s) => { const t = document.querySelector(s); return t ? t.placeholder : null; }",
                SEL_CHAT_INPUT,
            )
        except Exception:  # noqa: BLE001 - navigating
            ph = None
        if ph is not None and SIGNED_OUT_PLACEHOLDER not in ph:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await page.wait_for_timeout(500)


async def select_model(page: Page, label: str, log: Logger = _noop) -> None:
    """Pick ``label`` in the model menu (the choice persists across chats)."""
    button = page.locator(SEL_MODEL_BUTTON).first
    if await button.count() == 0:
        log("  [warn] model picker not found; staying on current model")
        return
    if await button.get_attribute("aria-label") == label:
        return
    await button.click()
    await page.wait_for_timeout(600)
    clicked = await page.evaluate(
        r"""(label) => {
            const b = [...document.querySelectorAll('button')].find(x =>
                x.offsetParent !== null && (x.innerText || '').trim().split('\n')[0].trim() === label);
            if (!b) return false;
            b.click();
            return true;
        }""",
        label,
    )
    await page.wait_for_timeout(600)
    if not clicked:
        await page.keyboard.press("Escape")
        log(f"  [warn] model {label!r} not in picker; staying on current model")
        return
    log(f"  model -> {await button.get_attribute('aria-label')}")


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

    The caller should have loaded MIMO_CHAT_URL and checked is_signed_in().
    """
    from ai_router.browser.sse_tap import CDPStreamTap, SSELineBuffer

    await accept_agreement(page, log)
    if model_label:
        await select_model(page, model_label, log)
    await _install_no_search(page)

    tap = CDPStreamTap(page, MIMO_COMPLETION_PATH)
    await tap.start()
    try:
        ta = page.locator(SEL_CHAT_INPUT).first
        await ta.click()
        await ta.fill(prompt)
        log(f"Submitting prompt: {prompt!r}")
        await ta.press("Enter")

        opened = await tap.next(open_timeout)
        if opened is None or opened[0] != "open":
            raise RuntimeError("MiMo completion stream not seen after sending the prompt")
        status = opened[1]
        lines = SSELineBuffer()
        parser = MimoSSEParser()
        raw_head = ""
        deadline = asyncio.get_event_loop().time() + timeout
        while not parser.done:
            remaining = deadline - asyncio.get_event_loop().time()
            item = await tap.next(remaining) if remaining > 0 else None
            if item is None:
                log("  [warn] MiMo stream timed out")
                break
            kind, value = item
            if kind == "text":
                if len(raw_head) < 2000:
                    raw_head += value
                for obj in lines.feed(value):
                    for d in parser.feed(obj):
                        yield d
                if parser.error:
                    raise RuntimeError(f"MiMo error: {parser.error}")
            elif kind == "end":
                for obj in lines.flush():
                    for d in parser.feed(obj):
                        yield d
                break
            elif kind == "error":
                log(f"  [warn] stream tap: {value}")

        if status >= 400 or (not parser.got_any and "messageType" not in raw_head and raw_head.strip()):
            raise RuntimeError(f"MiMo HTTP {status}: {raw_head.strip()[:300]}")
        if parser.error:
            raise RuntimeError(f"MiMo error: {parser.error}")
        if parser.other_types:
            log(f"  [warn] MiMo sent non-assistant messages: {sorted(parser.other_types)}")
    finally:
        await tap.stop()


async def chat(
    page: Page, prompt: str, *, log: Logger = _noop, timeout: float = 300.0,
    model_label: str | None = None,
) -> tuple[str, str | None]:
    """Send one prompt; return (answer, reasoning-or-None)."""
    return await collect(chat_stream(page, prompt, log=log, timeout=timeout, model_label=model_label))


# --- Account: Xiaomi Account sign-up / sign-in (captured live, 2026-11) ------
#
# MiMo "Sign in" → account.xiaomi.com login (sid=xiaomichatbot). Sign up tab →
# Region / Email / password ×2 / agree → Next → Google reCAPTCHA ("I'm not a
# robot", the user ticks it — headed) → /register/email/verify: 6-digit code
# from the email into ``input[name=ticket]`` → Submit → back on MiMo signed in
# → one-time "User Agreement & Privacy Policy" modal.

SEL_EMAIL = 'input[name="email"]'
SEL_ACCOUNT = 'input[name="account"]'
SEL_PASSWORD = 'input[name="password"]'
SEL_REPASSWORD = 'input[name="repassword"]'
SEL_AGREE = "label.ant-checkbox-wrapper"
SEL_CODE = 'input[name="ticket"]'
CODE_SUBJECT_HINT = "Xiaomi"

CodeProvider = Callable[[], Awaitable[str]]

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


async def _open_account_page(page: Page, log: Logger) -> bool:
    """From MiMo, follow "Sign in" to the Xiaomi Account login page.
    Returns False if already signed in."""
    await page.goto(MIMO_CHAT_URL, wait_until="domcontentloaded")
    if await is_signed_in(page):
        return False
    if await _click_button(page, "Decline All"):  # cookie banner re-renders the page
        await page.wait_for_timeout(1500)
    for attempt in range(4):
        if "account.xiaomi.com" in page.url:
            break
        try:
            if attempt % 2 == 0:
                await page.locator("button:visible", has_text="Sign in").last.click(timeout=5000)
            else:  # touching the composer also redirects a guest to the login
                await page.locator(f"{SEL_CHAT_INPUT}:visible").first.click(timeout=5000)
            # "commit": the Xiaomi page's load event can take very long.
            await page.wait_for_url("**account.xiaomi.com/**", timeout=12_000, wait_until="commit")
        except Exception:  # noqa: BLE001
            log(f"  [warn] still on {page.url[:60]} (attempt {attempt + 1})")
    if "account.xiaomi.com" not in page.url:
        raise RuntimeError("MiMo: did not reach the Xiaomi Account login page")
    await page.wait_for_timeout(2000)
    await _click_button(page, "Accept cookies")
    log("  Xiaomi Account page open")
    return True


async def _finish_account_flow(
    page: Page, *, get_code: CodeProvider | None, log: Logger, timeout: float
) -> bool:
    """After submitting a Xiaomi form: let the user tick reCAPTCHA, fill an
    e-mail code when asked, and wait until MiMo is signed in."""
    await page.bring_to_front()  # the mailbox tab may be on top
    deadline = asyncio.get_event_loop().time() + timeout
    told_captcha = code_done = sent = False
    while asyncio.get_event_loop().time() < deadline:
        await page.wait_for_timeout(1500)
        url = page.url
        if "aistudio.xiaomimimo.com" in url:
            if await is_signed_in(page, timeout=8):
                await accept_agreement(page, log)
                return True
            continue
        if not told_captcha and await page.locator('iframe[src*="recaptcha"]:visible').count():
            log("  Tick Google's reCAPTCHA (\"I'm not a robot\") in the Xiaomi window...")
            told_captcha = True
        if not sent and "/identity/" in url and await _click_button(page, "Send"):
            # New-device check ("Account Authentication"): ask for the e-mail code.
            log("  new-device check: requested an e-mail code")
            sent = True
            continue
        code_box = page.locator(f"{SEL_CODE}:visible")
        if not code_done and await code_box.count():
            if get_code is None:
                log("  [warn] Xiaomi asks for an e-mail code; enter it in the window")
                code_done = True
                continue
            log("  e-mail code requested — waiting for it in the mailbox...")
            code = await get_code()
            await code_box.first.fill(code)
            await _click_button(page, "Submit")
            code_done = True
    return False


async def mimo_sign_up(
    page: Page,
    *,
    email: str,
    password: str,
    get_code: CodeProvider,
    log: Logger = _noop,
    captcha_timeout: float = 420.0,
) -> str:
    """Create a Xiaomi Account by e-mail and sign MiMo in.
    Returns "already_signed_in" | "signed_up" | "needs_manual"."""
    if not await _open_account_page(page, log):
        log("Already signed in — skipping sign-up.")
        return "already_signed_in"
    await page.get_by_text("Sign up", exact=True).first.click()
    await page.wait_for_selector(SEL_EMAIL, timeout=20_000)
    await page.locator(SEL_EMAIL).fill(email)
    await page.locator(SEL_PASSWORD).fill(password)
    await page.locator(SEL_REPASSWORD).fill(password)
    await page.locator(SEL_AGREE).first.click()
    log(f"  sign-up form filled for {email}")
    await _click_button(page, "Next")
    if not await _finish_account_flow(page, get_code=get_code, log=log, timeout=captcha_timeout):
        log("  [warn] not signed in: " + (await _body(page))[:200].replace("\n", " | "))
        return "needs_manual"
    log("Signed up and signed in.")
    return "signed_up"


async def mimo_sign_in(
    page: Page,
    *,
    email: str,
    password: str,
    get_code: CodeProvider | None = None,
    log: Logger = _noop,
    captcha_timeout: float = 420.0,
) -> str:
    """Sign in with a saved Xiaomi Account. Returns "already_signed_in" |
    "signed_in" | "needs_manual"."""
    if not await _open_account_page(page, log):
        return "already_signed_in"
    await page.wait_for_selector(SEL_ACCOUNT, timeout=20_000)
    await page.locator(SEL_ACCOUNT).fill(email)
    await page.locator(SEL_PASSWORD).fill(password)
    agree = page.locator(SEL_AGREE).first
    if await agree.count() and not await agree.locator("input").is_checked():
        await agree.click()
    await page.locator('button[type="submit"]').first.click()
    if not await _finish_account_flow(page, get_code=get_code, log=log, timeout=captcha_timeout):
        log("  [warn] not signed in: " + (await _body(page))[:200].replace("\n", " | "))
        return "needs_manual"
    return "signed_in"
