"""Automated Qwen (chat.qwen.ai) email sign-up + smoke chat for ai-router.

Selectors captured live from chat.qwen.ai. Qwen sign-up is passwordless: email
-> "Continue" -> an Alibaba drag-puzzle CAPTCHA (CloakBrowser cannot auto-solve,
so the flow is headed and waits for the user) -> a 6-cell verification code ->
"Continue" -> an age gate (birth year) -> logged in. The code is supplied by
``get_code`` so this module stays independent of the mailbox source.

Chat: answer and reasoning are read from the page's own completion stream
(``/api/v2/chat/completions`` SSE, via CDP) — raw markdown/LaTeX plus the
"thinking_summary" steps. Reading the rendered page (answer bubble + "Thinking
and Search" panel) is only a fallback. ``disable_tools`` switches off web
search / tools / memory in the account settings.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from itertools import zip_longest
from typing import Any

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from ai_router.streaming import Delta, collect, prefix_delta

QWEN_URL = "https://chat.qwen.ai/"
QWEN_REGISTER_URL = "https://chat.qwen.ai/auth?mode=register"

SEL_EMAIL = 'input[name="email"], input[placeholder*="Email" i]'
SEL_CODE_CELL = "input.qwenchat-verification-code-input-cell"
SEL_CHAT_INPUT = "textarea.message-input-textarea, textarea"
SEL_ASSISTANT_MSG = ".qwen-chat-message:not(.qwen-chat-message-user)"
SEL_ASSISTANT_TEXT = ".qwen-markdown-text-node"
# The final answer sits in a "phase-answer" block of the assistant bubble.
SEL_ANSWER = ".response-message-content.phase-answer"
SEL_STOP = '[class*="stop" i], [aria-label*="Stop" i]'
SEL_NEW_CHAT = 'button:has-text("New Chat"), a:has-text("New Chat"), [aria-label*="New Chat" i]'
# Model picker: the trigger button shows the current model name (e.g. "Qwen3.8-Max").
SEL_MODEL_TRIGGER = ".wms-trigger"

# Reasoning lives in the "Thinking and Search" side panel, which only exists
# after the bubble's status card is clicked. The bubble card's title class is
# qwen-chat-status-card-title while thinking and
# qwen-chat-thinking-status-card-title-text once done.
SEL_STATUS_CARD_TITLE = ".qwen-chat-status-card-title, .qwen-chat-thinking-status-card-title-text"
SEL_PANEL = ".qwen-chat-thinking-and-sources"
SEL_PANEL_CLOSE = ".qwen-chat-thinking-and-sources-header-close"
SEL_PANEL_THINKING = ".qwen-chat-thinking-and-sources-content-thinking-container"

# Element -> text with KaTeX math as its TeX source ($...$). innerText would
# print each formula twice (rendered glyphs + the hidden accessible copy).
_CLEAN_TEXT_JS = r"""(el) => {
    const tex = k => {
        const a = k.querySelector('annotation[encoding="application/x-tex"]');
        return a ? a.textContent.trim() : (k.textContent || '');
    };
    const c = el.cloneNode(true);
    c.querySelectorAll('.katex-display').forEach(k => k.replaceWith('$$' + tex(k) + '$$'));
    c.querySelectorAll('.katex').forEach(k => k.replaceWith('$' + tex(k) + '$'));
    c.querySelectorAll('br').forEach(b => b.replaceWith('\n'));
    c.querySelectorAll('p,div,li,h1,h2,h3,h4,h5,h6,pre,blockquote,tr').forEach(b => b.append('\n'));
    c.querySelectorAll('td,th').forEach(b => b.append('\t'));
    return c.textContent.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}"""

# Thinking container -> [{title, body}] per reasoning step.
_STEPS_JS = (
    "(root) => { const clean = "
    + _CLEAN_TEXT_JS
    + r"""; return [...root.querySelectorAll('.qwen-chat-thinking-status-card')].map(card => {
        const t = card.querySelector('.qwen-chat-thinking-status-card-title-text');
        const b = card.querySelector('.qwen-markdown');
        return {title: t ? t.innerText.trim() : '', body: b ? clean(b) : ''};
    }); }"""
)

PUZZLE_MARKERS = ("access verification", "complete the puzzle", "verify that you are")
AGE_MARKERS = ("confirm your age", "what year were you born")

CodeProvider = Callable[[], Awaitable[str]]
Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


async def _body(page: Page) -> str:
    try:
        return (await page.locator("body").inner_text()).lower()
    except Exception:  # noqa: BLE001
        return ""


SEL_USER_MENU = ".user-menu-btn"  # account button; only rendered when signed in


async def _logged_in_now(page: Page) -> bool:
    if "/auth" in page.url:
        return False
    try:
        if await page.locator(SEL_USER_MENU).count() > 0:
            return True
        # Fallback if the account button's class changes: composer present and
        # no Log in / Sign up buttons (the composer also shows when logged out).
        body = await _body(page)
        if "log in" in body or "sign up" in body:
            return False
        return await page.locator(SEL_CHAT_INPUT).count() > 0
    except Exception:  # noqa: BLE001
        return False


async def is_logged_in(page: Page, timeout: float = 10.0) -> bool:
    """Whether the page is signed in, waiting up to ``timeout`` for the session.

    Right after a load — especially the first page of a freshly launched
    browser — Qwen briefly renders the logged-out header (Log in / Sign up)
    until it has resolved the session cookie, so a single instant check gives
    false negatives. Pass timeout=0 for an instant check inside a poll loop.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if await _logged_in_now(page):
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.5)


async def _clear_age_gate(page: Page, log: Logger, birth_year: str = "2000") -> None:
    body = await _body(page)
    if not any(m in body for m in AGE_MARKERS):
        return
    log("  age gate — entering birth year")
    try:
        yi = page.locator('input[type="number"], input[type="text"]').first
        if await yi.count():
            await yi.click()
            await yi.fill(birth_year)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(10):
        b = page.get_by_role("button", name="Continue", exact=True).first
        if await b.count() and not await b.is_disabled():
            await b.click()
            break
        await page.wait_for_timeout(500)
    await page.wait_for_timeout(2000)


async def qwen_sign_up(
    page: Page,
    *,
    email: str,
    get_code: CodeProvider,
    log: Logger = _noop,
    login_timeout: float = 300.0,
) -> str:
    """Sign in to Qwen with an email + verification code.

    Returns "already_logged_in" | "signed_up" | "needs_manual".
    """
    await page.goto(QWEN_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)
    if await is_logged_in(page):
        await _clear_age_gate(page, log)
        log("Already logged in — skipping sign-up.")
        return "already_logged_in"

    log("Opening Qwen sign-up page...")
    try:
        await page.goto(QWEN_REGISTER_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
    except PlaywrightTimeout:
        pass

    log(f"Entering email {email} ...")
    try:
        await page.locator(SEL_EMAIL).first.fill(email)
        await page.get_by_role("button", name="Continue", exact=True).first.click()
        log("  clicked Continue")
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] email step: {exc}")

    # The drag-puzzle appears here. Wait for the 6 code cells, letting the user
    # solve the puzzle by hand in the headed window.
    log("If a 'Drag to complete the puzzle' check appears, solve it in the window...")
    deadline = asyncio.get_event_loop().time() + login_timeout
    while asyncio.get_event_loop().time() < deadline:
        if await page.locator(SEL_CODE_CELL).count() >= 6:
            break
        await page.wait_for_timeout(1500)
    else:
        log("  [warn] code cells never appeared (puzzle unsolved?)")
        return "needs_manual"

    log("Code boxes ready — fetching the code from the mailbox...")
    try:
        code = (await get_code()).strip()
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] code retrieval failed: {exc}")
        return "needs_manual"

    cells = page.locator(SEL_CODE_CELL)
    for idx in range(min(6, len(code))):
        try:
            await cells.nth(idx).fill(code[idx])
        except Exception:  # noqa: BLE001
            break
    log(f"  entered code {code}")
    await page.wait_for_timeout(800)

    # Submit once Continue enables.
    for _ in range(20):
        b = page.get_by_role("button", name="Continue", exact=True).first
        if await b.count() and not await b.is_disabled():
            await b.click()
            log("  clicked Continue (code)")
            break
        await page.wait_for_timeout(500)

    # Settle: wait for login, clearing the age gate if it blocks.
    end = asyncio.get_event_loop().time() + 60
    while asyncio.get_event_loop().time() < end:
        await _clear_age_gate(page, log)
        if await is_logged_in(page, timeout=0):
            log("Logged in successfully.")
            return "signed_up"
        await page.wait_for_timeout(1500)
    log("Not logged in after automated attempt.")
    return "needs_manual"


async def _new_chat(page: Page) -> None:
    """Start a fresh conversation so each request is independent."""
    try:
        btn = page.locator(SEL_NEW_CHAT).first
        if await btn.count():
            await btn.click(timeout=3000)
            await page.wait_for_timeout(800)
    except Exception:  # noqa: BLE001
        pass


async def select_model(page: Page, keyword: str, log: Logger = _noop) -> bool:
    """Switch the Qwen model via the picker. keyword matches the dropdown label
    ('Plus', 'Max', 'Flash'). Returns True if the model is (now) selected."""
    trigger = page.locator(SEL_MODEL_TRIGGER).first
    if await trigger.count() == 0:
        log("  [warn] model picker not found; staying on current model")
        return False
    current = (await trigger.inner_text()).strip()
    if keyword.lower() in current.lower():
        return True
    try:
        await trigger.click(timeout=3000)
        await page.wait_for_timeout(600)
        item = page.get_by_text(re.compile(rf"Qwen[\w.\-]*{re.escape(keyword)}", re.I)).first
        if await item.count() == 0:
            log(f"  [warn] model {keyword!r} not in picker; staying on {current!r}")
            # close the menu
            await page.keyboard.press("Escape")
            return False
        await item.click()
        await page.wait_for_timeout(800)
        now = (await trigger.inner_text()).strip()
        ok = keyword.lower() in now.lower()
        log(f"  model -> {now}" if ok else f"  [warn] model switch to {keyword!r} unconfirmed ({now})")
        return ok
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] model switch failed: {exc}")
        return False


def format_steps(steps: list[dict]) -> str:
    """Join panel steps into reasoning text: "title\\nbody" blocks.

    Title-only cards are skipped — that is the trailing "Thinking completed"
    marker, or a step whose body hasn't started streaming yet (it is picked up
    on a later tick, appended at the end, so the text still only grows).
    """
    parts = []
    for step in steps:
        title = (step.get("title") or "").strip()
        body = (step.get("body") or "").strip()
        if not body:
            continue
        parts.append(f"{title}\n{body}" if title else body)
    return "\n\n".join(parts)


def _first_new_bubble(page: Page, before: int):
    # With Qwen's A/B "which response do you prefer?" two bubbles appear; use
    # the first one for both answer and reasoning so they match.
    return page.locator(SEL_ASSISTANT_MSG).nth(before)


async def _read_new_answer(page: Page, before: int) -> str:
    """Answer text of this turn's bubble (phase-answer blocks only).

    The bubble also holds the status card, whose label would otherwise leak
    into the content.
    """
    if await page.locator(SEL_ASSISTANT_MSG).count() <= before:
        return ""
    answers = _first_new_bubble(page, before).locator(SEL_ANSWER)
    try:
        n = await answers.count()
        texts = [await answers.nth(i).evaluate(_CLEAN_TEXT_JS) for i in range(n)]
    except Exception:  # noqa: BLE001
        return ""
    return "\n\n".join(t for t in texts if t)


async def _close_panel(page: Page) -> None:
    try:
        btn = page.locator(SEL_PANEL_CLOSE).first
        if await btn.count():
            await btn.click(timeout=2000)
            await page.wait_for_timeout(300)
    except Exception:  # noqa: BLE001
        pass


async def _open_panel(page: Page, before: int) -> bool | None:
    """Click this turn's status card so the reasoning panel is rendered.

    Returns None when there is no card to click yet, else whether the panel
    is now open.
    """
    if await page.locator(SEL_ASSISTANT_MSG).count() <= before:
        return None
    title = _first_new_bubble(page, before).locator(SEL_STATUS_CARD_TITLE).first
    try:
        if await title.count() == 0:
            return None
        # Click the title (left side) — the card's right side has a "Skip"
        # button that would cut the thinking short.
        await title.click(timeout=2000)
        await page.wait_for_timeout(400)
    except Exception:  # noqa: BLE001
        return False
    return await page.locator(SEL_PANEL).count() > 0


async def _read_reasoning(page: Page) -> str:
    container = page.locator(SEL_PANEL_THINKING).first
    try:
        if await container.count() == 0:
            return ""
        return format_steps(await container.evaluate(_STEPS_JS))
    except Exception:  # noqa: BLE001
        return ""


QWEN_COMPLETION_PATH = "/api/v2/chat/completions"


class QwenSSEParser:
    """Turn Qwen's completion SSE payloads into ("reasoning"|"content", delta).

    Observed schema (chat.qwen.ai, 2026-11): ``choices[0].delta`` has a
    ``phase``:
      - "thinking_summary": ``extra.summary_title.content`` /
        ``extra.summary_thought.content`` are *cumulative* lists of all steps
        so far; rebuilt each time and streamed as an append-only delta.
      - "answer": ``content`` is a real token delta (raw markdown + LaTeX);
        ``status: "finished"`` ends the reply.
    Locks onto the first response_id, so a second A/B response is ignored.
    """

    def __init__(self) -> None:
        self.response_id: str | None = None
        self.done = False
        self.error: str | None = None
        self.got_any = False
        self.other_phases: set[str] = set()
        self._sent_reasoning = ""

    def feed(self, obj: Any) -> list[Delta]:
        if not isinstance(obj, dict):
            return []
        if obj.get("error") or obj.get("success") is False:
            self.error = json.dumps(obj.get("error") or obj, ensure_ascii=False)[:300]
            return []
        choices = obj.get("choices")
        if not choices:
            return []
        rid = obj.get("response_id")
        if rid:
            if self.response_id is None:
                self.response_id = rid
            elif rid != self.response_id:
                return []
        self.got_any = True
        delta = (choices[0] or {}).get("delta") or {}
        phase = delta.get("phase")
        out: list[Delta] = []
        if phase == "thinking_summary":
            extra = delta.get("extra") or {}
            titles = (extra.get("summary_title") or {}).get("content") or []
            thoughts = (extra.get("summary_thought") or {}).get("content") or []
            if titles or thoughts:
                steps = [
                    {"title": t, "body": b}
                    for t, b in zip_longest(titles, thoughts, fillvalue="")
                ]
                self._sent_reasoning, d = prefix_delta(self._sent_reasoning, format_steps(steps))
                if d:
                    out.append(("reasoning", d))
        elif phase in ("think", "thinking", "reasoning"):
            # Raw chain-of-thought deltas, if a model streams those instead.
            if delta.get("content"):
                out.append(("reasoning", delta["content"]))
        elif phase == "answer":
            if delta.get("content"):
                out.append(("content", delta["content"]))
            if delta.get("status") == "finished":
                self.done = True
        elif phase:
            self.other_phases.add(phase)  # e.g. a tool/search phase
        return out


async def _submit(page: Page, prompt: str, log: Logger) -> int:
    """Type and send the prompt; return the assistant-bubble count before it."""
    before = await page.locator(SEL_ASSISTANT_MSG).count()
    ta = page.locator(SEL_CHAT_INPUT).first
    await ta.click()
    await ta.fill("")
    await ta.type(prompt, delay=15)
    log(f"Submitting prompt: {prompt!r}")
    await ta.press("Enter")
    return before


async def chat_stream(
    page: Page,
    prompt: str,
    *,
    log: Logger = _noop,
    timeout: float = 180.0,
    new_chat: bool = False,
    model_keyword: str | None = None,
    poll: float = 0.3,
    open_timeout: float = 20.0,
) -> AsyncIterator[Delta]:
    """Send one prompt; yield ("reasoning"|"content", delta) as Qwen generates.

    Reads the page's own completion stream (raw markdown/LaTeX answer + the
    thinking summary) via CDP. If that request isn't seen within
    ``open_timeout``, falls back to reading the rendered page. model_keyword
    ('Plus'/'Max'/'Flash') is applied after New Chat, which resets the picker.
    """
    from ai_router.browser.sse_tap import CDPStreamTap, SSELineBuffer

    await _clear_age_gate(page, log)
    if await page.locator(SEL_CHAT_INPUT).count() == 0:
        log("  no chat input found (not logged in)")
        return
    await _close_panel(page)  # never read a previous turn's reasoning (DOM fallback)
    if new_chat:
        await _new_chat(page)
    if model_keyword:
        await select_model(page, model_keyword, log)

    tap = CDPStreamTap(page, QWEN_COMPLETION_PATH)
    await tap.start()
    try:
        before = await _submit(page, prompt, log)
        opened = await tap.next(open_timeout)
        if opened is None or opened[0] != "open":
            log("  completion stream not seen; reading the rendered page instead")
            async for d in _dom_stream(page, before, timeout=timeout, poll=poll):
                yield d
            return

        status = opened[1]
        lines = SSELineBuffer()
        parser = QwenSSEParser()
        raw_head = ""  # for error reporting when the body isn't SSE
        deadline = asyncio.get_event_loop().time() + timeout
        while not parser.done:
            remaining = deadline - asyncio.get_event_loop().time()
            item = await tap.next(remaining) if remaining > 0 else None
            if item is None:
                log("  [warn] Qwen stream timed out")
                break
            kind, value = item
            if kind == "text":
                if len(raw_head) < 2000:
                    raw_head += value
                for obj in lines.feed(value):
                    for d in parser.feed(obj):
                        yield d
                if parser.error:
                    raise RuntimeError(f"Qwen error: {parser.error}")
            elif kind == "end":
                for obj in lines.flush():
                    for d in parser.feed(obj):
                        yield d
                break
            elif kind == "error":
                log(f"  [warn] stream tap: {value}")

        if status >= 400 or (not parser.got_any and raw_head.strip()):
            raise RuntimeError(f"Qwen HTTP {status}: {raw_head.strip()[:300]}")
        if parser.error:
            raise RuntimeError(f"Qwen error: {parser.error}")
        if parser.other_phases:
            log(f"  [warn] Qwen used non-answer phases: {sorted(parser.other_phases)}")
    finally:
        await tap.stop()


async def _dom_stream(
    page: Page, before: int, *, timeout: float, poll: float
) -> AsyncIterator[Delta]:
    """Fallback: stream from the rendered page (answer bubble + thinking panel)."""
    sent_content = ""
    sent_reasoning = ""
    panel_open = False
    clicks = 0
    last_text = ""
    stable = 0
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        generating = await page.locator(SEL_STOP).count() > 0
        # Only count real clicks: before the card exists there is nothing to
        # click, and clicking repeatedly could toggle the panel shut.
        if not panel_open and clicks < 5:
            result = await _open_panel(page, before)
            if result is not None:
                clicks += 1
                panel_open = result
        if panel_open:
            sent_reasoning, d = prefix_delta(sent_reasoning, await _read_reasoning(page))
            if d:
                yield ("reasoning", d)
        text = await _read_new_answer(page, before)
        sent_content, d = prefix_delta(sent_content, text)
        if d:
            yield ("content", d)
        if text and not generating:
            if text == last_text:
                stable += 1
                if stable >= 4:
                    break
            else:
                stable = 0
                last_text = text
        else:
            stable = 0
        await asyncio.sleep(poll)

    # Final reconcile. The status card stays after completion, so a turn whose
    # thinking was too quick to catch still gets its reasoning.
    if not panel_open:
        panel_open = bool(await _open_panel(page, before))
    if panel_open:
        _, d = prefix_delta(sent_reasoning, await _read_reasoning(page))
        if d:
            yield ("reasoning", d)
    _, d = prefix_delta(sent_content, await _read_new_answer(page, before))
    if d:
        yield ("content", d)


async def chat(
    page: Page,
    prompt: str,
    *,
    log: Logger = _noop,
    timeout: float = 180.0,
    new_chat: bool = False,
    model_keyword: str | None = None,
) -> tuple[str, str | None]:
    """Send one prompt; return (answer, reasoning-or-None). See chat_stream."""
    return await collect(
        chat_stream(
            page,
            prompt,
            log=log,
            timeout=timeout,
            new_chat=new_chat,
            model_keyword=model_keyword,
        )
    )


async def smoke_chat(page: Page, prompt: str, log: Logger = _noop, timeout: float = 180.0) -> str:
    """Back-compat wrapper used by provision-qwen: returns just the answer text."""
    answer, _ = await chat(page, prompt, log=log, timeout=timeout)
    return answer


# Every visible switch with its row label: [{label, checked}], in DOM order.
_SWITCH_ROWS_JS = r"""() => [...document.querySelectorAll('[role="switch"]')].map(s => {
    let row = s;
    for (let i = 0; row && i < 5 && (row.innerText || '').trim().length < 4; i++) row = row.parentElement;
    return {label: ((row && row.innerText) || '').trim().split('\n')[0],
            checked: s.getAttribute('aria-checked') === 'true',
            visible: s.offsetParent !== null};
})"""


async def disable_tools(page: Page, log: Logger = _noop) -> list[str]:
    """Turn off Qwen's tools and memory: Settings → Personalization, every switch.

    That tab holds web search, web scraping, image search/generation/editing,
    code interpreter and the memory switches, all on by default. This is an
    account-level setting. "Advanced" (the tools master switch) goes last so
    the individual switches are still visible while we flip them.
    Returns the labels that were switched off; leaves the page on QWEN_URL.
    """
    turned_off: list[str] = []
    try:
        await page.locator(".user-menu-btn").first.click(timeout=3000)
        await page.wait_for_timeout(600)
        await page.get_by_text("Settings", exact=True).first.click(timeout=3000)
        await page.wait_for_timeout(1200)
        await page.get_by_text("Personalization", exact=True).first.click(timeout=3000)
        await page.wait_for_timeout(1000)

        for _ in range(30):
            rows = await page.evaluate(_SWITCH_ROWS_JS)
            on = [(i, r) for i, r in enumerate(rows) if r["visible"] and r["checked"]]
            if not on:
                break
            not_master = [x for x in on if x[1]["label"].lower() != "advanced"]
            idx, row = (not_master or on)[0]
            await page.locator('[role="switch"]').nth(idx).click(timeout=3000)
            await page.wait_for_timeout(500)
            after = await page.evaluate(_SWITCH_ROWS_JS)
            if idx < len(after) and after[idx]["checked"]:
                log(f"  [warn] switch {row['label']!r} did not turn off")
                break
            turned_off.append(row["label"])
            log(f"  turned off: {row['label']}")
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] could not update Qwen tool settings: {exc}")
    finally:
        # Escape doesn't reliably close the settings modal; reload instead.
        await page.goto(QWEN_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
    return turned_off
