from __future__ import annotations

import asyncio
import json
import re
from playwright.async_api import Page

from ai_router.adapters.gemini.selectors import (
    RATE_LIMIT_MARKERS,
    SEL_PROMPT_INPUT,
    SEL_RESPONSE_BLOCK,
    SEL_RESPONSE_INNER,
    SEL_RESPONSE_TEXT,
    SEL_SEND_CONTAINER,
)
from ai_router.browser.profile import StreamDone
from ai_router.errors import AiRouterError
from ai_router.logger import trace

# Gemini StreamGenerate end-of-turn marker: ["e", ...]
STREAM_END_RE = re.compile(r'\[\s*"e"\s*,', re.I)

_FIND_EDITOR_BODY = """
    const selectors = [
        '[data-test-id="textarea-inner"] .ql-editor[role="textbox"]',
        '.ql-editor.textarea[contenteditable="true"][role="textbox"]',
        'rich-textarea .ql-editor[contenteditable="true"][role="textbox"]',
        'div.ql-editor[contenteditable="true"][role="textbox"]',
    ];
    let el = null;
    for (const sel of selectors) {
        el = document.querySelector(sel);
        if (el) break;
    }
"""

_QUILL_CLEAR_JS = (
    """() => {"""
    + _FIND_EDITOR_BODY
    + """
    if (!el) return { ok: false, reason: 'no_editor' };
    el.focus();
    const sel = window.getSelection();
    if (sel) {
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.removeAllRanges();
        sel.addRange(range);
    }
    document.execCommand('selectAll', false, null);
    document.execCommand('delete', false, null);
    el.classList.add('ql-blank');
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return { ok: true };
}"""
)

_QUILL_TYPE_JS = (
    """(text) => {
    """
    + _FIND_EDITOR_BODY
    + """
    if (!el) return { ok: false, reason: 'no_editor', text: '' };

    const dispatchInput = () => {
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new InputEvent('input', {
            bubbles: true,
            inputType: 'insertText',
            data: text,
        }));
    };

    const writeViaDom = (value) => {
        el.classList.remove('ql-blank');
        el.innerHTML = '';
        for (const line of value.split(/\\r?\\n/)) {
            const p = document.createElement('p');
            if (line) {
                p.textContent = line;
            } else {
                p.appendChild(document.createElement('br'));
            }
            el.appendChild(p);
        }
    };

    const insertMultiline = (value) => {
        const lines = value.split(/\\r?\\n/);
        for (let i = 0; i < lines.length; i++) {
            if (i > 0) {
                if (!document.execCommand('insertLineBreak', false, null)) {
                    document.execCommand('insertParagraph', false, null);
                }
            }
            if (lines[i]) {
                document.execCommand('insertText', false, lines[i]);
            }
        }
    };

    el.focus();
    el.classList.remove('ql-blank');
    const sel = window.getSelection();
    if (sel) {
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.removeAllRanges();
        sel.addRange(range);
    }
    document.execCommand('selectAll', false, null);
    document.execCommand('delete', false, null);

    if (text.includes('\\n')) {
        insertMultiline(text);
    } else {
        const ok = document.execCommand('insertText', false, text);
        if (!ok) {
            writeViaDom(text);
        }
    }

    let written = (el.innerText || el.textContent || '');
    const expected = text.trim();
    const writtenTrim = written.trim();
    // Quill single-line mode may drop everything after the first newline.
    if (expected.includes('\\n') && writtenTrim.length < expected.length * 0.85) {
        writeViaDom(text);
        written = (el.innerText || el.textContent || '');
    }

    dispatchInput();
    written = written.trim();
    return {
        ok: written.length > 0 && written.length >= expected.length * 0.85,
        text: written,
        expected_len: expected.length,
        written_len: written.length,
    };
}"""
)

_IS_GENERATING_JS = """() => {
    const containers = document.querySelectorAll(
        'div.send-button-container.visible, div[data-test-id="send-button-container"].visible'
    );
    for (const container of containers) {
        const stopIcon = container.querySelector('gem-icon-button.stop');
        if (stopIcon && stopIcon.getAttribute('aria-disabled') !== 'true') {
            return true;
        }
        const btn = container.querySelector('button');
        if (!btn || btn.disabled) continue;
        const label = (btn.getAttribute('aria-label') || '').toLowerCase();
        if (label.includes('stop') || label.includes('dừng')) {
            return true;
        }
    }
    return false;
}"""


def braces_balanced(text: str) -> bool:
    if "{" not in text:
        return True
    return text.count("{") == text.count("}")


def is_rate_limited(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in RATE_LIMIT_MARKERS)


def is_stream_end(body: str) -> bool:
    """True when StreamGenerate payload contains Gemini's end-of-turn tag."""
    return bool(STREAM_END_RE.search(body))


def extract_stream_answer(body: str) -> str | None:
    """Best-effort answer text from a finished StreamGenerate body."""
    if not is_stream_end(body):
        return None
    candidates: list[str] = []
    for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', body):
        raw = match.group(1)
        if len(raw) < 20:
            continue
        try:
            text = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            text = raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if len(stripped) < 20:
            continue
        if stripped.startswith("http") or stripped.startswith("rc_"):
            continue
        if stripped[0] in "[{":
            continue
        if re.fullmatch(r"[\w\-.:]+", stripped):
            continue
        candidates.append(stripped)
    if not candidates:
        return None
    return max(candidates, key=len)


def parse_stream_done(status: int, body: str) -> StreamDone:
    """Gemini StreamGenerate: done when the end-of-turn ["e", ...] tag appears."""
    if is_stream_end(body):
        return StreamDone(
            done=True,
            ok=True,
            answer_text=extract_stream_answer(body),
        )
    return StreamDone(done=False, ok=False)


async def _wait_prompt_input(page: Page) -> None:
    await page.wait_for_selector(SEL_PROMPT_INPUT, state="visible", timeout=15_000)


async def clear_prompt(page: Page) -> None:
    """Clear Gemini's Quill editor via execCommand (keyboard shortcuts fail)."""
    await _wait_prompt_input(page)
    result = await page.evaluate(_QUILL_CLEAR_JS)
    if not result.get("ok"):
        raise AiRouterError("ADAPTER_ERROR", "Gemini prompt input not found")


def _prompt_fully_written(prompt: str, written: str) -> bool:
    """True when the editor text contains the substantive lines we sent."""
    expected_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if not expected_lines:
        return bool(written.strip())
    hits = sum(1 for line in expected_lines if line in written)
    return hits >= max(1, len(expected_lines) - 1)


async def type_prompt(page: Page, prompt: str) -> None:
    """Insert text into Gemini's Quill editor and fire the input events Send needs."""
    await _wait_prompt_input(page)
    result = await page.evaluate(_QUILL_TYPE_JS, prompt)
    written = result.get("text") or ""
    trace(
        "gemini_type_result",
        ok=result.get("ok"),
        written_len=len(written),
        expected_len=len(prompt.strip()),
        preview=written[:60],
    )
    if not result.get("ok") or not _prompt_fully_written(prompt, written):
        raise AiRouterError(
            "ADAPTER_ERROR",
            "Failed to type full prompt into Gemini input "
            f"(expected_len={len(prompt.strip())}, written_len={len(written)})",
        )
    await asyncio.sleep(0.2)


async def send_button_ready(page: Page) -> bool:
    """True when Gemini's Send control is present and enabled."""
    for container_sel in (
        "div.send-button-container.visible",
        'div[data-test-id="send-button-container"].visible',
        ".input-buttons-wrapper-bottom",
        ".trailing-actions-wrapper",
    ):
        container = page.locator(container_sel).last
        if await container.count() == 0:
            continue
        wrapper = container.locator("gem-icon-button.send-button.submit").first
        if await wrapper.count() > 0:
            if await wrapper.get_attribute("aria-disabled") != "true":
                return True
        for label in ("Send message", "Gửi"):
            submit = container.locator(f'button[aria-label="{label}"]').first
            if await submit.count() > 0 and not await submit.is_disabled():
                return True
        submit = container.locator('button[aria-label*="Send" i]').first
        if await submit.count() > 0 and not await submit.is_disabled():
            label = (await submit.get_attribute("aria-label") or "").lower()
            if "stop" not in label and "dừng" not in label:
                return True
    return False


async def is_stop_visible(page: Page) -> bool:
    """True while Gemini is still generating (Stop control in the send area)."""
    try:
        return bool(await page.evaluate(_IS_GENERATING_JS))
    except Exception:
        container = page.locator(SEL_SEND_CONTAINER).last
        if await container.count() == 0:
            return False
        return await container.locator("gem-icon-button.stop").count() > 0


async def read_response_snapshot(page: Page) -> tuple[int, str]:
    """Return assistant turn count and text from the latest response block."""
    blocks = page.locator(SEL_RESPONSE_BLOCK)
    count = await blocks.count()
    if count:
        last = blocks.nth(count - 1)
        inner = last.locator(SEL_RESPONSE_INNER)
        if await inner.count():
            text = (await inner.first.inner_text()).strip()
        else:
            text = (await last.inner_text()).strip()
        return count, text

    texts = page.locator(SEL_RESPONSE_TEXT)
    count = await texts.count()
    if count:
        return count, (await texts.nth(count - 1).inner_text()).strip()

    return 0, ""
