"""Minimal OpenAI-compatible HTTP API in front of the browser providers.

Exposes ``GET /v1/models`` and ``POST /v1/chat/completions`` so OpenAI-SDK
clients can talk to the browser-automated providers (deepseek, qwen, zai, mimo).
The assistant's reasoning is returned in ``message.reasoning_content`` (the
DeepSeek-compatible field): DeepSeek's DeepThink block (R1 ids), Qwen's
"Thinking and Search" panel steps (whenever Qwen thinks), and Z.ai / MiMo
thinking. Web search and tools are kept off: DeepSeek's Search toggle per
request, Qwen's tool/memory switches in account settings once per server run,
MiMo's ``webSearchStatus`` in every request.

Answers come from driving a real browser, so a request takes as long as the
model does, and requests to the same provider are serialized (one tab each).
With ``stream: true`` the reply is streamed as the page renders it: the DOM is
polled (~300 ms) and each newly appended piece of text is sent as a chunk
(``delta.reasoning_content`` first, then ``delta.content``).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ai_router.config import AppConfig
from ai_router.streaming import READY, Delta, collect


@dataclass(frozen=True)
class Target:
    """What an OpenRouter-style model id maps to in the browser UI."""

    provider: str  # "deepseek" | "qwen" | "zai" | "mimo"
    # Model picker choice: Qwen keyword ("Plus"/"Max"/"Flash"), Z.ai label
    # ("GLM-5.3-Flash"/"GLM-5.3"/"GLM-5.2") or MiMo label ("MiMo-V2.6-Pro"/
    # "MiMo-V2.6-Flash"); None for DeepSeek (one web model).
    model_choice: str | None
    deep_think: bool  # enable DeepSeek DeepThink


# OpenRouter-style ids we advertise in /v1/models, mapped to the web UI model.
# (DeepSeek web has one model + a DeepThink toggle; Qwen and Z.ai have three.)
MODEL_MAP: dict[str, Target] = {
    "deepseek/deepseek-chat": Target("deepseek", None, False),
    "deepseek/deepseek-r1-0528": Target("deepseek", None, True),
    "qwen/qwen3.7-plus": Target("qwen", "Plus", False),
    "qwen/qwen3.8-max-0902": Target("qwen", "Max", False),
    "qwen/qwen3.8-omni-flash": Target("qwen", "Flash", False),
    "z-ai/glm-5.3-flash": Target("zai", "GLM-5.3-Flash", False),
    "z-ai/glm-5.3": Target("zai", "GLM-5.3", False),
    "z-ai/glm-5.2": Target("zai", "GLM-5.2", False),
    "xiaomi/mimo-v2.6-pro": Target("mimo", "MiMo-V2.6-Pro", False),
    "xiaomi/mimo-v2.6-flash": Target("mimo", "MiMo-V2.6-Flash", False),
}


class NotLoggedInError(RuntimeError):
    def __init__(self, provider: str) -> None:
        super().__init__(
            f"{provider} is not logged in. Run `ai-router browser provision-{provider}` "
            f"(or `ai-router browser login`) first."
        )
        self.provider = provider


# --- Pure helpers (unit-tested; no browser) ----------------------------------

def resolve_target(model: str | None) -> Target | None:
    """Resolve an OpenRouter-style model id to a Target.

    Exact ids from MODEL_MAP win; otherwise any ``deepseek/*``, ``qwen/*`` or
    ``z-ai/*`` slug (or a bare ``deepseek``/``qwen``/``glm``) is resolved by its
    keywords, so the API accepts OpenRouter's whole family and routes to the
    nearest web model.
    """
    if not model:
        return None
    key = model.strip().lower()
    if key in MODEL_MAP:
        return MODEL_MAP[key]
    vendor = key.split("/", 1)[0]
    if vendor == "deepseek" or key.startswith("deepseek"):
        think = any(t in key for t in ("r1", "reason", "think"))
        return Target("deepseek", None, think)
    if vendor == "qwen" or key.startswith("qwen"):
        if "max" in key:
            kw = "Max"
        elif "omni" in key or "flash" in key:
            kw = "Flash"
        else:
            kw = "Plus"  # plus, or any other qwen slug → default web model
        return Target("qwen", kw, False)
    if vendor in ("z-ai", "zai", "zhipu") or key.startswith(("glm", "z-ai", "zai")):
        if "flash" in key or "air" in key:
            label = "GLM-5.3-Flash"
        elif "5.2" in key:
            label = "GLM-5.2"
        else:
            label = "GLM-5.3"  # glm-5.3, glm-5, glm-latest, older glm-4.x → best web model
        return Target("zai", label, False)
    if vendor == "xiaomi" or key.startswith("mimo"):
        label = "MiMo-V2.6-Flash" if "flash" in key else "MiMo-V2.6-Pro"
        return Target("mimo", label, False)
    return None


def resolve_model(model: str | None) -> str | None:
    """Provider id for a model, or None if unsupported (back-compat helper)."""
    target = resolve_target(model)
    return target.provider if target else None


def wants_reasoning(model: str | None) -> bool:
    """Whether to enable DeepSeek's DeepThink for this model id."""
    target = resolve_target(model)
    return bool(target and target.deep_think)


def _content_to_text(content: Any) -> str:
    """OpenAI content may be a string or a list of parts; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def messages_to_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI chat messages into a single prompt for a fresh web chat.

    One lone user message is sent as-is; otherwise roles are labelled so the
    provider sees the system instructions and prior turns.
    """
    norm = [(m.get("role", "user"), _content_to_text(m.get("content"))) for m in messages]
    norm = [(r, c) for r, c in norm if c.strip()]
    if not norm:
        return ""
    users = [c for r, c in norm if r == "user"]
    if len(norm) == 1 and norm[0][0] == "user":
        return norm[0][1]
    if len(users) == 1 and all(r in ("user", "system") for r, _ in norm):
        system = "\n\n".join(c for r, c in norm if r == "system")
        return f"{system}\n\n{users[0]}".strip()
    label = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{label.get(r, r.title())}: {c}" for r, c in norm)


def build_completion(model: str, content: str, reasoning: str | None) -> dict:
    """OpenAI chat.completion response object (non-streaming)."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def delta_for(item: Delta) -> dict:
    """Map a provider delta to an OpenAI chunk 'delta' object."""
    kind, text = item
    return {"reasoning_content": text} if kind == "reasoning" else {"content": text}


def build_chunk(
    chunk_id: str,
    created: int,
    model: str,
    delta: dict,
    *,
    finish_reason: str | None = None,
) -> dict:
    """OpenAI chat.completion.chunk object."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def format_sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def build_models() -> dict:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "ai-router"}
            for name in MODEL_MAP
        ],
    }


def error_body(message: str, type_: str = "invalid_request_error", code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


# --- Browser-backed provider manager -----------------------------------------

class ProviderChatManager:
    """Owns the persistent browser context and one tab per provider."""

    def __init__(self, cfg: AppConfig, *, headless: bool = False) -> None:
        self.cfg = cfg
        self.headless = headless
        self._ctx = None
        self._pages: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Qwen's web search / tools / memory are account settings: switch them
        # off once per server run, on the first Qwen request.
        self._qwen_tools_checked = False

    async def start(self) -> None:
        from ai_router.browser.cloak import launch_persistent_context_async

        self.cfg.profile_dir.mkdir(parents=True, exist_ok=True)
        self._ctx = await launch_persistent_context_async(
            str(self.cfg.profile_dir), headless=self.headless
        )

    async def stop(self) -> None:
        if self._ctx is not None:
            try:
                await self._ctx.close()
            except Exception:  # noqa: BLE001
                pass
            self._ctx = None

    async def _page(self, provider: str):
        page = self._pages.get(provider)
        if page is None or page.is_closed():
            page = await self._ctx.new_page()
            self._pages[provider] = page
        return page

    async def stream(
        self,
        provider: str,
        prompt: str,
        *,
        deep_think: bool = False,
        model_choice: str | None = None,
    ) -> AsyncIterator[Delta]:
        """Yield READY, then ("content"|"reasoning", delta) while the provider generates.

        Holds the provider's lock for the whole generation, so one tab serves
        one request at a time. NotLoggedInError / ModelLockedError are raised
        before READY, so a caller that waits for READY can still answer with a
        proper HTTP error.
        """
        async with self._locks[provider]:
            page = await self._page(provider)
            if provider == "deepseek":
                from ai_router.adapters.base import SessionStatus
                from ai_router.adapters.deepseek.adapter import DeepSeekAdapter
                from ai_router.adapters.deepseek.auth import chat_stream

                if await DeepSeekAdapter().ensure_page_ready(page) != SessionStatus.LOGGED_IN:
                    raise NotLoggedInError("deepseek")
                gen = chat_stream(
                    page,
                    prompt,
                    timeout=float(self.cfg.deepseek_answer_timeout_s),
                    new_chat=True,
                    deep_think=deep_think,
                )
            elif provider == "qwen":
                from ai_router.adapters.qwen.auth import (
                    QWEN_URL,
                    _clear_age_gate,
                    chat_stream,
                    disable_tools,
                    is_logged_in,
                )
                from ai_router.logger import trace

                await page.goto(QWEN_URL, wait_until="domcontentloaded")
                await page.wait_for_timeout(1500)
                await _clear_age_gate(page, lambda _m: None)
                if not await is_logged_in(page):
                    raise NotLoggedInError("qwen")
                if not self._qwen_tools_checked:
                    off = await disable_tools(page, lambda m: trace("qwen_tools", msg=m))
                    trace("qwen_tools_off", switched=",".join(off) or "(already off)")
                    self._qwen_tools_checked = True
                gen = chat_stream(
                    page, prompt, timeout=300.0, new_chat=True, model_keyword=model_choice
                )
            elif provider == "zai":
                from ai_router.adapters.zai.auth import (
                    ZAI_URL,
                    chat_stream,
                    dismiss_overlays,
                    guest_or_signed_in,
                    select_model,
                )

                # A fresh load of the home page is a new chat.
                await page.goto(ZAI_URL, wait_until="domcontentloaded")
                await page.wait_for_timeout(1500)
                await dismiss_overlays(page)
                if not await guest_or_signed_in(page):
                    raise NotLoggedInError("zai")
                if model_choice:
                    await select_model(page, model_choice)  # ModelLockedError for guests
                gen = chat_stream(page, prompt, timeout=300.0)
            elif provider == "mimo":
                from ai_router.adapters.mimo.auth import (
                    MIMO_CHAT_URL,
                    accept_agreement,
                    chat_stream,
                    is_signed_in,
                    select_model,
                )

                await page.goto(MIMO_CHAT_URL, wait_until="domcontentloaded")
                if not await is_signed_in(page):
                    raise NotLoggedInError("mimo")
                await accept_agreement(page)
                if model_choice:
                    await select_model(page, model_choice)
                gen = chat_stream(page, prompt, timeout=300.0)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            yield READY
            async for item in gen:
                yield item

    async def complete(
        self,
        provider: str,
        prompt: str,
        *,
        deep_think: bool = False,
        model_choice: str | None = None,
    ) -> tuple[str, str | None]:
        return await collect(
            self.stream(provider, prompt, deep_think=deep_think, model_choice=model_choice)
        )


def create_app(cfg: AppConfig, *, headless: bool = False, keepalive_s: float = 15.0):
    """Build the Starlette ASGI app.

    keepalive_s: while a stream waits for the next token, send an SSE comment
    this often so clients/proxies don't drop an idle connection.
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    manager = ProviderChatManager(cfg, headless=headless)

    async def models(_request: Request) -> JSONResponse:
        return JSONResponse(build_models())

    async def chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(error_body("Invalid JSON body"), status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(error_body("Body must be a JSON object"), status_code=400)

        target = resolve_target(body.get("model"))
        if target is None:
            return JSONResponse(
                error_body(
                    f"Unsupported model {body.get('model')!r}. See /v1/models "
                    "(OpenRouter-style ids under deepseek/*, qwen/*, z-ai/* and xiaomi/*).",
                    code="model_not_found",
                ),
                status_code=404,
            )
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return JSONResponse(error_body("'messages' must be a non-empty array"), status_code=400)
        prompt = messages_to_prompt(messages)
        if not prompt:
            return JSONResponse(error_body("No text content in 'messages'"), status_code=400)

        def _fail(exc: Exception) -> JSONResponse:
            from ai_router.adapters.zai.auth import ModelLockedError

            if isinstance(exc, NotLoggedInError):
                return JSONResponse(
                    error_body(str(exc), type_="authentication_error", code="not_logged_in"),
                    status_code=401,
                )
            if isinstance(exc, ModelLockedError):
                return JSONResponse(
                    error_body(str(exc), type_="authentication_error", code="model_requires_login"),
                    status_code=401,
                )
            return JSONResponse(
                error_body(f"{target.provider} chat failed: {exc}", type_="api_error"),
                status_code=502,
            )

        model = body.get("model")
        opts = {"deep_think": target.deep_think, "model_choice": target.model_choice}

        if body.get("stream"):
            from starlette.responses import StreamingResponse

            agen = manager.stream(target.provider, prompt, **opts)
            # Advance to READY (login checked) before sending headers, so a
            # not-logged-in / driver failure is a real HTTP error — but don't
            # wait for the first token: Qwen can think silently for minutes.
            try:
                await agen.__anext__()
            except StopAsyncIteration:
                pass
            except Exception as exc:  # noqa: BLE001
                await agen.aclose()
                return _fail(exc)
            return StreamingResponse(
                _sse_events(model, agen, keepalive_s=keepalive_s),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            content, reasoning = await manager.complete(target.provider, prompt, **opts)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        return JSONResponse(build_completion(model, content, reasoning))

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    return Starlette(
        routes=[
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


KEEPALIVE = ": keep-alive\n\n"  # SSE comment line; clients ignore it
_END = object()


async def _sse_events(model: str, agen: AsyncIterator[Delta], *, keepalive_s: float = 15.0):
    """OpenAI-style SSE: role chunk, one chunk per DOM delta, finish, [DONE].

    While waiting for the next delta, emits a keep-alive comment every
    ``keepalive_s``. Always closes ``agen`` so the provider lock is released,
    including when the client disconnects mid-stream.
    """

    async def _next():
        try:
            return await agen.__anext__()
        except StopAsyncIteration:
            return _END

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    pending: asyncio.Task | None = None
    try:
        yield format_sse(build_chunk(chunk_id, created, model, {"role": "assistant"}))
        while True:
            if pending is None:
                pending = asyncio.ensure_future(_next())
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield KEEPALIVE
                continue
            item, pending = pending.result(), None
            if item is _END:
                break
            if item[0] in ("content", "reasoning"):
                yield format_sse(build_chunk(chunk_id, created, model, delta_for(item)))
    except Exception:  # noqa: BLE001 - headers already sent; end the stream cleanly
        pass
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except BaseException:  # noqa: BLE001 - CancelledError or the driver's error
                pass
        await agen.aclose()
    yield format_sse(build_chunk(chunk_id, created, model, {}, finish_reason="stop"))
    yield "data: [DONE]\n\n"


def run_openai_server(host: str, port: int, *, headless: bool = False) -> None:
    import uvicorn

    from ai_router.config import load_config

    cfg = load_config()
    app = create_app(cfg, headless=headless)
    uvicorn.run(app, host=host, port=port)
