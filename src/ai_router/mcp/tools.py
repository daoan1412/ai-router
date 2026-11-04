from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from ai_router.browser.pwcompat import PlaywrightError

from ai_router.adapters.base import SessionStatus
from ai_router.adapters.registry import ProviderRegistry, build_registry
from ai_router.browser.events import page_id_of
from ai_router.browser.manager import BrowserManager
from ai_router.browser.page_queue import AskJob, PageQueueRegistry
from ai_router.browser.page_router import PageRouter
from ai_router.browser.page_worker import PageWorker
from ai_router.browser.profile import ProviderProfile
from ai_router.config import AppConfig, load_config
from ai_router.errors import (
    AiRouterError,
    BrowserClosedError,
    ProviderNotReadyError,
    TimeoutError_,
)
from ai_router.router.resolve import resolve_provider
from ai_router.logger import trace


@dataclass
class AppState:
    config: AppConfig
    registry: ProviderRegistry
    browser: BrowserManager
    page_queues: PageQueueRegistry
    page_workers: dict[str, PageWorker]
    profiles: dict[str, ProviderProfile]
    page_router: PageRouter


def create_app_state(config: AppConfig | None = None) -> AppState:
    cfg = config or load_config()
    registry = build_registry()
    # No hasattr guard: an "available" adapter missing build_profile should
    # fail fast at startup (AttributeError) rather than surface later as a
    # confusing ProviderNotReadyError at ask time.
    profiles = {
        a.id: a.build_profile(cfg)
        for a in registry.list_all()
        if a.status == "available"
    }
    browser = BrowserManager(cfg)
    return AppState(
        config=cfg,
        registry=registry,
        browser=browser,
        page_queues=PageQueueRegistry(),
        page_workers={},
        profiles=profiles,
        page_router=PageRouter(browser, cfg.max_pages),
    )


def _prune_dead_workers(state: AppState) -> None:
    """Drop workers whose tab has closed.

    Pinned tabs get dropped and recreated (context resets, crashes) with a
    fresh page id each time. Without pruning, the dead workers pile up in
    page_workers and eventually trip BROWSER_BUSY even though only a few tabs
    are actually live. The getattr guard keeps this a no-op for test doubles
    that don't expose a real page.
    """
    for dead_id, worker in list(state.page_workers.items()):
        page = getattr(worker, "page", None)
        if page is None:
            continue
        try:
            closed = page.is_closed()
        except Exception:
            closed = False
        if closed:
            worker.stop()
            state.page_workers.pop(dead_id, None)
            state.page_queues.drop(page)
            trace(
                "worker_pruned",
                page_id=dead_id,
                worker_count=len(state.page_workers),
            )


def ensure_worker(
    state: AppState, page, default_provider: str | None = None
) -> PageWorker:
    _prune_dead_workers(state)
    pid = page_id_of(page)
    if pid not in state.page_workers:
        if len(state.page_workers) >= state.config.max_pages:
            raise AiRouterError("BROWSER_BUSY", "Maximum page workers reached")
        queue = state.page_queues.queue_for(page)
        worker = PageWorker(
            page,
            queue,
            state.config,
            state.profiles,
            default_provider or state.config.default_provider,
        )
        worker.start()
        state.page_workers[pid] = worker
        trace("worker_created", page_id=pid, worker_count=len(state.page_workers))
    return state.page_workers[pid]


async def handle_ask(
    state: AppState,
    *,
    prompt: str,
    provider: str | None,
    mcp_session_id: str | None,
) -> dict:
    adapter, routing_reason = resolve_provider(
        state.registry, provider, default=state.config.default_provider
    )
    if adapter.status == "coming_soon":
        raise ProviderNotReadyError(adapter.id)

    try:
        page = await state.page_router.page_for(adapter)
        worker = ensure_worker(state, page, default_provider=adapter.id)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        profile = state.profiles.get(adapter.id)
        timeout_s = (
            profile.answer_timeout_s
            if profile is not None and profile.answer_timeout_s
            else float(state.config.answer_timeout_s)
        )
        job = AskJob(
            job_id=str(uuid.uuid4()),
            mcp_session_id=mcp_session_id,
            prompt=prompt,
            provider_id=adapter.id,
            future=future,
            timeout_s=timeout_s,
        )
        page_id = page_id_of(page)
        trace(
            "ask_enqueue",
            job_id=job.job_id,
            mcp_session_id=mcp_session_id,
            page_id=page_id,
            provider=adapter.id,
            prompt=prompt[:80],
            queue_depth=state.page_queues.queue_for(page).depth(),
            running_job_id=getattr(worker, "_running_job_id", None),
        )
        await worker.enqueue(job)
        trace("ask_await", job_id=job.job_id, mcp_session_id=mcp_session_id)
        try:
            answer = await asyncio.wait_for(future, timeout=job.timeout_s)
        except asyncio.TimeoutError:
            future.cancel()
            trace("ask_timeout", job_id=job.job_id, mcp_session_id=mcp_session_id)
            raise TimeoutError_() from None
        trace(
            "ask_return",
            job_id=job.job_id,
            mcp_session_id=mcp_session_id,
            answer_len=len(answer),
        )
    except PlaywrightError as exc:
        if "closed" in str(exc).lower():
            raise BrowserClosedError() from exc
        raise AiRouterError("ADAPTER_ERROR", str(exc)) from exc

    return {
        "answer": answer,
        "provider": adapter.id,
        "routing_reason": routing_reason,
    }


async def handle_ask_multi(
    state: AppState,
    *,
    prompt: str,
    providers: list[str] | None = None,
    strategy: str | None = None,
    mcp_session_id: str | None,
) -> dict:
    chosen = strategy or state.config.parallel_default_strategy
    if chosen not in ("all", "first", "longest"):
        raise AiRouterError("INVALID_STRATEGY", f"Unknown strategy: {chosen}")
    ids = list(
        providers
        or state.config.parallel_default_providers
        or [a.id for a in state.registry.list_all() if a.status == "available"]
    )
    if not ids:
        raise AiRouterError("NO_PROVIDERS", "No providers available for ask_multi")

    async def _one(pid: str) -> tuple[dict, float]:
        started = time.monotonic()
        try:
            res = await handle_ask(
                state, prompt=prompt, provider=pid, mcp_session_id=mcp_session_id
            )
            elapsed = time.monotonic() - started
            entry = {
                "provider": res["provider"],
                "answer": res["answer"],
                "duration_s": round(elapsed, 1),
                "routing_reason": res["routing_reason"],
                "error": None,
            }
        except AiRouterError as exc:
            elapsed = time.monotonic() - started
            trace("ask_multi_provider_error", provider=pid, code=exc.code)
            entry = {
                "provider": pid,
                "answer": None,
                "duration_s": round(elapsed, 1),
                "routing_reason": "explicit param",
                "error": exc.code,
            }
        return entry, elapsed

    trace("ask_multi_fanout", providers=",".join(ids), strategy=chosen)
    results = await asyncio.gather(*(_one(pid) for pid in ids))
    answers = [entry for entry, _ in results]
    ok = [(entry, took) for entry, took in results if entry["error"] is None]
    selected = None
    if chosen == "first" and ok:
        selected = min(ok, key=lambda item: item[1])[0]
    elif chosen == "longest" and ok:
        selected = max(ok, key=lambda item: len(item[0]["answer"]))[0]
    return {"answers": answers, "selected": selected}


async def handle_list_providers(state: AppState) -> dict:
    return {
        "providers": [
            {"id": a.id, "name": a.name, "status": a.status}
            for a in state.registry.list_all()
        ]
    }


async def handle_session_status(
    state: AppState,
    *,
    provider: str | None,
) -> dict:
    # Session checks navigate — use a dedicated status tab so they never
    # touch a pinned provider tab that may be mid-generation.
    page = await state.page_router.status_page()
    targets = (
        [state.registry.get(provider)]
        if provider
        else state.registry.list_all()
    )
    result: dict[str, str] = {}
    for adapter in targets:
        if adapter.status == "coming_soon":
            result[adapter.id] = SessionStatus.UNKNOWN.value
            continue
        status = await adapter.check_session(page)
        result[adapter.id] = status.value
    return result
