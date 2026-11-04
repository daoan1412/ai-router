from __future__ import annotations

import asyncio
import os
import secrets
import string
from typing import Annotated

import typer
from playwright.async_api import Error as PlaywrightError

from ai_router.adapters.registry import build_registry
from ai_router.browser.cloak import launch_persistent_context_async
from ai_router.config import load_config

browser_app = typer.Typer(help="Browser login and session management")


def _generate_password(length: int = 16) -> str:
    """A strong password with upper/lower/digit/symbol (DeepSeek-friendly)."""
    alphabet = string.ascii_letters + string.digits
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length - 1)) + "!"
        if any(c.islower() for c in pw) and any(c.isupper() for c in pw) and any(
            c.isdigit() for c in pw
        ):
            return pw


async def _login(provider: str | None) -> None:
    cfg = load_config()
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    targets = (
        [registry.get(provider)]
        if provider
        else [a for a in registry.list_all() if a.status == "available"]
    )

    typer.echo("Launching browser...")
    ctx = await launch_persistent_context_async(
        str(cfg.profile_dir),
        headless=False,
    )
    try:
        for adapter in targets:
            page = await ctx.new_page()
            url = cfg.providers[adapter.id].url
            typer.echo(f"Opening {adapter.name}: {url}")
            await page.goto(url, wait_until="domcontentloaded")
        typer.echo("Log in to each provider, then close all browser windows...")
        while ctx.pages:
            await asyncio.sleep(0.5)
    finally:
        try:
            await ctx.close()
        except PlaywrightError:
            pass


@browser_app.command("login")
def login(
    provider: Annotated[
        str | None, typer.Option(help="Provider id (default: all available)")
    ] = None,
) -> None:
    """Open headed browser for manual login. Close window when done."""
    asyncio.run(_login(provider))


async def _mailbox_source(ctx, reuse_email, subject_hint, log):
    """Return (email, get_code): keyless mailtemp.vn website, or the API if keyed.

    get_code is a coroutine that resolves to the verification code once it lands
    in the mailbox. subject_hint is the message subject fragment used to open the
    right email on the website (ignored on the API path).
    """
    from ai_router import mailtemp_web
    from ai_router.mailtemp import MailtempClient

    api = MailtempClient()
    if api.has_key:
        log("Mailbox source: mailtemp.vn API (MAILTEMP_API_KEY set)")
        if reuse_email:
            email = reuse_email
            log(f"Reusing email: {email}")
        else:
            email = (await api.create_random_mailbox()).address
            log(f"Created mailbox (API): {email}")

        async def get_code() -> str:
            return await api.wait_for_code(email)

        return email, get_code

    log("Mailbox source: mailtemp.vn website (keyless)")
    mt_page = await ctx.new_page()
    if reuse_email:
        email = reuse_email
        log(f"Reusing email: {email}")
        await mt_page.goto(mailtemp_web.MT_URL, wait_until="networkidle")
    else:
        # fresh=True forces a new empty address so a persisted profile can't
        # hand us an old message / stale code from a previous run.
        email = await mailtemp_web.create_mailbox(mt_page, fresh=True, log=log)

    async def get_code() -> str:
        return await mailtemp_web.wait_for_code(mt_page, subject_hint=subject_hint, log=log)

    return email, get_code


async def _provision_deepseek(prompt: str, reuse: bool, headless: bool) -> None:
    from ai_router.adapters.deepseek.auth import deepseek_sign_up, smoke_chat
    from ai_router.dotenv import set_env_vars

    def log(msg: str) -> None:
        typer.echo(msg)

    cfg = load_config()  # also loads .env into os.environ
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    reused_email = os.getenv("DEEPSEEK_EMAIL") if reuse else None
    password = (os.getenv("DEEPSEEK_PASSWORD") if reuse else None) or _generate_password()

    typer.echo("Launching browser (headed — finish any Cloudflare check yourself)...")
    ctx = await launch_persistent_context_async(str(cfg.profile_dir), headless=headless)
    try:
        email, get_code = await _mailbox_source(ctx, reused_email, "verification code", log)
        env_path = set_env_vars({"DEEPSEEK_EMAIL": email, "DEEPSEEK_PASSWORD": password})
        typer.echo(f"Saved DEEPSEEK_EMAIL / DEEPSEEK_PASSWORD to {env_path}")

        ds_page = await ctx.new_page()
        outcome = await deepseek_sign_up(
            ds_page, email=email, password=password, get_code=get_code, log=log
        )
        if outcome == "needs_manual":
            typer.echo(
                "Could not confirm login automatically. If you finished it in the "
                "window, the chat test below will still work."
            )
        typer.echo("")
        typer.echo("--- Smoke test chat ---")
        answer = await smoke_chat(ds_page, prompt, log=log)
        typer.echo("")
        typer.echo("DeepSeek replied:")
        typer.echo(answer or "(no answer captured)")
    finally:
        try:
            await ctx.close()
        except PlaywrightError:
            pass


@browser_app.command("provision-deepseek")
def provision_deepseek(
    prompt: Annotated[
        str, typer.Option(help="Smoke-test prompt to send after login")
    ] = "Say hello in one short sentence.",
    reuse: Annotated[
        bool,
        typer.Option(help="Reuse DEEPSEEK_EMAIL/PASSWORD from .env instead of a new mailbox"),
    ] = False,
    headless: Annotated[bool, typer.Option(help="Run headless (not recommended)")] = False,
) -> None:
    """Create a temp email (keyless mailtemp.vn), sign up DeepSeek, send a test chat."""
    asyncio.run(_provision_deepseek(prompt, reuse, headless))


async def _provision_qwen(prompt: str, reuse: bool, headless: bool) -> None:
    from ai_router.adapters.qwen.auth import (
        QWEN_URL,
        disable_tools,
        is_logged_in,
        qwen_sign_up,
        smoke_chat,
    )
    from ai_router.dotenv import set_env_vars

    def log(msg: str) -> None:
        typer.echo(msg)

    cfg = load_config()
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    reused_email = os.getenv("QWEN_EMAIL") if reuse else None

    typer.echo("Launching browser (headed — you solve Qwen's drag puzzle when it appears)...")
    ctx = await launch_persistent_context_async(str(cfg.profile_dir), headless=headless)
    try:
        # Reuse a saved session first: if the profile cookie is still valid we
        # skip the mailbox, email, code and puzzle entirely.
        q_page = await ctx.new_page()
        await q_page.goto(QWEN_URL, wait_until="domcontentloaded")
        await q_page.wait_for_timeout(1500)

        if await is_logged_in(q_page):
            typer.echo("Already logged in (saved session) — no email/code/puzzle needed.")
        else:
            # Qwen's sign-in email subject is "Your Qwen sign-in code".
            email, get_code = await _mailbox_source(ctx, reused_email, "Qwen", log)
            env_path = set_env_vars({"QWEN_EMAIL": email})
            typer.echo(f"Saved QWEN_EMAIL to {env_path} (Qwen is passwordless — code per login)")
            outcome = await qwen_sign_up(q_page, email=email, get_code=get_code, log=log)
            if outcome == "needs_manual":
                typer.echo(
                    "Could not confirm login automatically (puzzle unsolved or code "
                    "expired). If you finished it in the window, the chat test may still work."
                )

        if await is_logged_in(q_page):
            typer.echo("Turning off Qwen web search / tools / memory (account settings)...")
            off = await disable_tools(q_page, log=log)
            typer.echo(f"  switched off: {', '.join(off) if off else '(already off)'}")

        typer.echo("")
        typer.echo("--- Smoke test chat ---")
        answer = await smoke_chat(q_page, prompt, log=log)
        typer.echo("")
        typer.echo("Qwen replied:")
        typer.echo(answer or "(no answer captured)")
    finally:
        try:
            await ctx.close()
        except PlaywrightError:
            pass


@browser_app.command("provision-qwen")
def provision_qwen(
    prompt: Annotated[
        str, typer.Option(help="Smoke-test prompt to send after login")
    ] = "Say hello in one short sentence.",
    reuse: Annotated[
        bool, typer.Option(help="Reuse QWEN_EMAIL from .env instead of a new mailbox")
    ] = False,
    headless: Annotated[bool, typer.Option(help="Run headless (not recommended)")] = False,
) -> None:
    """Create a temp email (keyless mailtemp.vn), sign in to Qwen, send a test chat.

    Headed: when Qwen shows its drag-puzzle, solve it in the window; the rest is
    automated (email -> code -> age gate -> chat).
    """
    asyncio.run(_provision_qwen(prompt, reuse, headless))


@browser_app.command("status")
def status(
    provider: Annotated[str | None, typer.Option(help="Provider id (default: all)")] = None,
) -> None:
    """Check login status for provider(s)."""
    from ai_router.mcp.tools import create_app_state, handle_session_status

    async def _run() -> dict:
        state = create_app_state()
        return await handle_session_status(state, provider=provider)

    result = asyncio.run(_run())
    for pid, st in result.items():
        typer.echo(f"{pid}: {st}")
