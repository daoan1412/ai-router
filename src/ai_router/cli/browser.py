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


async def _provision_deepseek(
    prompt: str,
    reuse: bool,
    headless: bool,
) -> None:
    from ai_router import mailtemp_web
    from ai_router.adapters.deepseek.auth import deepseek_sign_up, smoke_chat
    from ai_router.dotenv import set_env_vars
    from ai_router.mailtemp import MailtempClient

    def log(msg: str) -> None:
        typer.echo(msg)

    cfg = load_config()  # also loads .env into os.environ
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)

    # Code source: the keyless website by default; the API only if a key is set.
    api = MailtempClient()
    use_api = api.has_key
    log(
        "Mailbox source: mailtemp.vn API (MAILTEMP_API_KEY set)"
        if use_api
        else "Mailbox source: mailtemp.vn website (keyless)"
    )

    reused_email = os.getenv("DEEPSEEK_EMAIL") if reuse else None
    password = (os.getenv("DEEPSEEK_PASSWORD") if reuse else None) or _generate_password()

    typer.echo("Launching browser (headed — finish any Cloudflare check yourself)...")
    ctx = await launch_persistent_context_async(str(cfg.profile_dir), headless=headless)
    try:
        # Resolve the email + a coroutine that yields the verification code.
        if use_api:
            if reused_email:
                email = reused_email
                typer.echo(f"Reusing DEEPSEEK_EMAIL: {email}")
            else:
                mailbox = await api.create_random_mailbox()
                email = mailbox.address
                typer.echo(f"Created mailbox (API): {email}")

            async def get_code() -> str:
                return await api.wait_for_code(email)
        else:
            mt_page = await ctx.new_page()
            if reused_email:
                email = reused_email
                typer.echo(f"Reusing DEEPSEEK_EMAIL: {email}")
                await mt_page.goto(mailtemp_web.MT_URL, wait_until="networkidle")
            else:
                email = await mailtemp_web.create_mailbox(mt_page, fresh=False, log=log)

            async def get_code() -> str:
                return await mailtemp_web.wait_for_code(
                    mt_page, subject_hint="verification code", log=log
                )

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
