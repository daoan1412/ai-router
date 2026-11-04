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


async def _mailbox_source(ctx, reuse_email, subject_hint, log, link_env=None):
    """Return (email, get_code): keyless mailtemp.vn website, or the API if keyed.

    get_code is a coroutine that resolves to the verification code once it lands
    in the mailbox. subject_hint is the message subject fragment used to open the
    right email on the website (ignored on the API path). link_env names the .env
    key holding the website box's reopen link (address + access token): saved
    for a new box, used to reopen a reused one from any browser profile.
    """
    from ai_router.dotenv import set_env_vars

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
    after_count = None
    if reuse_email:
        email = reuse_email
        log(f"Reusing email: {email}")
        link = os.getenv(link_env) if link_env else None
        await mailtemp_web.open_mailbox(mt_page, email, link=link, log=log)
        # The reused inbox still holds earlier codes: only read a newer message.
        after_count = await mailtemp_web.count_messages(mt_page, subject_hint)
    else:
        # fresh=True forces a new empty address so a persisted profile can't
        # hand us an old message / stale code from a previous run.
        email = await mailtemp_web.create_mailbox(mt_page, fresh=True, log=log)
        link = await mailtemp_web.mailbox_link(mt_page, email)
        if link_env and link:
            set_env_vars({link_env: link})

    async def get_code() -> str:
        return await mailtemp_web.wait_for_code(
            mt_page, subject_hint=subject_hint, after_count=after_count, log=log
        )

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
        email, get_code = await _mailbox_source(
            ctx, reused_email, "verification code", log, "DEEPSEEK_MAILBOX_LINK"
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
            email, get_code = await _mailbox_source(
                ctx, reused_email, "Qwen", log, "QWEN_MAILBOX_LINK"
            )
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


async def _provision_zai(prompt: str, reuse: bool, headless: bool) -> None:
    from ai_router import mailtemp_web
    from ai_router.adapters.zai.auth import (
        MODEL_53,
        VERIFY_LINK_PART,
        ZAI_URL,
        chat,
        dismiss_overlays,
        signed_in_now,
        zai_sign_in,
        zai_sign_up,
    )
    from ai_router.dotenv import set_env_vars

    def log(msg: str) -> None:
        typer.echo(msg)

    cfg = load_config()
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    saved_email = os.getenv("ZAI_EMAIL") if reuse else None
    saved_password = os.getenv("ZAI_PASSWORD") if reuse else None

    typer.echo("Launching browser (headed — you drag Z.ai's Aliyun puzzle when it appears)...")
    ctx = await launch_persistent_context_async(str(cfg.profile_dir), headless=headless)
    try:
        z_page = await ctx.new_page()
        await z_page.goto(ZAI_URL, wait_until="domcontentloaded")
        await z_page.wait_for_timeout(3000)
        await dismiss_overlays(z_page, log)

        if await signed_in_now(z_page):
            typer.echo("Already signed in (saved session) — no email/puzzle needed.")
            outcome = "already_signed_in"
        elif saved_email and saved_password:
            typer.echo(f"Signing in as {saved_email}...")
            outcome = await zai_sign_in(z_page, email=saved_email, password=saved_password, log=log)
        else:
            # Z.ai verifies by a link in the email; the website mailbox exposes it.
            mt_page = await ctx.new_page()
            email = await mailtemp_web.create_mailbox(mt_page, fresh=True, log=log)
            password = _generate_password()
            env_path = set_env_vars({"ZAI_EMAIL": email, "ZAI_PASSWORD": password})
            typer.echo(f"Saved ZAI_EMAIL / ZAI_PASSWORD to {env_path}")

            async def get_link() -> str:
                return await mailtemp_web.wait_for_link(
                    mt_page, href_contains=VERIFY_LINK_PART, subject_hint="Verify Your Email", log=log
                )

            outcome = await zai_sign_up(
                z_page,
                email=email,
                password=password,
                name=email.split("@")[0],
                get_link=get_link,
                log=log,
            )

        if outcome == "needs_manual":
            typer.echo(
                "Could not confirm sign-in (puzzle unsolved?). The chat test below "
                "falls back to guest mode (GLM-5.3-Flash)."
            )
        await z_page.goto(ZAI_URL, wait_until="domcontentloaded")
        await z_page.wait_for_timeout(2000)
        await dismiss_overlays(z_page, log)
        signed_in = await signed_in_now(z_page)
        typer.echo("")
        typer.echo(f"--- Smoke test chat ({MODEL_53 if signed_in else 'guest'}) ---")
        answer, reasoning = await chat(
            z_page, prompt, log=log, model_label=MODEL_53 if signed_in else None
        )
        typer.echo("")
        if reasoning:
            typer.echo(f"Z.ai reasoning ({len(reasoning)} chars): {reasoning[:200]}...")
        typer.echo("Z.ai replied:")
        typer.echo(answer or "(no answer captured)")
    finally:
        try:
            await ctx.close()
        except PlaywrightError:
            pass


@browser_app.command("provision-zai")
def provision_zai(
    prompt: Annotated[
        str, typer.Option(help="Smoke-test prompt to send after login")
    ] = "Say hello in one short sentence.",
    reuse: Annotated[
        bool,
        typer.Option(help="Sign in with ZAI_EMAIL/ZAI_PASSWORD from .env instead of a new account"),
    ] = False,
    headless: Annotated[bool, typer.Option(help="Run headless (not recommended)")] = False,
) -> None:
    """Create a Z.ai account with a temp email (or sign in), then send a test chat.

    Headed: drag the Aliyun puzzle when it appears; the rest is automated
    (form -> verification link from the mailbox -> complete registration).
    A signed-in account unlocks GLM-5.3 / GLM-5.2 (guests only get GLM-5.3-Flash).
    """
    asyncio.run(_provision_zai(prompt, reuse, headless))


async def _provision_mimo(prompt: str, reuse: bool, headless: bool) -> None:
    from ai_router.adapters.mimo.auth import (
        CODE_SUBJECT_HINT,
        MIMO_CHAT_URL,
        MODEL_PRO,
        chat,
        is_signed_in,
        mimo_sign_in,
        mimo_sign_up,
    )
    from ai_router.dotenv import set_env_vars

    def log(msg: str) -> None:
        typer.echo(msg)

    cfg = load_config()
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    saved_email = os.getenv("MIMO_EMAIL") if reuse else None
    saved_password = os.getenv("MIMO_PASSWORD") if reuse else None

    typer.echo("Launching browser (headed — you tick Xiaomi's reCAPTCHA when it appears)...")
    ctx = await launch_persistent_context_async(str(cfg.profile_dir), headless=headless)
    try:
        m_page = await ctx.new_page()
        await m_page.goto(MIMO_CHAT_URL, wait_until="domcontentloaded")

        if await is_signed_in(m_page):
            typer.echo("Already signed in (saved session) — no email/captcha needed.")
            outcome = "already_signed_in"
        elif saved_email and saved_password:
            typer.echo(f"Signing in as {saved_email}...")
            # A new device is asked for an e-mail code: reopen the saved mailbox.
            _, get_code = await _mailbox_source(
                ctx, saved_email, CODE_SUBJECT_HINT, log, "MIMO_MAILBOX_LINK"
            )
            outcome = await mimo_sign_in(
                m_page, email=saved_email, password=saved_password, get_code=get_code, log=log
            )
        else:
            email, get_code = await _mailbox_source(
                ctx, None, CODE_SUBJECT_HINT, log, "MIMO_MAILBOX_LINK"
            )
            password = _generate_password()
            env_path = set_env_vars({"MIMO_EMAIL": email, "MIMO_PASSWORD": password})
            typer.echo(f"Saved MIMO_EMAIL / MIMO_PASSWORD to {env_path}")
            outcome = await mimo_sign_up(
                m_page, email=email, password=password, get_code=get_code, log=log
            )

        if outcome == "needs_manual":
            typer.echo("Could not confirm sign-in (reCAPTCHA unsolved or code expired).")
            return
        await m_page.goto(MIMO_CHAT_URL, wait_until="domcontentloaded")
        if not await is_signed_in(m_page):
            typer.echo("Not signed in — skipping the chat test.")
            return
        typer.echo("")
        typer.echo(f"--- Smoke test chat ({MODEL_PRO}) ---")
        answer, reasoning = await chat(m_page, prompt, log=log, model_label=MODEL_PRO)
        typer.echo("")
        if reasoning:
            typer.echo(f"MiMo reasoning ({len(reasoning)} chars): {reasoning[:200]}...")
        typer.echo("MiMo replied:")
        typer.echo(answer or "(no answer captured)")
    finally:
        try:
            await ctx.close()
        except PlaywrightError:
            pass


@browser_app.command("provision-mimo")
def provision_mimo(
    prompt: Annotated[
        str, typer.Option(help="Smoke-test prompt to send after login")
    ] = "Say hello in one short sentence.",
    reuse: Annotated[
        bool,
        typer.Option(help="Sign in with MIMO_EMAIL/MIMO_PASSWORD from .env instead of a new account"),
    ] = False,
    headless: Annotated[bool, typer.Option(help="Run headless (not recommended)")] = False,
) -> None:
    """Create a Xiaomi Account with a temp email (or sign in) for MiMo Studio, then chat.

    Headed: tick Google's reCAPTCHA when it appears; the rest is automated
    (form -> e-mail code from the mailbox -> MiMo user agreement).
    """
    asyncio.run(_provision_mimo(prompt, reuse, headless))


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
