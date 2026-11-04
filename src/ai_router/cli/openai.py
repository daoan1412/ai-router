from __future__ import annotations

from typing import Annotated

import typer

from ai_router.config import load_config


def serve_openai_cmd(
    host: Annotated[str | None, typer.Option(help="Bind host")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port")] = None,
    headless: Annotated[
        bool, typer.Option(help="Run the browser headless (needs a valid saved session)")
    ] = False,
) -> None:
    """Serve an OpenAI-compatible API (/v1/chat/completions) over the browser providers.

    Returns the assistant answer in message.content and the reasoning in
    message.reasoning_content (DeepSeek R1 thinking; Qwen's "Thinking" panel).
    Web search and tools are kept off for both providers.
    Providers must already be logged in — run `ai-router browser provision-deepseek`
    / `provision-qwen` first.
    """
    from ai_router.openai_api import run_openai_server

    cfg = load_config()
    bind_host = host or cfg.host
    bind_port = port or 8088
    typer.echo(f"OpenAI-compatible API on http://{bind_host}:{bind_port}/v1 (models: deepseek, qwen)")
    run_openai_server(bind_host, bind_port, headless=headless)
