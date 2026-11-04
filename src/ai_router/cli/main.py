import importlib.metadata
from typing import Annotated

import typer

from ai_router.cli.browser import browser_app
from ai_router.cli.openai import serve_openai_cmd
from ai_router.cli.serve import serve_cmd


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("mcp-ai-router"))
        raise typer.Exit()


app = typer.Typer(name="ai-router", help="ai-router — web AI provider automation")


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show installed version and exit",
        ),
    ] = None,
) -> None:
    """MCP server routing prompts to web AI providers via CloakBrowser."""
    pass


app.command("serve")(serve_cmd)
app.command("serve-openai")(serve_openai_cmd)
app.add_typer(browser_app, name="browser")

if __name__ == "__main__":
    app()
