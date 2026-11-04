# AI Router

AI Router is a Python stdio MCP server. Install the **`mcp-ai-router`** package from PyPI, run the **`ai-router`** CLI, and connect any MCP client with `"command": "ai-router"` and `"args": ["serve"]`.

It automates authenticated **Gemini**, **ChatGPT**, **Claude**, **DeepSeek**, and **Kimi** web sessions through CloakBrowser — **no API keys**, but you do need provider web accounts and a one-time CLI login.

| | Name |
|---|------|
| **PyPI / uv** | [`mcp-ai-router`](https://pypi.org/project/mcp-ai-router/) |
| **CLI command** | `ai-router` |
| **GitHub** | [scriptkid23/ai-router](https://github.com/scriptkid23/ai-router) |

## Quick start

```bash
uv tool install mcp-ai-router       # 1. install
ai-router browser login             # 2. log in (opens browser windows)
ai-router browser status            # 3. expect gemini/chatgpt/claude/deepseek/kimi: logged_in
```

Add MCP config (see [Connect MCP client](#connect-mcp-client)), reload your client, then verify:

> Ask your agent: *"Call list_providers, then ask Gemini to reply with exactly: router working"*

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.11+ (uv installs it for you if missing) |
| [uv](https://docs.astral.sh/uv/) | 0.13+ |
| Disk + network | ~200 MB free for first browser download |

Tested on **macOS**, **Linux**, and **Windows**. No Node.js or repo clone required for normal stdio MCP use.

**Provider accounts:** valid Gemini, ChatGPT, Claude, DeepSeek, and/or Kimi web accounts (free tiers work). Login is manual via the browser UI.

## Install

```bash
# install uv itself (pick one)
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
winget install astral-sh.uv                        # Windows
# open a new terminal, then:
uv tool install mcp-ai-router
```

uv provisions its own Python 3.11+ for the tool, so no system Python is required. See [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) if the commands above fail. `uv tool install` puts `ai-router` in uv's tool bin directory — run `uv tool update-shell` if it is not on your PATH.

Verify:

```bash
ai-router --version
ai-router --help
```

Upgrade later:

```bash
uv tool upgrade mcp-ai-router
```

Uninstall:

```bash
uv tool uninstall mcp-ai-router
# optional manual cleanup:
# rm -rf ~/.ai-router ~/.cloakbrowser
```

## Login (one-time)

```bash
ai-router browser login                  # all available providers
ai-router browser login --provider gemini
ai-router browser login --provider chatgpt
ai-router browser login --provider claude
ai-router browser login --provider deepseek
ai-router browser login --provider kimi
```

1. Visible browser windows open (one per provider being configured).
2. Log in with your provider account in each window.
3. Close **all** browser windows when done — this saves the session to disk.

Sessions are stored in `~/.ai-router/profile/`. Logging in to one provider only is fine; use `ai-router browser status` to check.

```bash
ai-router browser status
# gemini: logged_in
# chatgpt: logged_in
# claude: logged_in
# deepseek: logged_in
```

> **CloakBrowser download (first use)**  
> `uv tool install` does **not** download the browser. The first `browser login` or `ask` triggers a one-time download of a CloakBrowser-managed Chromium binary (~200 MB) to `~/.cloakbrowser/`. You do **not** need `playwright install` or a separate Chrome install — automation uses the CloakBrowser binary, not your local Chrome app.

### Auto-provision a DeepSeek account (temp email)

Instead of logging in manually, `provision-deepseek` creates a disposable
mailbox on [mailtemp.vn](https://mailtemp.vn), signs up DeepSeek by email, reads
the verification code from that mailbox, saves the credentials to a project
`.env`, and sends one smoke-test chat.

```bash
ai-router browser provision-deepseek                   # new mailbox + account + test chat
ai-router browser provision-deepseek --reuse           # reuse DEEPSEEK_EMAIL/PASSWORD from .env
ai-router browser provision-deepseek --prompt "What is 2+2?"
```

By default this uses the mailtemp.vn **website** — no API key needed. (The
mailtemp.vn *API* requires `x-api-key`; set `MAILTEMP_API_KEY` in `.env` to use
that path instead.) The browser runs **headed**: automation fills email, code,
password and confirm-password, but if Cloudflare's Turnstile appears, solve it
in the window — the command waits up to 5 min for login. Credentials land in
`.env` as `DEEPSEEK_EMAIL` / `DEEPSEEK_PASSWORD` (gitignored).

## Connect MCP client

Works with any MCP client that supports stdio — **Cursor**, **Claude Code**, **Claude Desktop**, and others.

**Cursor** — `~/.cursor/mcp.json` or project `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ai-router": {
      "type": "stdio",
      "command": "ai-router",
      "args": ["serve"]
    }
  }
}
```

**Claude Code** — `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ai-router": {
      "command": "ai-router",
      "args": ["serve"]
    }
  }
}
```

**Claude Desktop** — MCP config file for your OS ([Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/mcp)); same `command` / `args`.

The client spawns `ai-router serve` (stdio, default). No separate terminal, no Node.js.

#### If `"command": "ai-router"` fails

GUI apps often use a different PATH than your terminal. Use the absolute path from:

```bash
command -v ai-router    # macOS/Linux
where ai-router       # Windows
```

```json
{
  "mcpServers": {
    "ai-router": {
      "type": "stdio",
      "command": "/absolute/path/to/ai-router",
      "args": ["serve"]
    }
  }
}
```

Use a full absolute path in JSON — do not use `~` (most clients do not expand it).

Reload MCP in your client after saving.

## MCP tools

| Tool | Description |
|------|-------------|
| `ask(prompt, provider?)` | Send a prompt; default provider is Gemini. Set `provider` to `"chatgpt"`, `"claude"`, `"deepseek"`, or `"kimi"`. |
| `ask_multi(prompt, providers?, strategy?)` | Fan out to multiple providers in parallel (`strategy`: `all`, `first`, `longest`). |
| `list_providers()` | List providers and status. |
| `session_status(provider?)` | Check login state without opening a chat. |

Login is **CLI only** — run `ai-router browser login` manually; there is no MCP login tool.

**Conversation behavior:** each `ask` opens a **new** provider web chat. Follow-up context is not preserved across calls. Your MCP client's thread and the provider's web chat are separate.

**Concurrency:** one MCP server process handles requests through a page queue. Concurrent `ask` calls to the same provider may return `BROWSER_BUSY` — wait and retry. `ask_multi` fans out across providers in parallel within one server.

## CLI reference

```bash
ai-router --version
ai-router serve [--transport stdio|http] [--host 127.0.0.1] [--port 8087]
ai-router browser login [--provider gemini|chatgpt|claude|deepseek|kimi]
ai-router browser status [--provider gemini|chatgpt|claude|deepseek|kimi]
```

| Transport | Use case |
|-----------|----------|
| `stdio` (default) | MCP clients — `"args": ["serve"]` |
| `http` | Local debugging only — see below |

## Config (optional)

Create `~/.ai-router/config.yaml`. Values merge with built-in defaults:

```yaml
default_provider: gemini
host: 127.0.0.1
port: 8087
answer_timeout_s: 120
profile_dir: ~/.ai-router/profile
providers:
  gemini:
    url: https://gemini.google.com/app
  chatgpt:
    url: https://chatgpt.com/
  claude:
    url: https://claude.ai/new
  deepseek:
    url: https://chat.deepseek.com/
  kimi:
    url: https://www.kimi.com/?chat_enter_method=new_chat
```

Precedence: environment variables override YAML; YAML overrides defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_ROUTER_PROFILE_DIR` | `~/.ai-router/profile` | Browser profile directory |
| `AI_ROUTER_DEFAULT_PROVIDER` | `gemini` | Default provider for `ask` |
| `AI_ROUTER_HOST` | `127.0.0.1` | HTTP bind address |
| `AI_ROUTER_PORT` | `8087` | HTTP port |
| `AI_ROUTER_ANSWER_TIMEOUT_S` | `120` | Per-request timeout (seconds) |
| `AI_ROUTER_DEEPSEEK_ANSWER_TIMEOUT_S` | `600` | DeepSeek per-request timeout (seconds; supports long thinking runs) |
| `AI_ROUTER_KIMI_ANSWER_TIMEOUT_S` | `600` | Kimi per-request timeout (seconds; supports long thinking runs) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv: command not found` | Install uv (see [Install](#install)), then open a new terminal |
| `ai-router: command not found` | Run `uv tool update-shell`, open a new terminal |
| MCP client cannot find `ai-router` | Use absolute path in `"command"` |
| `uv tool install mcp-ai-router` fails | Run `uv self update`, then check https://pypi.org/project/mcp-ai-router/ |
| `gemini: logged_out` / `NOT_LOGGED_IN` | Run `ai-router browser login` |
| Slow first login or first `ask` | CloakBrowser downloading ~200 MB, or cold browser start |
| Browser does not open | Needs network + disk; requires `cloakbrowser` ≥ 0.4.4 |
| `BROWSER_BUSY` | Another request is in progress — wait and retry |
| Profile lock / browser errors | Only one MCP server per profile; restart client to clear stale processes |
| Need HTTP debug | `ai-router serve --transport http` |

### HTTP debug (advanced)

Not needed for stdio MCP. Run the server in a terminal, then bridge with Node.js:

```bash
ai-router serve --transport http
```

```json
{
  "mcpServers": {
    "ai-router": {
      "command": "npx",
      "args": ["-y", "mcp-remote@latest", "http://127.0.0.1:8087/mcp"]
    }
  }
}
```

HTTP binds to localhost only and has **no authentication** — do not expose it beyond your machine.

## Development

```bash
git clone https://github.com/scriptkid23/ai-router.git
cd ai-router
uv sync                 # creates .venv from uv.lock, incl. the dev group
uv run pytest -v
uv run ruff check src tests
uv run ai-router serve --transport http
```

Dependencies are declared in `pyproject.toml` (PEP 621) and pinned in `uv.lock`; commit the lockfile with any dependency change. Add or drop packages with `uv add <pkg>` / `uv remove <pkg>` (`--dev` for the dev group) rather than editing the file by hand, and refresh pins with `uv lock --upgrade`.

Smoke-test a wheel (macOS/Linux example):

```bash
uv build
uv tool install --force dist/mcp_ai_router-*.whl
mkdir -p /tmp/ai-router-smoke && cd /tmp/ai-router-smoke
ai-router --help && ai-router --version && ai-router browser status
```

## Security

- `~/.ai-router/profile/` holds live session cookies — treat like a password. Deleting it logs you out locally.
- Logs may be written to `~/.ai-router/logs/` and stderr.
- HTTP debug mode binds to `127.0.0.1` with no auth — localhost only.

## License

MIT — see [LICENSE](LICENSE).
