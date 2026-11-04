from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path.home() / ".ai-router"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


@dataclass
class ProviderConfig:
    url: str


@dataclass
class AppConfig:
    profile_dir: Path
    default_provider: str
    host: str
    port: int
    answer_timeout_s: int
    idle_streak_required: int = 6
    generating_streak_required: int = 2
    answer_stable_ticks: int = 4
    # DOM-only completion fallback: with a submitted job and NO network
    # stream-end signal, accept the answer after this many consecutive
    # stable DOM ticks with the stop button gone (20 ticks ~= 10s).
    no_stream_fallback_ticks: int = 20
    stream_quiet_s: float = 5.0
    dom_tick_interval_ms: int = 500
    chatgpt_answer_timeout_s: float = 300.0
    claude_answer_timeout_s: float = 300.0
    deepseek_answer_timeout_s: float = 600.0
    kimi_answer_timeout_s: float = 600.0
    max_pages: int = 10
    # ask_multi fan-out defaults; empty list = all "available" providers
    parallel_default_providers: list[str] = field(default_factory=list)
    parallel_default_strategy: str = "all"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _defaults() -> AppConfig:
    return AppConfig(
        profile_dir=CONFIG_DIR / "profile",
        default_provider="gemini",
        host="127.0.0.1",
        port=8087,
        answer_timeout_s=120,
        providers={
            "gemini": ProviderConfig(url="https://gemini.google.com/app"),
            "chatgpt": ProviderConfig(url="https://chatgpt.com/"),
            "claude": ProviderConfig(url="https://claude.ai/new"),
            "deepseek": ProviderConfig(url="https://chat.deepseek.com/"),
            "kimi": ProviderConfig(
                url="https://www.kimi.com/?chat_enter_method=new_chat"
            ),
        },
    )


def load_config(path: Path | None = None) -> AppConfig:
    from ai_router.dotenv import load_dotenv

    load_dotenv()
    cfg = _defaults()
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if "profile_dir" in raw:
            cfg.profile_dir = _expand(raw["profile_dir"])
        if "default_provider" in raw:
            cfg.default_provider = raw["default_provider"]
        if "host" in raw:
            cfg.host = raw["host"]
        if "port" in raw:
            cfg.port = int(raw["port"])
        if "answer_timeout_s" in raw:
            cfg.answer_timeout_s = int(raw["answer_timeout_s"])
        if "stream_quiet_s" in raw:
            cfg.stream_quiet_s = float(raw["stream_quiet_s"])
        if "chatgpt_answer_timeout_s" in raw:
            cfg.chatgpt_answer_timeout_s = float(raw["chatgpt_answer_timeout_s"])
        if "claude_answer_timeout_s" in raw:
            cfg.claude_answer_timeout_s = float(raw["claude_answer_timeout_s"])
        if "deepseek_answer_timeout_s" in raw:
            cfg.deepseek_answer_timeout_s = float(raw["deepseek_answer_timeout_s"])
        if "kimi_answer_timeout_s" in raw:
            cfg.kimi_answer_timeout_s = float(raw["kimi_answer_timeout_s"])
        if "providers" in raw:
            for pid, pdata in raw["providers"].items():
                cfg.providers[pid] = ProviderConfig(url=pdata["url"])
        if "parallel_ask" in raw:
            pa = raw["parallel_ask"] or {}
            if "default_providers" in pa:
                cfg.parallel_default_providers = [str(p) for p in pa["default_providers"]]
            if "default_strategy" in pa:
                cfg.parallel_default_strategy = str(pa["default_strategy"])

    if v := os.getenv("AI_ROUTER_PROFILE_DIR"):
        cfg.profile_dir = _expand(v)
    if v := os.getenv("AI_ROUTER_DEFAULT_PROVIDER"):
        cfg.default_provider = v
    if v := os.getenv("AI_ROUTER_HOST"):
        cfg.host = v
    if v := os.getenv("AI_ROUTER_PORT"):
        cfg.port = int(v)
    if v := os.getenv("AI_ROUTER_ANSWER_TIMEOUT_S"):
        cfg.answer_timeout_s = int(v)
    if v := os.getenv("AI_ROUTER_IDLE_STREAK_REQUIRED"):
        cfg.idle_streak_required = int(v)
    if v := os.getenv("AI_ROUTER_STREAM_QUIET_S"):
        cfg.stream_quiet_s = float(v)
    if v := os.getenv("AI_ROUTER_CHATGPT_ANSWER_TIMEOUT_S"):
        cfg.chatgpt_answer_timeout_s = float(v)
    if v := os.getenv("AI_ROUTER_CLAUDE_ANSWER_TIMEOUT_S"):
        cfg.claude_answer_timeout_s = float(v)
    if v := os.getenv("AI_ROUTER_DEEPSEEK_ANSWER_TIMEOUT_S"):
        cfg.deepseek_answer_timeout_s = float(v)
    if v := os.getenv("AI_ROUTER_KIMI_ANSWER_TIMEOUT_S"):
        cfg.kimi_answer_timeout_s = float(v)

    return cfg
