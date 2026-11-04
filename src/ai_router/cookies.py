"""Turn a pasted browser cookie export into Playwright ``add_cookies`` entries.

Lets a user sign in on their own browser and hand the session over: they paste
whatever their browser gives them and it is loaded into the ai-router profile.
Accepted formats (auto-detected):

- a ``Cookie:`` request header value (DevTools → Network → request headers),
- a "Copy as cURL" command (its ``-H 'cookie: …'`` / ``-b '…'`` part),
- a JSON export (Cookie-Editor / EditThisCookie: a list of cookie objects),
- a Netscape ``cookies.txt`` file.

A bare header has no attributes, so those cookies get the provider's parent
domain, path ``/``, Secure, and a one-year expiry; ``__Host-`` cookies are
host-only on the provider's host.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any

_YEAR_S = 365 * 24 * 3600
_SAMESITE = {
    "no_restriction": "None",
    "none": "None",
    "lax": "Lax",
    "strict": "Strict",
}


def _cookie(name: str, value: str, *, domain: str, host: str, now: float) -> dict[str, Any]:
    c: dict[str, Any] = {
        "name": name,
        "value": value,
        "secure": True,
        "httpOnly": False,
        "expires": now + _YEAR_S,
        # Third-party-partitioned Google cookies (…3PSID…) are SameSite=None.
        "sameSite": "None" if "3P" in name else "Lax",
    }
    if name.startswith("__Host-"):
        c["url"] = f"https://{host}/"
    else:
        c["domain"], c["path"] = domain, "/"
    return c


def _from_header(header: str, *, domain: str, host: str, now: float) -> list[dict[str, Any]]:
    header = re.sub(r"^\s*cookie\s*:\s*", "", header.strip(), flags=re.I)
    out = []
    for part in header.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name.strip():
            out.append(_cookie(name.strip(), value.strip(), domain=domain, host=host, now=now))
    return out


def _from_curl(text: str) -> str | None:
    """The cookie string inside a 'Copy as cURL' command (bash or cmd quoting)."""
    flat = text.replace("^\n", " ").replace("\\\n", " ").replace("^", "")
    try:
        args = shlex.split(flat, posix=True)
    except ValueError:
        args = flat.split()
    for i, arg in enumerate(args[:-1]):
        nxt = args[i + 1]
        if arg in ("-H", "--header") and nxt.lower().startswith("cookie:"):
            return nxt
        if arg in ("-b", "--cookie"):
            return nxt
    return None


def _from_json(data: Any, *, host: str, now: float) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("cookies", [data])
    out = []
    for c in data if isinstance(data, list) else []:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        entry: dict[str, Any] = {
            "name": str(c["name"]),
            "value": str(c.get("value", "")),
            "secure": bool(c.get("secure", True)),
            "httpOnly": bool(c.get("httpOnly", False)),
        }
        same = _SAMESITE.get(str(c.get("sameSite", "")).lower())
        if same:
            entry["sameSite"] = same
        if same == "None":
            entry["secure"] = True
        exp = c.get("expirationDate", c.get("expires"))
        entry["expires"] = float(exp) if isinstance(exp, (int, float)) and exp > 0 else now + _YEAR_S
        domain = str(c.get("domain") or "")
        if c.get("hostOnly") or entry["name"].startswith("__Host-") or not domain:
            entry["url"] = f"https://{domain.lstrip('.') or host}/"
        else:
            entry["domain"], entry["path"] = domain, str(c.get("path") or "/")
        out.append(entry)
    return out


def _from_netscape(text: str, *, now: float) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        http_only = line.startswith("#HttpOnly_")
        if http_only:
            line = line[len("#HttpOnly_"):]
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        domain, _flag, path, secure, expires, name, value = parts
        exp = float(expires) if expires.strip().isdigit() and int(expires) > 0 else now + _YEAR_S
        out.append({
            "name": name, "value": value.strip(), "domain": domain, "path": path or "/",
            "secure": secure.upper() == "TRUE", "httpOnly": http_only, "expires": exp,
        })
    return out


def parse_cookie_paste(
    text: str, *, domain: str = ".google.com", host: str = "gemini.google.com",
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Parse a pasted cookie export into Playwright cookie dicts (last wins)."""
    now = time.time() if now is None else now
    t = text.strip().lstrip("﻿")
    cookies: list[dict[str, Any]]
    if t[:1] in "[{":
        cookies = _from_json(json.loads(t), host=host, now=now)
    elif "\t" in t and any(len(ln.split("\t")) == 7 for ln in t.splitlines()):
        cookies = _from_netscape(t, now=now)
    elif re.match(r"^\s*curl(\.exe)?\s", t, flags=re.I):
        header = _from_curl(t)
        if header is None:
            raise ValueError("no cookie header (-H 'cookie: …' or -b) found in the cURL command")
        cookies = _from_header(header, domain=domain, host=host, now=now)
    else:
        cookies = _from_header(" ".join(t.splitlines()), domain=domain, host=host, now=now)
    merged: dict[tuple, dict[str, Any]] = {}
    for c in cookies:
        merged[(c["name"], c.get("domain") or c.get("url"), c.get("path", "/"))] = c
    return list(merged.values())


GOOGLE_SESSION_COOKIES = ("__Secure-1PSID", "__Secure-3PSID", "SID")


def has_google_session(cookies: list[dict[str, Any]]) -> bool:
    names = {c["name"] for c in cookies}
    return any(n in names for n in GOOGLE_SESSION_COOKIES)
