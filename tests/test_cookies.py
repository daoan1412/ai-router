from __future__ import annotations

import json

import pytest

from ai_router.cookies import has_google_session, parse_cookie_paste

NOW = 1_800_000_000.0


def _by_name(cookies):
    return {c["name"]: c for c in cookies}


def test_header_value_with_prefix_and_attributes():
    got = _by_name(parse_cookie_paste(
        "cookie: SID=abc; __Secure-1PSID=s1; __Secure-3PSID=s3; __Host-GAPS=h; NID=n=x",
        now=NOW,
    ))
    assert set(got) == {"SID", "__Secure-1PSID", "__Secure-3PSID", "__Host-GAPS", "NID"}
    assert got["SID"] == {
        "name": "SID", "value": "abc", "secure": True, "httpOnly": False,
        "expires": NOW + 365 * 24 * 3600, "sameSite": "Lax", "domain": ".google.com", "path": "/",
    }
    assert got["__Secure-3PSID"]["sameSite"] == "None"
    assert got["__Host-GAPS"]["url"] == "https://gemini.google.com/" and "domain" not in got["__Host-GAPS"]
    assert got["NID"]["value"] == "n=x"  # only the first '=' splits


def test_curl_bash_and_cmd():
    bash = "curl 'https://gemini.google.com/app' \\\n  -H 'accept: */*' \\\n  -H 'cookie: SID=a; HSID=b' \\\n  -H 'x: y'"
    assert {c["name"] for c in parse_cookie_paste(bash, now=NOW)} == {"SID", "HSID"}
    cmd = 'curl ^"https://gemini.google.com/app^" ^\n  -b ^"SID=a; __Secure-1PSID=z^" ^\n  -H ^"x: y^"'
    assert _by_name(parse_cookie_paste(cmd, now=NOW))["__Secure-1PSID"]["value"] == "z"


def test_cookie_editor_json():
    data = [
        {"domain": ".google.com", "name": "__Secure-1PSID", "value": "v", "path": "/", "secure": True,
         "httpOnly": True, "sameSite": "unspecified", "expirationDate": 1900000000.5, "hostOnly": False},
        {"domain": "gemini.google.com", "name": "__Host-X", "value": "h", "hostOnly": True,
         "sameSite": "no_restriction"},
    ]
    got = _by_name(parse_cookie_paste(json.dumps(data), now=NOW))
    assert got["__Secure-1PSID"]["domain"] == ".google.com" and got["__Secure-1PSID"]["httpOnly"]
    assert got["__Secure-1PSID"]["expires"] == 1900000000.5 and "sameSite" not in got["__Secure-1PSID"]
    assert got["__Host-X"]["url"] == "https://gemini.google.com/" and got["__Host-X"]["sameSite"] == "None"


def test_netscape_cookies_txt():
    txt = "# Netscape HTTP Cookie File\n#HttpOnly_.google.com\tTRUE\t/\tTRUE\t1900000000\t__Secure-1PSID\tv\n"
    c = parse_cookie_paste(txt, now=NOW)[0]
    assert (c["name"], c["domain"], c["httpOnly"], c["expires"]) == ("__Secure-1PSID", ".google.com", True, 1900000000.0)


def test_session_detection_and_dedupe():
    cookies = parse_cookie_paste("NID=1; NID=2; AEC=x", now=NOW)
    assert [c["value"] for c in cookies if c["name"] == "NID"] == ["2"]
    assert not has_google_session(cookies)
    assert has_google_session(parse_cookie_paste("__Secure-1PSID=x", now=NOW))


def test_curl_without_cookie_is_an_error():
    with pytest.raises(ValueError):
        parse_cookie_paste("curl 'https://gemini.google.com/' -H 'accept: */*'")
