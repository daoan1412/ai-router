from __future__ import annotations

from ai_router.mailtemp_web import box_link, extract_code, parse_address, pick_link


def test_box_link_matches_site_link_format():
    # Same URL as the site's "Link" button (web boxes reopen only with the token).
    assert box_link("sg5vqp6c@braetoo.cutflow.io.vn", "tok123") == (
        "https://mailtemp.vn/#addr=sg5vqp6c%40braetoo.cutflow.io.vn&t=tok123"
    )


def test_pick_link_finds_verify_link_not_page_links():
    links = [
        {"text": "API", "href": "https://mailtemp.vn/api-docs"},
        {"text": "Verify Email", "href": "https://chat.z.ai/auth/verify_email?token=verify-abc&email=x%40y.vn"},
        {"text": "support@z.ai", "href": "mailto:support@z.ai"},
    ]
    assert pick_link(links, "/auth/verify_email?token=") == links[1]["href"]
    assert pick_link(links, "/nothing") is None


def test_parse_address_joins_local_and_domain_lines():
    body = "TMail\nEmail của bạn\n8y5iyjya\n@niewirn.capteam.io.vn\nChép\nĐang chờ thư mới"
    assert parse_address(body) == "8y5iyjya@niewirn.capteam.io.vn"


def test_parse_address_does_not_absorb_copy_label():
    # Regression: globbing would have produced '...io.vnChép'.
    body = "0aj70dcj\n@cleckea.capedit.io.vn\nChép"
    assert parse_address(body) == "0aj70dcj@cleckea.capedit.io.vn"


def test_parse_address_single_line_fallback():
    assert parse_address("foo\nuser@example.com\nbar") == "user@example.com"


def test_parse_address_none_when_absent():
    assert parse_address("no mail here") is None


def test_extract_code_prefers_six_digits():
    body = "verify your account with the code below: 823820 This code will expire"
    assert extract_code(body) == "823820"


def test_extract_code_ignores_year_in_timestamp():
    # The date's 2026 must not be taken as the code.
    assert extract_code("Received 10:52:45 4/11/2026 — subject only") is None


def test_extract_code_xxx_xxx_fallback():
    assert extract_code("Your code is 123-456 now", min_digits=6) == "123-456"
