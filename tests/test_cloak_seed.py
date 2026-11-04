from __future__ import annotations

from ai_router.browser.cloak import SEED_FILE, stable_stealth_args


def _seed(args: list[str]) -> str:
    return next(a for a in args if a.startswith("--fingerprint=")).split("=", 1)[1]


def test_seed_is_pinned_per_profile(tmp_path):
    first = stable_stealth_args(str(tmp_path / "p1"))
    assert _seed(first).isdigit()
    assert (tmp_path / "p1" / SEED_FILE).read_text() == _seed(first)
    # Every later launch of the same profile reuses it (CloakBrowser alone re-rolls).
    assert all(_seed(stable_stealth_args(str(tmp_path / "p1"))) == _seed(first) for _ in range(5))


def test_saved_seed_wins_and_other_args_kept(tmp_path):
    (tmp_path / SEED_FILE).write_text("12345")
    args = stable_stealth_args(str(tmp_path))
    assert _seed(args) == "12345"
    assert "--no-sandbox" in args
