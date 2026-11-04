"""Playwright exception types that also cover Patchright.

The browser can be driven by Playwright (CloakBrowser) or Patchright (real
Chrome); their exception classes are unrelated, so ``except`` clauses use these
tuples to catch either.
"""

from __future__ import annotations

from playwright.async_api import Error as _PwError
from playwright.async_api import TimeoutError as _PwTimeout

try:
    from patchright.async_api import Error as _PrError
    from patchright.async_api import TimeoutError as _PrTimeout
except ImportError:  # pragma: no cover - patchright is optional
    PlaywrightError: tuple[type[BaseException], ...] = (_PwError,)
    PlaywrightTimeout: tuple[type[BaseException], ...] = (_PwTimeout,)
else:
    PlaywrightError = (_PwError, _PrError)
    PlaywrightTimeout = (_PwTimeout, _PrTimeout)
