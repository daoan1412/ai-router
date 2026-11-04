from dataclasses import dataclass


@dataclass
class AiRouterError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class NotLoggedInError(AiRouterError):
    def __init__(self, message: str = "Not logged in. Run: ai browser login"):
        super().__init__("NOT_LOGGED_IN", message)


class ProviderNotReadyError(AiRouterError):
    def __init__(self, provider: str):
        super().__init__(
            "PROVIDER_NOT_READY",
            f"Provider '{provider}' is not implemented yet",
        )


class BrowserBusyError(AiRouterError):
    def __init__(self):
        super().__init__("BROWSER_BUSY", "Browser is busy with another request")


class BrowserClosedError(AiRouterError):
    def __init__(self):
        super().__init__(
            "BROWSER_CLOSED",
            "Browser window was closed. Keep the headed browser open while ai serve is running, "
            "or restart: uv run ai-router serve",
        )


class TimeoutError_(AiRouterError):
    def __init__(self, message: str = "Answer did not arrive in time"):
        super().__init__("TIMEOUT", message)


class RateLimitedError(AiRouterError):
    def __init__(self, message: str = "Rate limit reached. Try again later"):
        super().__init__("RATE_LIMITED", message)
