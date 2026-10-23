class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.code = code
        self.safe_message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class ProviderError(RuntimeError):
    """A model-provider failure that the caller may route around by trying another provider."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        throttled: bool = False,
        transient: bool = False,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.provider = provider
        self.throttled = throttled
        self.transient = transient or throttled
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


def providers_exhausted(failures: list[ProviderError], default_retry_after: int) -> DomainError:
    """Turn the failures of every configured provider into one safe, user-facing error."""
    if failures and all(failure.transient for failure in failures):
        waits = [f.retry_after_seconds for f in failures if f.retry_after_seconds]
        wait = max(waits) if waits else default_retry_after
        return DomainError(
            "providers_busy",
            f"The AI services are receiving too many requests right now. Please wait about {wait} seconds and try again.",
            429,
            retry_after_seconds=wait,
        )
    return DomainError(
        "providers_unavailable",
        "The AI services could not complete this request. Check the provider configuration and try again.",
        502,
    )
