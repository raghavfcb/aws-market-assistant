from __future__ import annotations

from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError

from market_partner.adapters.sarvam import SarvamAPIError
from market_partner.errors import ProviderError

THROTTLE_CODES = frozenset({"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"})
TRANSIENT_CODES = frozenset(
    {"ServiceUnavailableException", "ModelTimeoutException", "ModelNotReadyException", "InternalServerException"}
)


def classify_bedrock_error(error: Exception) -> ProviderError:
    """Map boto3 failures to a ProviderError without leaking request details."""
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        headers = error.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        retry_after = headers.get("retry-after", "")
        wait = int(retry_after) if retry_after.isdigit() else None
        if code in THROTTLE_CODES:
            return ProviderError("bedrock", "Bedrock is throttling requests.", throttled=True, retry_after_seconds=wait)
        if code in TRANSIENT_CODES:
            return ProviderError("bedrock", "Bedrock is temporarily unavailable.", transient=True, retry_after_seconds=wait)
        return ProviderError("bedrock", f"Bedrock rejected the request ({code or 'unknown'}).")
    if isinstance(error, (ReadTimeoutError, ConnectTimeoutError)):
        return ProviderError("bedrock", "Bedrock did not respond in time.", transient=True)
    if isinstance(error, BotoCoreError):
        return ProviderError("bedrock", "Bedrock could not be reached.", transient=True)
    return ProviderError("bedrock", "Bedrock returned an unusable response.")


def classify_sarvam_error(error: SarvamAPIError) -> ProviderError:
    return ProviderError(
        "sarvam",
        str(error),
        throttled=error.throttled,
        transient=error.transient,
        retry_after_seconds=error.retry_after_seconds,
    )
