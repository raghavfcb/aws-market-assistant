from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from botocore.exceptions import ClientError, ReadTimeoutError

from market_partner.adapters.bedrock_errors import classify_bedrock_error
from market_partner.adapters.sarvam import SarvamAPIError
from market_partner.config import Settings
from market_partner.conversation import MCPConversationHost
from market_partner.errors import DomainError, ProviderError, providers_exhausted
from market_partner.models import ChatRequest
from market_partner.workflows.research import FallbackResearchModel


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")


class StubModel:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def analyze(self, role: str, question: str, sources: list[Any]) -> dict[str, Any]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class ClassificationTests(unittest.TestCase):
    def test_throttling_is_transient_and_throttled(self) -> None:
        failure = classify_bedrock_error(client_error("ThrottlingException"))
        self.assertTrue(failure.throttled)
        self.assertTrue(failure.transient)

    def test_timeout_is_transient(self) -> None:
        failure = classify_bedrock_error(ReadTimeoutError(endpoint_url="https://example.invalid"))
        self.assertTrue(failure.transient)
        self.assertFalse(failure.throttled)

    def test_access_denied_is_not_transient(self) -> None:
        failure = classify_bedrock_error(client_error("AccessDeniedException"))
        self.assertFalse(failure.transient)

    def test_sarvam_429_is_throttled(self) -> None:
        error = SarvamAPIError("busy", status_code=429, retry_after_seconds=12)
        self.assertTrue(error.throttled)
        self.assertEqual(error.retry_after_seconds, 12)


class ExhaustedTests(unittest.TestCase):
    def test_all_transient_asks_user_to_wait(self) -> None:
        error = providers_exhausted(
            [ProviderError("bedrock", "t", throttled=True), ProviderError("sarvam", "t", transient=True)], 30
        )
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_seconds, 30)
        self.assertIn("30 seconds", error.safe_message)

    def test_provider_supplied_wait_wins(self) -> None:
        error = providers_exhausted([ProviderError("sarvam", "t", throttled=True, retry_after_seconds=45)], 30)
        self.assertEqual(error.retry_after_seconds, 45)

    def test_misconfiguration_is_not_reported_as_busy(self) -> None:
        error = providers_exhausted([ProviderError("bedrock", "denied")], 30)
        self.assertEqual(error.status_code, 502)
        self.assertIsNone(error.retry_after_seconds)


class ResearchFallbackTests(unittest.TestCase):
    def test_falls_back_on_throttling(self) -> None:
        primary = StubModel(error=client_error("ThrottlingException"))
        backup = StubModel(result={"summary": "ok"})
        model = FallbackResearchModel([primary, backup], default_retry_after=30)  # type: ignore[list-item]
        self.assertEqual(model.analyze("r", "q", []), {"summary": "ok"})
        self.assertEqual((primary.calls, backup.calls), (1, 1))

    def test_falls_back_on_bad_json(self) -> None:
        primary = StubModel(error=ValueError("The research model did not return valid JSON"))
        backup = StubModel(result={"summary": "ok"})
        model = FallbackResearchModel([primary, backup], default_retry_after=30)  # type: ignore[list-item]
        self.assertEqual(model.analyze("r", "q", []), {"summary": "ok"})

    def test_primary_success_skips_backup(self) -> None:
        primary = StubModel(result={"summary": "ok"})
        backup = StubModel(result={"summary": "backup"})
        model = FallbackResearchModel([primary, backup], default_retry_after=30)  # type: ignore[list-item]
        self.assertEqual(model.analyze("r", "q", []), {"summary": "ok"})
        self.assertEqual(backup.calls, 0)

    def test_both_throttled_raises_wait_error(self) -> None:
        primary = StubModel(error=client_error("ThrottlingException"))
        backup = StubModel(error=SarvamAPIError("busy", status_code=429))
        model = FallbackResearchModel([primary, backup], default_retry_after=30)  # type: ignore[list-item]
        with self.assertRaises(DomainError) as caught:
            model.analyze("r", "q", [])
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after_seconds, 30)


class ChatFallbackTests(unittest.TestCase):
    def host(self, **overrides: Any) -> MCPConversationHost:
        settings = Settings(
            bedrock_enabled=True,
            bedrock_model_id="test-model",
            sarvam_enabled=True,
            sarvam_api_key="key",  # type: ignore[arg-type]
            **overrides,
        )
        return MCPConversationHost(settings)

    def respond(self, host: MCPConversationHost) -> Any:
        return asyncio.run(host.respond(ChatRequest(message="hello there")))

    def test_bedrock_first_then_sarvam_on_throttle(self) -> None:
        host = self.host()
        sarvam_result = ("sarvam reply", [], None, None)
        with (
            patch.object(host, "_discover", AsyncMock(return_value=[])),
            patch.object(host, "_bedrock_response", AsyncMock(side_effect=client_error("ThrottlingException"))),
            patch.object(host, "_sarvam_response", AsyncMock(return_value=sarvam_result)),
        ):
            response = self.respond(host)
        self.assertEqual(response.mode, "sarvam")
        self.assertEqual(response.reply, "sarvam reply")

    def test_bedrock_success_never_calls_sarvam(self) -> None:
        host = self.host()
        sarvam = AsyncMock()
        with (
            patch.object(host, "_discover", AsyncMock(return_value=[])),
            patch.object(host, "_bedrock_response", AsyncMock(return_value=("bedrock reply", [], None))),
            patch.object(host, "_sarvam_response", sarvam),
        ):
            response = self.respond(host)
        self.assertEqual(response.mode, "bedrock")
        sarvam.assert_not_called()

    def test_both_throttled_returns_wait_error(self) -> None:
        host = self.host(provider_retry_after_seconds=20)
        with (
            patch.object(host, "_discover", AsyncMock(return_value=[])),
            patch.object(host, "_bedrock_response", AsyncMock(side_effect=client_error("ThrottlingException"))),
            patch.object(host, "_sarvam_response", AsyncMock(side_effect=SarvamAPIError("busy", status_code=429))),
        ):
            with self.assertRaises(DomainError) as caught:
                self.respond(host)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after_seconds, 20)


if __name__ == "__main__":
    unittest.main()
