"""Admission tests use tiny local ASGI requests and no model inference."""

import asyncio
from dataclasses import replace
import json
import os
from unittest import TestCase, main, skipUnless
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi import Request
from fastapi.testclient import TestClient
from redis import Redis
from redis.exceptions import ConnectionError

from src.backend.cache.latest import LatestObservationCache
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.security.config import SecuritySettings
from src.backend.security.rate_limit import RequestRateLimiter


class RateLimiterTests(TestCase):
    def setUp(self):
        self.now = 100.0
        self.settings = SecuritySettings(requests_per_minute=3, predictions_per_minute=1)
        self.limiter = RequestRateLimiter(self.settings, clock=lambda: self.now)

    def test_prediction_quota_is_additional_and_other_ips_are_independent(self):
        self.assertEqual(self.limiter.check("client-a", True), 0)
        self.assertEqual(self.limiter.check("client-a", True), 60)
        self.assertEqual(self.limiter.check("client-a", False), 0)
        self.assertEqual(self.limiter.check("client-a", False), 0)
        self.assertEqual(self.limiter.check("client-a", False), 60)
        self.assertEqual(self.limiter.check("client-b", True), 0)

    def test_expiry_resets_counts_without_extending_window_on_each_hit(self):
        self.limiter.check("client", True)
        self.now += 30
        self.assertEqual(self.limiter.check("client", True), 30)
        self.now += 30
        self.assertEqual(self.limiter.check("client", True), 0)

    def test_local_table_is_bounded_without_evicting_active_ips(self):
        limiter = RequestRateLimiter(
            replace(self.settings, max_tracked_clients=1), clock=lambda: self.now
        )
        self.assertEqual(limiter.check("a", False), 0)
        self.assertGreater(limiter.check("b", False), 0)
        self.assertEqual(list(limiter.windows), ["a"])
        self.now += 60
        self.assertEqual(limiter.check("b", False), 0)
        self.assertEqual(list(limiter.windows), ["b"])

    def test_redis_outage_does_not_reset_existing_local_quota(self):
        client = MagicMock()
        client.eval.return_value = 0
        limiter = RequestRateLimiter(self.settings, client)
        self.assertEqual(limiter.check("a", True), 0)
        client.eval.side_effect = ConnectionError("private-password")
        with self.assertLogs("src.backend.security.rate_limit", "WARNING") as logs:
            self.assertEqual(limiter.check("a", False), 0)
        self.assertNotIn("private-password", " ".join(logs.output))
        self.assertGreater(limiter.check("a", True), 0)
        self.assertEqual(limiter.check("a", False), 0)
        self.assertGreater(limiter.check("a", False), 0)

    def test_invalid_configuration_is_rejected(self):
        for origin in (
            "*",
            "null",
            "https://example.com/path",
            "https://user:secret@example.com",
            "https://example.com:bad",
        ):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                SecuritySettings(cors_origins=(origin,))
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SecuritySettings(max_concurrent_predictions=value)
        with self.assertRaises(ValueError):
            SecuritySettings(body_timeout_seconds=float("nan"))
        with patch.dict(os.environ, {"API_RATE_LIMIT_PER_MINUTE": "bad"}, clear=True):
            with self.assertRaisesRegex(ValueError, "API_RATE_LIMIT_PER_MINUTE"):
                SecuritySettings.from_env()


class PublicApiSecurityTests(TestCase):
    def setUp(self):
        self.settings = SecuritySettings(max_body_bytes=128, body_timeout_seconds=0.02)
        self.limiter = RequestRateLimiter(self.settings)
        self.application = create_app(
            engine=MagicMock(),
            registry=StationModelRegistry([]),
            latest_cache=LatestObservationCache(),
            security_settings=self.settings,
            rate_limiter=self.limiter,
        )
        self.calls = 0

        @self.application.api_route("/probe", methods=["GET", "POST"], include_in_schema=False)
        async def probe(request: Request):
            self.calls += 1
            return {"size": len(await request.body())}

        @self.application.get("/explode")
        def explode():
            raise RuntimeError("private-password")

        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()["error"]["code"], code)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_allowlisted_origin_and_preflight(self):
        origin = "http://localhost:5173"
        preflight = self.client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["Access-Control-Allow-Origin"], origin)
        response = self.client.post("/probe", content=b"ok", headers={"Origin": origin})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
        self.assertIn("X-Request-ID", response.headers["Access-Control-Expose-Headers"])
        self.assertNotIn("Access-Control-Allow-Credentials", response.headers)

    def test_unknown_origin_is_rejected_before_route_execution(self):
        response = self.client.post(
            "/probe", content=b"ok", headers={"Origin": "https://evil.invalid"}
        )
        self.assert_error(response, 403, "ORIGIN_NOT_ALLOWED")
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        self.assertEqual(self.calls, 0)
        preflight = self.client.options(
            "/probe",
            headers={
                "Origin": "https://evil.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(preflight.status_code, 400)

    def test_size_boundary_and_oversized_body(self):
        accepted = self.client.post("/probe", content=b"x" * 128)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()["size"], 128)
        response = self.client.post(
            "/probe", content=b"x" * 129, headers={"Origin": "http://localhost:5173"}
        )
        self.assert_error(response, 413, "REQUEST_TOO_LARGE")
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertEqual(self.calls, 1)

    def test_rate_limit_is_per_ip_and_forwarded_header_cannot_bypass_it(self):
        settings = replace(self.settings, requests_per_minute=2)
        self.application.state.request_rate_limiter = RequestRateLimiter(settings)
        for address in ("1.2.3.4", "5.6.7.8"):
            self.assertEqual(
                self.client.get("/probe", headers={"X-Forwarded-For": address}).status_code, 200
            )
        response = self.client.get(
            "/probe", headers={"X-Forwarded-For": "9.9.9.9", "Origin": "http://localhost:5173"}
        )
        self.assert_error(response, 429, "RATE_LIMITED")
        self.assertGreater(int(response.headers["Retry-After"]), 0)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertEqual(self.calls, 2)

    def test_invalid_predictions_consume_prediction_quota_before_model_loading(self):
        settings = replace(self.settings, predictions_per_minute=1)
        self.application.state.request_rate_limiter = RequestRateLimiter(settings)
        first = self.client.post("/api/v1/predictions", json={})
        self.assert_error(first, 422, "INVALID_REQUEST")
        second = self.client.post("/api/v1/predictions/", json={})
        self.assert_error(second, 429, "RATE_LIMITED")
        self.assertEqual(self.application.state.prediction_services, {})

    def test_safe_route_500_has_cors_and_request_id(self):
        response = self.client.get("/explode", headers={"Origin": "http://localhost:5173"})
        self.assert_error(response, 500, "INTERNAL_ERROR")
        self.assertNotIn("private-password", response.text)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")

    def test_compressed_and_invalid_length_requests_are_rejected(self):
        response = self.client.post("/probe", content=b"x", headers={"Content-Encoding": "gzip"})
        self.assert_error(response, 415, "UNSUPPORTED_CONTENT_ENCODING")
        response = self.client.post("/probe", content=b"x", headers={"Content-Length": "-1"})
        self.assert_error(response, 400, "INVALID_REQUEST")
        self.assertEqual(self.calls, 0)

    def raw_request(self, chunks, headers=(), delay=0):
        async def call():
            messages = iter(chunks)
            sent = []

            async def receive():
                if delay:
                    await asyncio.sleep(delay)
                return next(messages, {"type": "http.disconnect"})

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/probe",
                "raw_path": b"/probe",
                "root_path": "",
                "query_string": b"",
                "headers": list(headers),
                "client": ("203.0.113.8", 1234),
                "server": ("testserver", 80),
            }
            await self.application(scope, receive, send)
            status = next(
                message["status"] for message in sent if message["type"] == "http.response.start"
            )
            payload = b"".join(
                message.get("body", b"")
                for message in sent
                if message["type"] == "http.response.body"
            )
            return status, json.loads(payload)

        return asyncio.run(call())

    def test_chunked_actual_size_is_checked_without_trusting_content_length(self):
        for headers in ((), ((b"content-length", b"1"),)):
            with self.subTest(headers=headers):
                status, payload = self.raw_request(
                    [
                        {"type": "http.request", "body": b"x" * 100, "more_body": True},
                        {"type": "http.request", "body": b"x" * 29, "more_body": False},
                    ],
                    headers,
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload["error"]["code"], "REQUEST_TOO_LARGE")
        self.assertEqual(self.calls, 0)

    def test_slow_body_times_out_before_route_execution(self):
        status, payload = self.raw_request(
            [
                {"type": "http.request", "body": b"x", "more_body": False},
            ],
            delay=0.1,
        )
        self.assertEqual(status, 408)
        self.assertEqual(payload["error"]["code"], "REQUEST_BODY_TIMEOUT")
        self.assertEqual(self.calls, 0)

    def test_openapi_documents_public_rejections(self):
        responses = self.client.get("/openapi.json").json()["paths"]["/api/v1/predictions"]["post"][
            "responses"
        ]
        self.assertTrue({"413", "429", "503"}.issubset(responses))


@skipUnless(os.environ.get("REDIS_TEST_URL"), "Set REDIS_TEST_URL for shared Redis quota checks.")
class SharedRedisRateTests(TestCase):
    def setUp(self):
        self.client = Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
        self.namespace = f"security-test:{uuid4().hex}"
        self.settings = SecuritySettings(requests_per_minute=2, predictions_per_minute=1)
        self.first = RequestRateLimiter(self.settings, self.client, namespace=self.namespace)
        self.second = RequestRateLimiter(self.settings, self.client, namespace=self.namespace)

    def tearDown(self):
        self.client.delete(*self.first.keys("client"))
        self.client.close()

    def test_separate_process_counters_share_general_and_prediction_quotas(self):
        self.assertEqual(self.first.check("client", True), 0)
        self.assertGreater(self.second.check("client", True), 0)
        self.assertEqual(self.second.check("client", False), 0)
        third = RequestRateLimiter(self.settings, self.client, namespace=self.namespace)
        self.assertGreater(third.check("client", False), 0)
        for key in self.first.keys("client"):
            self.assertTrue(0 < self.client.ttl(key) <= 60)


if __name__ == "__main__":
    main()
