"""Deterministic ASGI chaos tests for streaming provider lifecycle behavior."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse

from app import proxy


class FakeProviderResponse:
    def __init__(self, chunks):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.chunks = chunks
        self.closed = False

    async def aiter_text(self):
        for chunk in self.chunks:
            yield chunk

    async def aread(self):
        return b""

    async def aclose(self):
        self.closed = True


class FakeProviderClient:
    def __init__(self, response=None, error=None, sequence=None):
        self.response = response
        self.error = error
        self.sequence = list(sequence or [])
        self.closed = False

    def build_request(self, *args, **kwargs):
        return httpx.Request(*args, **kwargs)

    async def send(self, request, stream=True):
        item = self.sequence.pop(0) if self.sequence else self.response
        if isinstance(item, BaseException):
            raise item
        if item is not None:
            return item
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self):
        self.closed = True


def make_request(body: dict) -> Request:
    raw = json.dumps(body).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": raw, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    return Request(scope, receive=receive)


class ChaosStreamingASGITests(unittest.IsolatedAsyncioTestCase):
    async def test_split_sse_line_is_reassembled_and_recorded(self):
        response = FakeProviderResponse([
            'data: {"choices":[{"delta":{"content":"Hel',
            'lo"}}]}\n\ndata: [DONE]\n\n',
        ])
        client = FakeProviderClient(response=response)
        recorded = []
        body = {"model": "mock", "stream": True, "messages": [{"role": "system", "content": "You are a title generator."}]}

        with patch.object(proxy.httpx, "AsyncClient", return_value=client), patch.object(proxy, "add_message", side_effect=lambda role, content: recorded.append((role, content))):
            result = await proxy.chat(make_request(body))
            self.assertIsInstance(result, StreamingResponse)
            streamed = []
            async for chunk in result.body_iterator:
                streamed.append(chunk)

        self.assertEqual("".join(recorded[i][1] for i in range(len(recorded))), 'Hello')
        self.assertTrue(response.closed)
        self.assertTrue(client.closed)
        self.assertIn("data:", "".join(streamed))

    async def test_client_cancellation_closes_upstream_resources(self):
        response = FakeProviderResponse(['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'])
        client = FakeProviderClient(response=response)
        body = {"model": "mock", "stream": True, "messages": [{"role": "system", "content": "You are a title generator."}]}

        with patch.object(proxy.httpx, "AsyncClient", return_value=client):
            result = await proxy.chat(make_request(body))
            iterator = result.body_iterator
            first = await anext(iterator)
            self.assertIn("data:", first)
            await iterator.aclose()

        self.assertTrue(response.closed)
        self.assertTrue(client.closed)

    async def test_upstream_timeout_returns_bounded_504_and_closes_client(self):
        client = FakeProviderClient(error=httpx.ReadTimeout("simulated partition"))
        body = {"model": "mock", "stream": True, "messages": [{"role": "system", "content": "You are a title generator."}]}

        with patch.object(proxy.httpx, "AsyncClient", return_value=client):
            with patch.object(proxy, "MODEL_FALLBACK", []):
                result = await proxy.chat(make_request(body))

        self.assertEqual(result.status_code, 504)
        self.assertTrue(client.closed)

    async def test_timeout_can_fallback_before_any_downstream_bytes(self):
        recovered = FakeProviderResponse(['data: {"choices":[{"delta":{"content":"recovered"}}]}\n\ndata: [DONE]\n\n'])
        client = FakeProviderClient(sequence=[httpx.ReadTimeout("simulated partition"), recovered])
        body = {"model": "primary", "stream": True, "messages": [{"role": "system", "content": "You are a title generator."}]}

        with patch.object(proxy.httpx, "AsyncClient", return_value=client), patch.object(proxy, "MODEL_FALLBACK", ["secondary"]):
            result = await proxy.chat(make_request(body))
            self.assertEqual(result.status_code, 200)
            chunks = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)

        self.assertIn("recovered", "".join(chunks))
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
