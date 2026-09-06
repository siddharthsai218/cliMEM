"""Process-level chaos tests using real uvicorn sockets."""

import queue
import signal
import threading
import time
import unittest

import httpx

from test.e2e import CLIMEM_URL, MOCK_URL, ROOT, start_climem, start_mock, wait_ready


class ChaosProcessTests(unittest.TestCase):
    def setUp(self):
        self.mock = start_mock()
        self.climem = None
        self.log_path = ROOT / "test" / "_chaos_process.log"
        self.log_path.unlink(missing_ok=True)
        self.assertTrue(wait_ready(f"{MOCK_URL}/failmode"), "mock provider did not start")

    @staticmethod
    def reap(process):
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except Exception:
            process.kill()
            process.wait(timeout=2)
        handle = getattr(process, "_climem_log_handle", None)
        if handle is not None and not handle.closed:
            handle.close()

    def tearDown(self):
        self.reap(self.climem)
        self.reap(self.mock)
        self.log_path.unlink(missing_ok=True)
        (ROOT / "test" / "_overlay.env").unlink(missing_ok=True)

    def start_proxy(self):
        self.climem = start_climem(self.log_path)
        self.assertTrue(wait_ready(f"{CLIMEM_URL}/"), "cliMEM did not start")

    @staticmethod
    def request_body():
        return {
            "model": "mock-chat",
            "stream": True,
            "messages": [{"role": "system", "content": "You are a title generator."}],
        }

    def stream_worker(self, events):
        try:
            with httpx.stream(
                "POST", f"{CLIMEM_URL}/v1/chat/completions",
                json=self.request_body(),
                timeout=httpx.Timeout(2.0, read=1.0),
            ) as response:
                events.put(("status", response.status_code))
                for line in response.iter_lines():
                    events.put(("line", line))
                events.put(("eof", None))
        except BaseException as exc:
            events.put(("error", exc))

    def start_stream(self):
        events = queue.Queue()
        thread = threading.Thread(target=self.stream_worker, args=(events,), daemon=True)
        thread.start()
        kind, value = events.get(timeout=5)
        self.assertEqual(kind, "status")
        self.assertEqual(value, 200)
        kind, value = events.get(timeout=5)
        self.assertEqual(kind, "line")
        self.assertIn("data:", value)
        return events, thread

    def test_upstream_socket_partition_after_first_event_is_bounded(self):
        configured = httpx.post(
            f"{MOCK_URL}/chaos/stream",
            json={"emit_chunks": 2, "delay_between_ms": 1000, "never_complete": True},
            timeout=3,
        )
        self.assertEqual(configured.status_code, 200)
        self.start_proxy()
        events, thread = self.start_stream()

        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            try:
                events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
        self.mock.kill()
        self.mock.wait(timeout=3)
        kind = None
        while time.monotonic() < deadline + 4:
            try:
                candidate, _ = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if candidate in {"error", "eof"}:
                kind = candidate
                break
        self.assertIn(kind, {"error", "eof"})
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

        self.mock = start_mock()
        self.assertTrue(wait_ready(f"{MOCK_URL}/failmode"), "mock provider did not restart")
        httpx.delete(f"{MOCK_URL}/reset", timeout=3)
        recovered = httpx.post(f"{CLIMEM_URL}/v1/chat/completions", json=self.request_body(), timeout=5)
        self.assertEqual(recovered.status_code, 200)

    def test_provider_crash_during_active_stream_allows_restart(self):
        configured = httpx.post(
            f"{MOCK_URL}/chaos/stream",
            json={"emit_chunks": 3, "delay_between_ms": 1000},
            timeout=3,
        )
        self.assertEqual(configured.status_code, 200)
        self.start_proxy()
        events, thread = self.start_stream()
        self.mock.kill()
        self.mock.wait(timeout=3)
        kind = None
        value = None
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                candidate, candidate_value = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if candidate in {"error", "eof"}:
                kind, value = candidate, candidate_value
                break
        self.assertIn(kind, {"error", "eof"})
        if kind == "error":
            self.assertIsInstance(value, BaseException)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

        self.mock = start_mock()
        self.assertTrue(wait_ready(f"{MOCK_URL}/failmode"), "mock provider did not restart")
        httpx.delete(f"{MOCK_URL}/reset", timeout=3)
        recovered = httpx.post(f"{CLIMEM_URL}/v1/chat/completions", json=self.request_body(), timeout=5)
        self.assertEqual(recovered.status_code, 200)


if __name__ == "__main__":
    unittest.main()
