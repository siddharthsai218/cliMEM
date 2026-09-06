"""e2e.py — End-to-end test for cliMEM on cognee 1.5.3.

Orchestrates the full pipeline against a local mock OpenAI-compatible
provider (test/mock_provider.py) so no real LLM-provider credits are
spent on the proxy hop. Cognee itself runs in real local mode
(kuzu + lancedb + fastembed + the configured NIM LLM for cognify).

Phases
------
A. REMEMBER
   1. Start mock provider (port 9919) + climem server (port 8000),
      with PROVIDER_BASE_URL pointed at the mock.
   2. Send a chat whose text contains extractable facts.
   3. SIGINT the server -> graceful shutdown -> session save
      (filter -> cognee.add -> cognify -> improve).
B. RECALL
   4. Restart the server (same cwd => same dataset).
   5. Send a query chat about the same topic.
   6. Assert the forwarded request's system message contains the
      remembered context.

Run:  .venv/bin/python test/e2e.py
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PROVIDER_PORT = 9919
# Dedicated port so the test NEVER collides with a user-run climem on 8000.
CLIMEM_PORT = int(os.getenv("CLIMEM_TEST_PORT", "8123"))
MOCK_URL = f"http://127.0.0.1:{PROVIDER_PORT}"
CLIMEM_URL = f"http://127.0.0.1:{CLIMEM_PORT}"
SAVE_TIMEOUT = float(os.getenv("CLIMEM_E2E_SAVE_TIMEOUT", "45"))
POST_SAVE_TIMEOUT = float(os.getenv("CLIMEM_E2E_POST_SAVE_TIMEOUT", "3"))
CLEANUP_TIMEOUT = float(os.getenv("CLIMEM_E2E_CLEANUP_TIMEOUT", "5"))

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name}" + (f" — {detail}" if detail else "")
    print(line, flush=True)
    (PASS if ok else FAIL).append(name)


def wait_ready(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_mock() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "test.mock_provider:app",
         "--port", str(PROVIDER_PORT), "--log-level", "warning"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def start_climem(log_path: Path, workdir: Path | None = None) -> subprocess.Popen:
    # Route the proxy at the mock provider via CLIMEM_ENV_FILE (applied after
    # cognee's import-time dotenv override — see app/config.py).
    overlay = ROOT / "test" / "_overlay.env"
    overlay.write_text(
        "PROVIDER_NAME=MockProvider\n"
        "PROVIDER_API_KEY=test-key-not-real\n"
        f"PROVIDER_BASE_URL={MOCK_URL}/v1\n"
        "CLI_TOOL=opencode\n"
        # Keep Cognee fully offline and point both its chat and embedding
        # clients at the OpenAI-compatible mock provider.
        "LLM_PROVIDER=openai\n"
        "LLM_API_KEY=test-key-not-real\n"
        "OPENAI_API_KEY=test-key-not-real\n"
        "LLM_MODEL=openai/mock-chat\n"
        f"LLM_ENDPOINT={MOCK_URL}/v1\n"
        "EMBEDDING_PROVIDER=openai_compatible\n"
        "EMBEDDING_MODEL=mock-embedding\n"
        f"EMBEDDING_ENDPOINT={MOCK_URL}/v1\n"
        "EMBEDDING_API_KEY=test-key-not-real\n"
        "EMBEDDING_DIMENSIONS=384\n"
        "EMBEDDING_MAX_TOKENS=256\n"
        "VECTOR_DB_PROVIDER=lancedb\n"
        "GRAPH_DATABASE_PROVIDER=kuzu\n"
        "COGNEE_MODE=local\n"
        "CHUNKS_PER_BATCH=2000\n"
        "TRIPLET_EMBEDDING=false\n"
        "CONTRADICTION_DETECTION=false\n"
        "PROVENANCE_TRACKING=false\n"
    )
    env = os.environ.copy()
    # Cognee is imported before app.config applies CLIMEM_ENV_FILE, so expose
    # these values directly in the child environment as well. This prevents
    # Cognee's import-time settings singleton from caching empty credentials.
    env.update({
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "test-key-not-real",
        "OPENAI_API_KEY": "test-key-not-real",
        "LLM_MODEL": "openai/mock-chat",
        "LLM_ENDPOINT": f"{MOCK_URL}/v1",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": "mock-embedding",
        "EMBEDDING_ENDPOINT": f"{MOCK_URL}/v1",
        "EMBEDDING_API_KEY": "test-key-not-real",
        "EMBEDDING_DIMENSIONS": "384",
        "EMBEDDING_MAX_TOKENS": "256",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "COGNEE_MODE": "local",
        "CHUNKS_PER_BATCH": "2000",
        "TRIPLET_EMBEDDING": "false",
        "CONTRADICTION_DETECTION": "false",
        "PROVENANCE_TRACKING": "false",
    })
    env["CLIMEM_ENV_FILE"] = str(overlay)
    log = open(log_path, "ab")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(CLIMEM_PORT), "--log-level", "info"],
        cwd=str(workdir or ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    process._climem_log_handle = log
    return process


def print_stage_diagnostics(stage: str, log_path: Path) -> None:
    """Print bounded diagnostics when a Cognee stage fails or times out."""
    print(f"--- diagnostics: {stage} ---", flush=True)
    if log_path.exists():
        print(f"server log tail ({log_path}):", flush=True)
        print(log_path.read_text(errors="replace")[-6000:], flush=True)
    try:
        response = httpx.get(f"{MOCK_URL}/request-log", timeout=5)
        requests = response.json().get("requests", [])
        structured = sum(1 for item in requests if item.get("response_format"))
        embeddings = sum(1 for item in requests if "input" in item and "model" in item)
        models = [item.get("model", "?") for item in requests[-10:]]
        print(
            f"mock requests: total={len(requests)} structured={structured} "
            f"embedding-like={embeddings} recent_models={models}",
            flush=True,
        )
    except Exception as exc:
        print(f"mock diagnostics unavailable: {exc!r}", flush=True)


def stop_climem(
    proc: subprocess.Popen,
    log_path: Path,
    save_timeout: float = SAVE_TIMEOUT,
    expect_save: bool = True,
):
    """Stop a server with bounded persistence and cleanup waits."""
    proc.send_signal(signal.SIGINT)
    deadline = time.time() + (save_timeout if expect_save else POST_SAVE_TIMEOUT)
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        text = log_path.read_text(errors="replace")
        if not expect_save or "Session memory saved" in text or "No sessions recorded" in text:
            # Persistence has completed (or is not under test); do not spend
            # another 15 seconds waiting for uvicorn cleanup.
            try:
                proc.wait(timeout=POST_SAVE_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.terminate()
            break
        time.sleep(0.2)
    else:
        print_stage_diagnostics("Cognee save timeout", log_path)
        proc.terminate()
    try:
        proc.wait(timeout=CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    handle = getattr(proc, "_climem_log_handle", None)
    if handle is not None and not handle.closed:
        handle.close()
    return log_path.read_text(errors="replace")


def chat(user_text: str, model: str = "proxy") -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": user_text},
        ],
        "stream": True,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{CLIMEM_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        body = r.text
    return {"status": r.status_code, "sse": body}


FACT_USER = (
    "For the record: we decided to use PostgreSQL instead of SQLite "
    "for the climem session storage. The migration must always preserve "
    "the sessions table schema."
)
FACT_ASSISTANT_HINT = (
    "Understood. The decision was to adopt PostgreSQL for session storage. "
    "We will migrate the sessions table next sprint."
)
QUERY_USER = (
    "What database engine did we choose for the climem session storage?"
)


def main() -> int:
    print("=" * 70)
    print("cliMEM end-to-end test (cognee 1.5.3)")
    print("=" * 70)

    # Unique throwaway "project" per run. cliMEM scopes memory by working
    # directory, so a unique directory guarantees a virgin cognee dataset —
    # wiping the folder alone would NOT reset memory (the dataset lives in
    # ~/.cognee, keyed by the original path hash).
    for old in ROOT.glob("test/_sandbox_*"):
        shutil.rmtree(old, ignore_errors=True)
    sandbox = ROOT / "test" / f"_sandbox_{int(time.time())}"
    (sandbox / "src").mkdir(parents=True, exist_ok=True)
    (sandbox / "src" / "main.py").write_text("print('sandbox')\n")

    mock = start_mock()
    try:
        check("mock provider ready", wait_ready(f"{MOCK_URL}/docs"))

        # ---------------- Phase A: remember ----------------
        log_a = ROOT / "test" / "_server_a.log"
        log_a.write_bytes(b"")
        srv_a = start_climem(log_a, workdir=sandbox)
        ready_a = wait_ready(f"{CLIMEM_URL}/", timeout=240)
        check("climem server A ready", ready_a)
        if not ready_a:
            print("server A log tail:\n" + log_a.read_text(errors="replace")[-1500:])
            return 1

        resp = chat(FACT_USER)
        check("proxy forwarded streamed reply", resp["status"] == 200
              and "Acknowledged" in resp["sse"], f"status={resp['status']}")

        r = httpx.get(f"{MOCK_URL}/last-request", timeout=10)
        fwd = r.json()
        sys_content = "".join(
            m.get("content", "") for m in fwd.get("messages", [])
            if m.get("role") == "system"
        )
        check("first turn has no stale memory injection",
              "prior context remembered" not in sys_content)

        r = httpx.get(f"{CLIMEM_URL}/chat-log", timeout=10)
        chat_log = r.json()
        check("chat log captured turns", isinstance(chat_log, list)
              and len(chat_log) >= 2, f"entries={len(chat_log)}")

        print("-- shutting down server A (graceful session save) --", flush=True)
        save_started = time.perf_counter()
        text_a = stop_climem(srv_a, log_a)
        print(f"Cognee save/shutdown elapsed: {time.perf_counter() - save_started:.2f}s", flush=True)

        check("session started marker", "Session started" in text_a)
        check("shutdown save ran", "Saving session..." in text_a)
        memory_saved = "Session memory saved" in text_a
        check("memory stored via cognee", memory_saved,
              "(requires facts extracted + add/cognify/improve success)")
        if not memory_saved:
            print_stage_diagnostics("Cognee persistence", log_a)
        if "Traceback" in text_a:
            tail = text_a[text_a.rfind("Traceback"):][:1500]
            print(tail)

        # ---------------- Phase B: recall ----------------
        httpx.delete(f"{MOCK_URL}/reset", timeout=10)

        log_b = ROOT / "test" / "_server_b.log"
        log_b.write_bytes(b"")
        srv_b = start_climem(log_b, workdir=sandbox)
        check("climem server B ready", wait_ready(f"{CLIMEM_URL}/", timeout=240))

        resp = chat(QUERY_USER)
        check("query turn proxied", resp["status"] == 200)

        r = httpx.get(f"{MOCK_URL}/last-request", timeout=10)
        fwd = r.json()
        sys_content = "\n".join(
            m.get("content", "") for m in fwd.get("messages", [])
            if m.get("role") == "system"
        )
        injected = "prior context remembered" in sys_content
        recalled_pg = "postgres" in sys_content.lower()
        check("memory injected into system message", injected)
        check("recalled fact mentions PostgreSQL", recalled_pg)
        if not injected or not recalled_pg:
            print_stage_diagnostics("Cognee recall", log_b)
        check("project file tree injected",
              "main.py" in sys_content and "src" in sys_content)

        # Title-generation requests must bypass memory injection.
        httpx.delete(f"{MOCK_URL}/reset", timeout=10)
        payload = {
            "model": "proxy",
            "messages": [
                {"role": "system",
                 "content": "You are a title generator. Summarize in 5 words."},
                {"role": "user", "content": QUERY_USER},
            ],
        }
        with httpx.Client(timeout=30.0) as client:
            client.post(f"{CLIMEM_URL}/v1/chat/completions", json=payload)
        r = httpx.get(f"{MOCK_URL}/last-request", timeout=10)
        fwd = r.json()
        title_sys = "\n".join(
            m.get("content", "") for m in fwd.get("messages", [])
            if m.get("role") == "system"
        )
        check("title-generator requests skip memory injection",
              "prior context remembered" not in title_sys)

        print("-- shutting down server B --", flush=True)
        cleanup_started = time.perf_counter()
        stop_climem(srv_b, log_b, expect_save=False)
        print(f"server B cleanup elapsed: {time.perf_counter() - cleanup_started:.2f}s", flush=True)

    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    print("=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failed checks:")
        for name in FAIL:
            print(f"  - {name}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
