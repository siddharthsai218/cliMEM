# cliMEM Streaming SSE Chaos Test Plan

## Purpose

This suite validates that cliMEM remains bounded, observable, and recoverable when the upstream provider becomes unavailable during an active Server-Sent Events (SSE) response. The suite must distinguish a clean client cancellation from an upstream partition and from a crashed cliMEM process. It should run deterministically in CI with fixed seeds, bounded deadlines, and no external provider credentials.

## Fault model

| Fault | Injection point | Expected client behavior | Expected server behavior |
|---|---|---|---|
| Delayed first chunk | Mock provider before first SSE event | Client remains within its configured deadline or receives a timeout | Upstream task is cancelled; no leaked request/client remains |
| Delayed subsequent chunk | Mock provider after one or more SSE events | Client receives a valid prefix, then a bounded disconnect/timeout | Partial assistant text may be recorded only according to the documented policy |
| Upstream half-partition | Provider sends headers and one event, then stops producing bytes | Client request terminates by deadline, never hangs indefinitely | Provider response and HTTP client are closed in `finally` |
| Upstream hard disconnect | Provider closes the stream without `[DONE]` | Client sees an incomplete stream or transport error | No retry after bytes have reached the client; cleanup remains bounded |
| cliMEM crash during stream | Kill cliMEM process after first upstream event | Client gets a transport failure and can reconnect | Restarted cliMEM accepts a fresh request; no stale process or port remains |
| Provider crash during stream | Kill mock provider after first event | Client gets a transport failure | cliMEM exits the active stream and can recover after provider restart |
| Partition during fallback | Primary fails before bytes; fallback remains available | Client receives fallback stream | Fallback is allowed only before any downstream bytes are emitted |
| Cancellation by client | Client closes response after first event | No error is reported as a provider outage | Upstream response and client are closed; no zombie task remains |

## Test harness architecture

The suite should use two layers. The first layer invokes the ASGI application directly with a controllable upstream transport for fast deterministic tests. The second layer starts real uvicorn processes for process-crash and socket-level partition tests.

The mock provider should expose test-only stream controls through an authenticated-free local endpoint, for example:

```json
POST /chaos/stream
{
  "model": "chaos-model",
  "delay_before_ms": 0,
  "delay_between_ms": 50,
  "emit_chunks": 3,
  "disconnect_after": 1,
  "never_complete": false
}
```

The endpoint must validate positive integer delays, non-negative disconnect positions, and a bounded maximum stream duration. `DELETE /chaos/stream` should restore the normal one-chunk response. These controls belong only in `test/mock_provider.py`; they must never be enabled by production configuration.

For process tests, the harness should allocate ports dynamically, write each child’s stdout and stderr to separate files, and always terminate process groups in a `finally` block. A test-wide deadline of 30 seconds and per-request deadlines of 1–3 seconds prevent a fault from hanging CI.

## Core scenarios

### 1. Partial upstream partition after the first event

Configure three chunks with a 10 ms inter-chunk delay and disconnect after chunk one. Start a streaming chat request, assert that the first SSE event is valid, then assert that the request terminates within its deadline without a `[DONE]` event. Verify that the upstream response is closed and that the mock request log contains exactly one request.

### 2. Silent upstream after headers

Configure the provider to send response headers but no event for 200 ms. Run the client with a 50 ms deadline. Assert a timeout, then immediately issue a normal request and verify it succeeds. This proves that a timed-out request does not poison the shared client or provider state.

### 3. Provider crash during an active stream

Start mock provider and cliMEM as separate processes. Begin a stream, wait until the first event is observed, terminate the provider process, and assert that the client exits within 2 seconds. Restart the provider and issue a fresh request; it must succeed. Check that no child process retains the old provider port.

### 4. cliMEM crash during an active stream

Start both processes, begin a stream, consume the first event, and terminate the cliMEM process. The client must observe a transport failure or EOF within 2 seconds. Restart cliMEM with the same sandbox and verify that a new request succeeds without stale session corruption.

### 5. Client cancellation after a partial response

Consume exactly one SSE event and close the response context without reading the remainder. Assert that the request completes cleanup within 1 second. Run a follow-up request and verify normal streaming. This scenario must not be classified as a provider failure and must not trigger fallback after downstream bytes have been emitted.

### 6. Fallback partition before downstream bytes

Configure the primary model to return a transient error before streaming and the secondary model to return a valid stream. Verify one fallback response and no partial primary SSE reaches the client. Repeat with the primary returning one streamed event before disconnecting; verify that fallback is not attempted after downstream output begins.

### 7. Concurrent chaos mix

Run 100 requests concurrently with a fixed seed: 60 normal streams, 15 delayed-first-event streams, 10 post-first-event disconnects, 10 client cancellations, and 5 provider failures. Require every request to end in one of the explicitly allowed outcomes. Verify that the request log remains capped and that subsequent normal requests have a 100% recovery rate.

## Assertions and invariants

| Invariant | Check |
|---|---|
| Bounded completion | Every request ends before its per-case deadline; the process-level deadline is always enforced. |
| No retry after output | Once any downstream `data:` event is emitted, cliMEM must not invoke another model or fallback path. |
| Resource cleanup | Upstream response, `httpx.AsyncClient`, stream generator, and subprocesses are closed after success, cancellation, timeout, disconnect, and crash. |
| Recovery | After restoring the provider, at least 10 consecutive normal requests succeed. |
| State isolation | A failed partial stream does not inject incomplete content into a subsequent unrelated request. |
| Diagnostics | Failure output includes fault configuration, elapsed time, received event count, child exit codes, and bounded log tails. |
| Privacy | Chaos diagnostics contain metadata only; prompt and response bodies are not persisted in the request log. |

## Metrics

Each test should record total latency, time to first event, events received, bytes received, timeout classification, and cleanup latency. The concurrent scenario should additionally report success/error counts and p50/p95/p99 latency for normal requests. A useful regression threshold is that cleanup p95 remains below 1 second for ASGI tests and below 3 seconds for process tests.

The suite should not assert an unrealistically low total throughput for process-level tests. Instead, it should compare the faulted request’s cleanup time with a baseline normal stream and fail only on unbounded growth, leaked processes, or materially increased tail latency.

## CI integration

Add the chaos suite as a separate workflow step after the normal mock-provider tests and before the Cognee e2e test. Upload bounded provider and cliMEM logs on failure. Keep the process-crash cases in a dedicated job if they make the standard job exceed two minutes. The suite should run on Linux with Python 3.12 and use only local services.

The first implementation should land in `test/test_chaos_streaming.py`, with reusable helpers for starting processes, waiting for readiness, reading the first SSE event, killing a process group, and asserting bounded cleanup. The normal unit suite should continue to test schema generation independently; chaos tests should focus on stream lifecycle and fault recovery rather than Cognee persistence.

## Known implementation gap to address before enforcing the suite

The current proxy creates an `httpx.AsyncClient(timeout=None)` and parses each received network chunk independently. A real partition can split an SSE line across chunks, so the production proxy should first gain a persistent line buffer and an explicit upstream timeout policy. The chaos suite should initially mark split-line handling as an expected failure or add the buffering fix before making that case a required CI gate.
