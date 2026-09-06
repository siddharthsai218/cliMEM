import asyncio
import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import (
    API_KEY_FINGERPRINT,
    CLI_TOOL,
    MODEL_FALLBACK,
    MODEL_MAP,
    PROVIDER_API_KEY,
    PROVIDER_BASE_URL,
    PROVIDER_NAME,
)
from .filter import add_message
from .memory import inject_memory
from .session import (
    ensure_session,
    get_current_session,
    mark_activity,
    save_current_session,
    start_session,
)
from .utils import get_cwd

router = APIRouter()
logger = logging.getLogger("climem")
PROVIDER_TIMEOUT_SECONDS = float(os.getenv("CLIMEM_PROVIDER_TIMEOUT_SECONDS", "30"))
if PROVIDER_TIMEOUT_SECONDS <= 0:
    raise ValueError("CLIMEM_PROVIDER_TIMEOUT_SECONDS must be positive")


@router.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    logger.debug(body)

    start_time = time.perf_counter()
    logger.info("→ %s", request.url.path)

    messages = body.get("messages", [])

    # Resolve model alias before session check so the session
    # records the real model name, not the alias.
    requested_model = body.get("model", "proxy")
    resolved_model = MODEL_MAP.get(requested_model, requested_model)
    body["model"] = resolved_model

    # Detect OpenCode's internal title generation requests.
    is_title_request = False
    if messages:
        first = messages[0]
        if (
            first.get("role") == "system"
            and "You are a title generator" in first.get("content", "")
        ):
            is_title_request = True

    if not is_title_request:
        previous_session = get_current_session()

        # Ensure a session exists and rotate if anything changed.
        rotation_reason = ensure_session(
            working_directory=str(get_cwd()),
            cli_tool=CLI_TOOL,
            provider_name=PROVIDER_NAME,
            model=resolved_model,
            api_key_fingerprint=API_KEY_FINGERPRINT,
        )

        if rotation_reason:
            await save_current_session(rotation_reason)

            start_session(
                working_directory=str(get_cwd()),
                cli_tool=CLI_TOOL,
                provider_name=PROVIDER_NAME,
                model=resolved_model,
                api_key_fingerprint=API_KEY_FINGERPRINT,
            )

        if messages:
            last = messages[-1]
            if last.get("role") == "user":
                add_message("user", last.get("content", ""))
                mark_activity()

        # Inject project memory only for real user conversations.
        body = await inject_memory(
            body,
            str(get_cwd()),
        )

    headers = {
        "Authorization": f"Bearer {PROVIDER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    client = httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS)

    # Providers like NVIDIA NIM intermittently answer with transient
    # 404/429/5xx even for valid requests. Retry those once transparently,
    # then walk MODEL_FALLBACK (a chain of backup models) in order before
    # giving up. Safe: nothing has been streamed to the client yet.
    transient = {402, 404, 408, 425, 429, 500, 502, 503, 504}
    response = None

    def _log_fallback(prev_model: str | None, next_model: str | None) -> None:
        logger.warning(
            "model %s unavailable — falling back to %s",
            prev_model, next_model or "no further fallbacks",
        )

    chain = [resolved_model] + [m for m in MODEL_FALLBACK if m != resolved_model]

    for attempt, model in enumerate(chain):
        body["model"] = model

        provider_request = client.build_request(
            "POST",
            f"{PROVIDER_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
        )
        try:
            response = await client.send(provider_request, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("provider transport failure for %s: %s", model, exc)
            if attempt < len(chain) - 1:
                _log_fallback(model, chain[attempt + 1])
                await asyncio.sleep(0.1)
                continue
            await client.aclose()
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "upstream provider unavailable", "type": "upstream_timeout"}},
            )

        if response.status_code == 200:
            break

        if response.status_code not in transient:
            # Permanent error (400/401/403...): another attempt of the SAME
            # model cannot help; a different fallback model might.
            await response.aclose()
            if attempt < len(chain) - 1:
                _log_fallback(model, chain[attempt + 1])
                continue
            break

        # Transient provider error.
        if attempt < len(chain) - 1:
            # More models left in the chain: quick backoff, then move on.
            await response.aclose()
            _log_fallback(model, chain[attempt + 1])
            await asyncio.sleep(1.5)
            continue

        # Last entry in the chain: one extra same-model retry for transient
        # errors (the original single-model behavior), then surface it.
        await asyncio.sleep(1.5)
        provider_request = client.build_request(
            "POST",
            f"{PROVIDER_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
        )
        try:
            response = await client.send(provider_request, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("provider retry transport failure for %s: %s", model, exc)
            await client.aclose()
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "upstream provider unavailable", "type": "upstream_timeout"}},
            )
        break

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info("← %d (%.0f ms)", response.status_code, elapsed)

    if response.status_code != 200:
        text = await response.aread()
        await client.aclose()
        return Response(
            content=text,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/plain"),
        )

    async def stream_generator():
        assistant_response = ""
        pending = ""

        try:
            async for chunk in response.aiter_text():
                pending += chunk
                lines = pending.splitlines(keepends=True)
                pending = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines.pop()
                for line in lines:
                    line = line.rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue

                    data = line[6:].strip()

                    if data == "[DONE]":
                        continue

                    try:
                        obj = json.loads(data)

                        choices = obj.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {}).get("content")

                        if delta:
                            assistant_response += delta

                    except json.JSONDecodeError:
                        continue

                yield chunk
        finally:
            if assistant_response:
                add_message("assistant", assistant_response)

            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )