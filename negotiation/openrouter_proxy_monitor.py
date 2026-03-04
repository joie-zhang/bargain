import asyncio
import json
import logging
import shutil
import os
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

POLL_DIR = Path("/home/jz4391/openrouter_proxy")
PROCESSED_DIR = POLL_DIR / "processed"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning(f"Invalid {name}={raw!r}; using default={default}")
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning(f"Invalid {name}={raw!r}; using default={default}")
        return default


# Lower poll interval improves queue pickup latency under high load.
POLL_INTERVAL = max(0.01, _float_env("OPENROUTER_PROXY_POLL_INTERVAL", 0.1))

# High-throughput defaults; allow explicit override.
# OPENROUTER_PROXY_MAX_CONCURRENCY=0 means "unbounded".
DEFAULT_MAX_CONCURRENCY = max(64, min(512, (os.cpu_count() or 8) * 16))
MAX_CONCURRENT_REQUESTS = _int_env("OPENROUTER_PROXY_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)

# aiohttp connector limit: 0 means unlimited connections.
HTTP_MAX_CONNECTIONS = max(0, _int_env("OPENROUTER_PROXY_HTTP_MAX_CONNECTIONS", 0))

def extract_content_from_response(data: dict, model_id: str = None) -> str:
    """
    Extract content from OpenRouter response, handling various formats.

    Different models return content in different structures:
    - Standard: choices[0].message.content
    - DeepSeek-R1: May have reasoning_content alongside content
    - Some models: May return empty content with finish_reason
    """
    if "error" in data:
        error_msg = data.get("error", {})
        if isinstance(error_msg, dict):
            raise Exception(f"API Error: {error_msg.get('message', error_msg)}")
        raise Exception(f"API Error: {error_msg}")

    if "choices" not in data or not data["choices"]:
        raise Exception(f"No choices in response: {json.dumps(data)[:500]}")

    choice = data["choices"][0]
    message = choice.get("message", {})

    # Standard content field
    content = message.get("content")

    # Handle reasoning models (DeepSeek-R1, etc.)
    # They may have content=None but reasoning_content populated
    if content is None or content == "":
        # Check for reasoning_content (DeepSeek-R1 specific)
        reasoning = message.get("reasoning_content")
        if reasoning:
            log.warning(f"Model returned reasoning_content instead of content")
            content = reasoning

    # OpenAI reasoning models via OpenRouter can return content=None with
    # reasoning_details summaries. Use these summaries as best-effort fallback.
    if content is None or content == "":
        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            summary_parts = []
            for entry in reasoning_details:
                if not isinstance(entry, dict):
                    continue
                summary = entry.get("summary")
                if isinstance(summary, str) and summary.strip():
                    summary_parts.append(summary.strip())
                    continue
                if isinstance(summary, list):
                    for chunk in summary:
                        if not isinstance(chunk, dict):
                            continue
                        text = chunk.get("text")
                        if isinstance(text, str) and text.strip():
                            summary_parts.append(text.strip())
            if summary_parts:
                log.warning(
                    f"Model returned no content; using reasoning_details fallback "
                    f"(parts={len(summary_parts)})"
                )
                content = "\n".join(summary_parts)

    # Some models return content as empty string with finish_reason
    if content is None:
        finish_reason = choice.get("finish_reason")
        raise Exception(
            f"Empty content from model. finish_reason={finish_reason}, "
            f"message keys={list(message.keys())}, "
            f"full response: {json.dumps(data)[:1000]}"
        )

    if content == "":
        finish_reason = choice.get("finish_reason")
        log.warning(f"Model returned empty string content. finish_reason={finish_reason}")
        raise Exception(
            f"Model returned empty content string. finish_reason={finish_reason}, "
            f"full response: {json.dumps(data)[:1000]}"
        )

    return content


async def prompt_openrouter(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    timeout: float,
) -> tuple:
    """Make OpenRouter request and return (content, usage) tuple."""
    model_id = payload.get("model", "unknown")
    response = await session.post(
        url,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout)
    )

    # Check HTTP status
    if response.status != 200:
        error_text = await response.text()
        # Provide more helpful error messages for authentication issues
        if response.status == 401:
            try:
                error_data = await response.json()
                error_msg = error_data.get("error", {}).get("message", error_text)
                raise Exception(
                    f"HTTP 401: {error_msg}\n"
                    f"This usually means the OpenRouter API key is invalid, expired, or the account is suspended.\n"
                    f"Please verify your API key at https://openrouter.ai/keys"
                )
            except Exception:
                pass
        raise Exception(f"HTTP {response.status}: {error_text[:500]}")

    data = await response.json(encoding='utf-8')

    # Debug logging
    log.info(f"Model: {model_id}")
    log.debug(f"Request payload: {json.dumps(payload)[:200]}")
    log.debug(f"Response: {json.dumps(data)[:500]}")

    content = extract_content_from_response(data, model_id)
    usage = data.get("usage", {})
    return content, usage

async def process_request(session: aiohttp.ClientSession, request_json_fpath) -> tuple:
    """Process request file and return (content, usage) tuple."""
    with open(request_json_fpath, 'r') as f:
        request_json = json.load(f)
    url, headers, payload, timeout = request_json['url'], request_json['headers'], request_json['payload'], request_json['timeout']
    return await prompt_openrouter(session, url, headers, payload, timeout)

async def handle_request(session: aiohttp.ClientSession, request_path: Path):
    suffix = request_path.stem.removeprefix("request_")
    response_path = POLL_DIR / f"response_{suffix}.json"
    log.info(f"Processing {request_path.name}")

    try:
        content, usage = await process_request(session, str(request_path))
        response = {
            "result": content,
            "error": None,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "total_tokens": usage.get("total_tokens")
            }
        }
        log.info(f"Success {suffix} ({len(content)} chars)")
    except Exception as e:
        response = {"result": None, "error": f"{type(e).__name__}: {e}", "usage": None}
        log.error(f"Failed {suffix}: {type(e).__name__}: {e}")

    with open(response_path, 'w') as f:
        json.dump(response, f)

    # Request may already be moved/removed by another worker/process.
    if request_path.exists():
        shutil.move(request_path, PROCESSED_DIR / request_path.name)


async def _run_and_cleanup(
    task_map: dict,
    session: aiohttp.ClientSession,
    request_path: Path,
):
    """Run one request task and always release it from the active map."""
    try:
        await handle_request(session, request_path)
    finally:
        task_map.pop(request_path, None)

async def main():
    PROCESSED_DIR.mkdir(exist_ok=True)
    max_concurrency_label = (
        "unbounded" if MAX_CONCURRENT_REQUESTS <= 0 else str(MAX_CONCURRENT_REQUESTS)
    )
    log.info(
        f"Polling {POLL_DIR} every {POLL_INTERVAL}s "
        f"(max_concurrency={max_concurrency_label}, http_conn_limit={HTTP_MAX_CONNECTIONS})"
    )

    active_tasks = {}
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            # File was moved/deleted concurrently.
            return float("inf")

    connector = aiohttp.TCPConnector(force_close=False, limit=HTTP_MAX_CONNECTIONS)
    async with aiohttp.ClientSession(
        connector=connector,
        json_serialize=lambda x: json.dumps(x, ensure_ascii=False),
    ) as session:
        while True:
            # Sort by mtime so oldest requests get launched first.
            request_paths = sorted(
                POLL_DIR.glob("request_*.json"),
                key=_safe_mtime
            )

            for request_path in request_paths:
                if not request_path.exists():
                    continue
                if request_path in active_tasks:
                    continue
                if MAX_CONCURRENT_REQUESTS > 0 and len(active_tasks) >= MAX_CONCURRENT_REQUESTS:
                    break
                task = asyncio.create_task(
                    _run_and_cleanup(active_tasks, session, request_path)
                )
                active_tasks[request_path] = task

            # Reap completed tasks so exceptions surface in logs.
            done = [p for p, t in active_tasks.items() if t.done()]
            for p in done:
                t = active_tasks.pop(p, None)
                if t is not None:
                    try:
                        await t
                    except Exception as exc:
                        log.error(f"Unhandled task error for {p.name}: {exc}")

            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
