import asyncio
import json
import logging
import shutil
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
POLL_INTERVAL = 1.0

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


async def prompt_openrouter(url, headers, payload, timeout) -> str:
    model_id = payload.get("model", "unknown")
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(
        connector=connector,
        json_serialize=lambda x: json.dumps(x, ensure_ascii=False)
    ) as session:
        response = await session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        )

        # Check HTTP status
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"HTTP {response.status}: {error_text[:500]}")

        data = await response.json(encoding='utf-8')

        # Debug logging
        log.info(f"Model: {model_id}")
        log.debug(f"Request payload: {json.dumps(payload)[:200]}")
        log.debug(f"Response: {json.dumps(data)[:500]}")

        return extract_content_from_response(data, model_id)

async def process_request(request_json_fpath) -> str:
    with open(request_json_fpath, 'r') as f:
        request_json = json.load(f)
    url, headers, payload, timeout = request_json['url'], request_json['headers'], request_json['payload'], request_json['timeout']
    return await prompt_openrouter(url, headers, payload, timeout)

async def handle_request(request_path: Path):
    suffix = request_path.stem.removeprefix("request_")
    response_path = POLL_DIR / f"response_{suffix}.json"
    log.info(f"Processing {request_path.name}")

    result = None
    try:
        result = await process_request(str(request_path))
        response = {"result": result, "error": None}
        log.info(f"Success {suffix} ({len(result)} chars)")
    except Exception as e:
        response = {"result": None, "error": f"{type(e).__name__}: {e}"}
        log.error(f"Failed {suffix}: {type(e).__name__}: {e}")

    with open(response_path, 'w') as f:
        json.dump(response, f)

    shutil.move(request_path, PROCESSED_DIR / request_path.name)

async def main():
    PROCESSED_DIR.mkdir(exist_ok=True)
    log.info(f"Polling {POLL_DIR} every {POLL_INTERVAL}s")

    while True:
        for request_path in POLL_DIR.glob("request_*.json"):
            await handle_request(request_path)
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())