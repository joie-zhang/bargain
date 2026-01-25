"""XAI proxy monitor - run on login node. Uses OpenAI-compatible API."""
import asyncio
import json
import logging
import shutil
from pathlib import Path
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

POLL_DIR = Path("/home/jz4391/xai_proxy")
PROCESSED_DIR = POLL_DIR / "processed"
POLL_INTERVAL = 1.0


async def make_request(api_key: str, model: str, messages: list, temperature: float, timeout: float) -> tuple:
    """Make a request to XAI API and return (content, usage_data)."""
    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        if resp.status != 200:
            raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage


async def handle_request(request_path: Path):
    suffix = request_path.stem.removeprefix("request_")
    response_path = POLL_DIR / f"response_{suffix}.json"
    log.info(f"Processing {request_path.name}")

    try:
        with open(request_path) as f:
            req = json.load(f)
        content, usage = await make_request(req['api_key'], req['model'], req['messages'], req.get('temperature', 0.7), req.get('timeout', 300))
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
        log.info(f"Success {suffix}")
    except Exception as e:
        response = {"result": None, "error": f"{type(e).__name__}: {e}", "usage": None}
        log.error(f"Failed {suffix}: {e}")

    with open(response_path, 'w') as f:
        json.dump(response, f)
    shutil.move(request_path, PROCESSED_DIR / request_path.name)


async def main():
    POLL_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    log.info(f"Polling {POLL_DIR}")
    while True:
        for p in POLL_DIR.glob("request_*.json"):
            await handle_request(p)
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
