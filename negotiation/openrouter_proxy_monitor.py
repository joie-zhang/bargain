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

async def prompt_openrouter(url, headers, payload, timeout) -> str:
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
        data = await response.json(encoding='utf-8')
        print("\n")
        print(f"url={url}, headers={headers}, payload={payload}, timeout={timeout}")
        print(f"{data}")
        print("\n")
        return data["choices"][0]["message"]["content"]

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