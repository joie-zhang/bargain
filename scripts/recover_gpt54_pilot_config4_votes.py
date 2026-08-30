#!/usr/bin/env python3
"""Recover the four missing round-three votes in the GPT-5.4 pilot.

This script replays the exact saved message payloads through the native OpenAI
API. It never imports or calls the OpenRouter client. The original failed run
is not modified; recovery responses and usage are written to a separate file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI


MISSING_AGENTS = ("Agent_6", "Agent_8", "Agent_9", "Agent_10")
INPUT_PRICE_PER_MILLION = 2.50
OUTPUT_PRICE_PER_MILLION = 15.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-dir",
        type=Path,
        default=Path("/home/jz4391/openrouter_proxy/processed"),
    )
    parser.add_argument(
        "--request-prefix",
        default="request_1788064049",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def api_key() -> str:
    value = os.getenv("JOIE_OPENAI_API_KEY_1") or os.getenv("OPENAI_API_KEY")
    if not value:
        raise RuntimeError("No native OpenAI API key is configured")
    return value


def agent_id_from_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("Saved request has no messages")
    match = re.search(r"You are (Agent_[0-9]+)", str(messages[0].get("content", "")))
    if not match:
        raise ValueError("Could not identify the agent in the saved request")
    return match.group(1)


def load_requests(request_dir: Path, prefix: str) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for path in sorted(request_dir.glob(f"{prefix}*.json")):
        payload = json.loads(path.read_text(encoding="utf-8")).get("payload")
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        agent_id = agent_id_from_messages(messages)
        if agent_id in MISSING_AGENTS and agent_id not in requests:
            requests[agent_id] = {
                "source_request_path": str(path),
                "messages": messages,
                "temperature": payload.get("temperature", 1.0),
                "max_completion_tokens": int(payload.get("max_tokens", 32768)),
            }
    missing = sorted(set(MISSING_AGENTS) - set(requests))
    if missing:
        raise RuntimeError(f"Missing saved vote requests for {missing}")
    return requests


def parse_vote_response(content: str) -> list[dict[str, Any]]:
    payload = json.loads(content)
    votes = payload.get("votes")
    if not isinstance(votes, list) or len(votes) != 10:
        raise ValueError("Expected exactly ten recovered votes")
    numbers = sorted(int(vote.get("proposal_number")) for vote in votes)
    if numbers != list(range(1, 11)):
        raise ValueError(f"Unexpected recovered proposal numbers: {numbers}")
    for vote in votes:
        if str(vote.get("vote", "")).lower() not in {"accept", "reject"}:
            raise ValueError(f"Invalid vote decision: {vote}")
    return votes


async def recover_one(
    client: AsyncOpenAI,
    agent_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model="gpt-5.4",
        messages=request["messages"],
        temperature=float(request["temperature"]),
        max_completion_tokens=int(request["max_completion_tokens"]),
        reasoning_effort="high",
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Native OpenAI returned no content for {agent_id}")
    votes = parse_vote_response(content)
    usage = response.usage.model_dump() if response.usage is not None else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    estimated_cost = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    return {
        "agent_id": agent_id,
        "provider": "openai",
        "model": "gpt-5.4",
        "source_request_path": request["source_request_path"],
        "votes": votes,
        "raw_response": content,
        "usage": usage,
        "estimated_cost_usd": estimated_cost,
    }


async def async_main(args: argparse.Namespace) -> None:
    requests = load_requests(args.request_dir, args.request_prefix)
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=60.0)
    async with AsyncOpenAI(api_key=api_key(), timeout=timeout, max_retries=0) as client:
        recovered = await asyncio.gather(
            *(recover_one(client, agent_id, requests[agent_id]) for agent_id in MISSING_AGENTS)
        )
    output = {
        "recovery_type": "native_openai_exact_saved_message_replay",
        "original_run_modified": False,
        "provider_fallback_allowed": False,
        "responses": recovered,
        "estimated_cost_usd": sum(row["estimated_cost_usd"] for row in recovered),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "agents": list(MISSING_AGENTS),
        "estimated_cost_usd": output["estimated_cost_usd"],
    }, indent=2))


def main() -> int:
    args = parse_args()
    asyncio.run(async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
