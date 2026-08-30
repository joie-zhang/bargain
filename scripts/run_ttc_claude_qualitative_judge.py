#!/usr/bin/env python3
"""Judge prepared Claude TTC rollouts with GPT-5.5-xhigh via OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "analysis/ttc_claude_seed_qualitative_adjudication_20260728"
MANIFEST_PATH = ROOT / "all_available_rollouts_manifest.jsonl"
CODEBOOK_PATH = ROOT / "llm_tag_codebook_full50.json"
OUTPUT_ROOT = ROOT / "judge_outputs"
DEFAULT_MODEL = "openai/gpt-5.5"
DEFAULT_PROXY_DIR = Path("/home/jz4391/openrouter_proxy")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_SOURCE_KINDS = {"conversation_log", "interaction", "formal_outcome"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def compact_codebook(codebook: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tag_code": row["tag_code"],
            "tag_title": row["tag_title"],
            "category": row["category"],
            "definition": row["definition"],
            "scope_hint": row["scope_hint"],
        }
        for row in codebook
    ]


def output_schema(tag_codes: list[str]) -> dict[str, Any]:
    nullable_integer = {"type": ["integer", "null"]}
    nullable_string = {"type": ["string", "null"]}
    return {
        "name": "ttc_target_strategic_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rollout_id": {"type": "string"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tag_code": {"type": "string", "enum": tag_codes},
                            "source_kind": {
                                "type": "string",
                                "enum": sorted(VALID_SOURCE_KINDS),
                            },
                            "log_index": nullable_integer,
                            "interaction_index": nullable_integer,
                            "quote": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": sorted(VALID_CONFIDENCES),
                            },
                        },
                        "required": [
                            "tag_code",
                            "source_kind",
                            "log_index",
                            "interaction_index",
                            "quote",
                            "rationale",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "audit": {
                    "type": "object",
                    "properties": {
                        "ambiguous": {"type": "array", "items": {"type": "string"}},
                        "unsupported": {"type": "array", "items": {"type": "string"}},
                        "new_tag_ideas": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ambiguous", "unsupported", "new_tag_ideas"],
                    "additionalProperties": False,
                },
            },
            "required": ["rollout_id", "events", "audit"],
            "additionalProperties": False,
        },
    }


def build_messages(
    manifest: dict[str, Any],
    view: dict[str, Any],
    codebook: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a meticulous qualitative research adjudicator. Apply the supplied "
        "operational codebook semantically, never by keyword matching. Return only "
        "the requested JSON object."
    )
    user = f"""
Label the target-authored strategic behavior in one N=2 test-time-compute
negotiation rollout.

Rules:
1. Read the entire rollout. Public messages by the baseline agent are context
   only; emit events only for the target agent `{manifest["target_agent"]}`,
   which is Claude Sonnet 4.6.
2. For public discussion evidence, cite `conversation_logs` and its exact
   `log_index`. For private thinking, proposal, voting, or reflection evidence,
   cite `target_private_interactions` and its exact `interaction_index`.
   Do not label a public discussion response a second time from the interaction
   stream.
3. Output one event per positive tag per source record. A source may receive
   multiple different tags. Do not duplicate the same tag on the same source.
4. Use high precision with high recall. Reject negated, hypothetical, quoted,
   merely mentioned, or setup-inherited concepts unless the target actually
   performs the behavior.
5. `quote` must be a short exact substring of the cited `content` or `response`.
6. Use `formal_outcome` only for a structurally defined codebook tag supported
   by the saved outcome. For it, both indices must be null.
7. Respect every tag's game and minimum-agent scope. Coalition tags requiring
   three or more agents cannot apply in this N=2 experiment.
8. If no tag applies, return an empty `events` array. Still return the audit.

Rollout ID: {manifest["rollout_id"]}

Codebook:
{json.dumps(codebook, ensure_ascii=False)}

Rollout:
{json.dumps(view, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def source_maps(view: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    logs = {int(row["log_index"]): row for row in view["conversation_logs"]}
    interactions = {
        int(row["interaction_index"]): row
        for row in view["target_private_interactions"]
    }
    return logs, interactions


def validate_and_expand(
    response: dict[str, Any],
    manifest: dict[str, Any],
    view: dict[str, Any],
    codebook_by_code: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if response.get("rollout_id") != manifest["rollout_id"]:
        raise ValueError(
            f"rollout_id mismatch: {response.get('rollout_id')!r} "
            f"!= {manifest['rollout_id']!r}"
        )
    compact_events = response.get("events")
    audit = response.get("audit")
    if not isinstance(compact_events, list):
        raise ValueError("events must be a list")
    if not isinstance(audit, dict):
        raise ValueError("audit must be an object")
    for field in ("ambiguous", "unsupported", "new_tag_ideas"):
        if not isinstance(audit.get(field), list) or not all(
            isinstance(value, str) for value in audit[field]
        ):
            raise ValueError(f"audit.{field} must be a list of strings")

    logs, interactions = source_maps(view)
    seen: set[tuple[str, str, int | None]] = set()
    expanded: list[dict[str, Any]] = []
    for position, compact in enumerate(compact_events):
        if not isinstance(compact, dict):
            raise ValueError(f"events[{position}] must be an object")
        tag_code = compact.get("tag_code")
        if tag_code not in codebook_by_code:
            raise ValueError(f"events[{position}] has unknown tag_code {tag_code!r}")
        source_kind = compact.get("source_kind")
        if source_kind not in VALID_SOURCE_KINDS:
            raise ValueError(f"events[{position}] has invalid source_kind")
        confidence = compact.get("confidence")
        if confidence not in VALID_CONFIDENCES:
            raise ValueError(f"events[{position}] has invalid confidence")
        quote = compact.get("quote")
        rationale = compact.get("rationale")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"events[{position}] quote must be non-empty")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"events[{position}] rationale must be non-empty")

        log_index = compact.get("log_index")
        interaction_index = compact.get("interaction_index")
        if log_index is not None and not isinstance(log_index, int):
            raise ValueError(f"events[{position}] log_index must be integer or null")
        if interaction_index is not None and not isinstance(interaction_index, int):
            raise ValueError(
                f"events[{position}] interaction_index must be integer or null"
            )

        if source_kind == "conversation_log":
            if log_index not in logs or interaction_index is not None:
                raise ValueError(f"events[{position}] has invalid conversation pointer")
            source = logs[log_index]
            if source.get("speaker_agent") != manifest["target_agent"]:
                raise ValueError(
                    f"events[{position}] cites non-target public speaker "
                    f"{source.get('speaker_agent')!r}"
                )
            source_text = str(source.get("content") or "")
            phase = "discussion"
            evidence_type = "utterance"
            round_number = source.get("round")
            discussion_turn = source.get("discussion_turn")
            speaker_order = source.get("speaker_order")
            total_speakers = source.get("total_speakers")
            source_pointer = log_index
        elif source_kind == "interaction":
            if interaction_index not in interactions or log_index is not None:
                raise ValueError(f"events[{position}] has invalid interaction pointer")
            source = interactions[interaction_index]
            source_text = str(source.get("response") or "")
            phase = source.get("phase")
            evidence_type = {
                "private_thinking": "private_thinking",
                "proposal": "proposal_reasoning",
                "voting": "vote_reasoning",
                "reflection": "reflection",
            }.get(phase)
            if evidence_type is None:
                raise ValueError(
                    f"events[{position}] cites unsupported interaction phase {phase!r}"
                )
            round_number = source.get("round")
            discussion_turn = source.get("discussion_turn")
            speaker_order = None
            total_speakers = None
            source_pointer = interaction_index
        else:
            if log_index is not None or interaction_index is not None:
                raise ValueError(f"events[{position}] formal outcome must have null indices")
            source_text = ""
            tag = codebook_by_code[tag_code]
            if not tag.get("scope_hint", {}).get("structural"):
                raise ValueError(
                    f"events[{position}] uses formal_outcome for nonstructural tag "
                    f"{tag_code}"
                )
            phase = "final_outcome"
            evidence_type = "formal_outcome"
            round_number = manifest.get("final_round")
            discussion_turn = None
            speaker_order = None
            total_speakers = None
            source_pointer = None

        if source_kind != "formal_outcome" and quote not in source_text:
            raise ValueError(
                f"events[{position}] quote is not an exact substring of its source"
            )

        scope = codebook_by_code[tag_code].get("scope_hint") or {}
        eligible_games = scope.get("games") or []
        if eligible_games and manifest["game_label"] not in eligible_games:
            raise ValueError(
                f"events[{position}] tag {tag_code} is out of game scope"
            )
        min_agents = scope.get("min_agents")
        if min_agents is not None and int(manifest["n_agents"]) < int(min_agents):
            raise ValueError(
                f"events[{position}] tag {tag_code} requires {min_agents} agents"
            )

        duplicate_key = (tag_code, source_kind, source_pointer)
        if duplicate_key in seen:
            raise ValueError(
                f"events[{position}] duplicates tag/source event {duplicate_key}"
            )
        seen.add(duplicate_key)

        target_agent = manifest["target_agent"]
        agent_models = manifest.get("agent_model_map") or {}
        agent_elos = manifest.get("agent_elo_map") or {}
        tag = codebook_by_code[tag_code]
        expanded.append(
            {
                "rollout_id": manifest["rollout_id"],
                "seed": int(manifest["seed"]),
                "source_config_id": int(manifest["source_config_id"]),
                "result_path": manifest["result_path"],
                "interactions_path": manifest["interactions_path"],
                "rollout_view_path": manifest["rollout_view_path"],
                "family": manifest["family"],
                "level": manifest["level"],
                "level_index": int(manifest["level_index"]),
                "provider": manifest["provider"],
                "game_label": manifest["game_label"],
                "game_cell": manifest["game_cell"],
                "game_type": manifest["game_type"],
                "n_agents": int(manifest["n_agents"]),
                "order": manifest["order"],
                "target_agent": target_agent,
                "baseline_agent": manifest["baseline_agent"],
                "speaker_agent": target_agent,
                "speaker_model": agent_models.get(target_agent),
                "speaker_elo": agent_elos.get(target_agent),
                "speaker_role": "target",
                "speaker_is_target": True,
                "speaker_is_baseline": False,
                "tag_code": tag_code,
                "tag_title": tag["tag_title"],
                "tag_category": tag["category"],
                "evidence_type": evidence_type,
                "source_kind": source_kind,
                "phase": phase,
                "round": round_number,
                "discussion_turn": discussion_turn,
                "log_index": log_index,
                "interaction_index": interaction_index,
                "speaker_order": speaker_order,
                "total_speakers": total_speakers,
                "quote": quote,
                "rationale": rationale,
                "confidence": confidence,
                "negation_checked": True,
            }
        )
    return expanded, audit


async def request_direct(
    *,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/joie-zhang/bargain",
        "X-Title": "TTC qualitative adjudication",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"OpenRouter HTTP {response.status}: {body[:1000]}")
            data = json.loads(body)
    if data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response contains no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter response contains no message content")
    return content, data.get("usage") or {}


async def request_proxy(
    *,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    proxy_dir: Path,
    endpoint_url: str = OPENROUTER_URL,
) -> tuple[str, dict[str, Any]]:
    processed_dir = proxy_dir / "processed"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    identifier = f"ttcqual_{time.time_ns()}_{uuid.uuid4().hex}"
    request_path = proxy_dir / f"request_{identifier}.json"
    response_path = proxy_dir / f"response_{identifier}.json"
    request_payload = {
        "url": endpoint_url,
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/joie-zhang/bargain",
            "X-Title": "TTC qualitative adjudication",
        },
        "payload": payload,
        "timeout": timeout,
    }
    atomic_write_text(request_path, json.dumps(request_payload))
    start = time.monotonic()
    while not response_path.exists():
        if time.monotonic() - start > timeout + 300:
            if request_path.exists():
                request_path.unlink()
            raise TimeoutError(f"No proxy response for {identifier}")
        await asyncio.sleep(0.2)
    while True:
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            break
        except json.JSONDecodeError:
            await asyncio.sleep(0.1)
    destination = processed_dir / response_path.name
    if destination.exists():
        destination.unlink()
    shutil.move(str(response_path), str(destination))
    if response.get("error"):
        raise RuntimeError(f"OpenRouter proxy error: {response['error']}")
    content = response.get("result")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter proxy response contains no result text")
    return content, response.get("usage") or {}


def acquire_lock(rollout_id: str, stale_seconds: int) -> Path | None:
    lock_path = OUTPUT_ROOT / "locks" / f"{rollout_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age > stale_seconds:
            lock_path.unlink()
        else:
            return None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at_unix": time.time(),
            },
            handle,
        )
    return lock_path


def completion_path(rollout_id: str) -> Path:
    return OUTPUT_ROOT / "completions" / f"{rollout_id}.json"


async def judge_one(
    manifest: dict[str, Any],
    *,
    codebook: list[dict[str, Any]],
    codebook_by_code: dict[str, dict[str, Any]],
    model: str,
    reasoning_effort: str,
    api_key: str,
    transport: str,
    proxy_dir: Path,
    timeout: float,
    max_tokens: int,
    max_attempts: int,
    stale_lock_seconds: int,
) -> str:
    rollout_id = manifest["rollout_id"]
    complete_path = completion_path(rollout_id)
    if complete_path.exists():
        return "already_complete"
    lock_path = acquire_lock(rollout_id, stale_lock_seconds)
    if lock_path is None:
        return "locked"
    try:
        if complete_path.exists():
            return "already_complete"
        view = json.loads(Path(manifest["rollout_view_path"]).read_text(encoding="utf-8"))
        messages = build_messages(manifest, view, compact_codebook(codebook))
        schema = output_schema(list(codebook_by_code))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            payload: dict[str, Any] = {
                "model": (
                    model.removeprefix("openai/")
                    if transport == "openai_proxy"
                    else model
                ),
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            }
            if transport == "openai_proxy":
                payload["max_completion_tokens"] = max_tokens
                payload["reasoning_effort"] = reasoning_effort
            else:
                payload["max_tokens"] = max_tokens
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "exclude": True,
                }
            started = time.time()
            try:
                if transport in {"proxy", "openai_proxy"}:
                    raw_text, usage = await request_proxy(
                        payload=payload,
                        api_key=api_key,
                        timeout=timeout,
                        proxy_dir=proxy_dir,
                        endpoint_url=(
                            OPENAI_URL
                            if transport == "openai_proxy"
                            else OPENROUTER_URL
                        ),
                    )
                else:
                    raw_text, usage = await request_direct(
                        payload=payload,
                        api_key=api_key,
                        timeout=timeout,
                    )
                parsed = parse_model_json(raw_text)
                events, audit = validate_and_expand(
                    parsed, manifest, view, codebook_by_code
                )
                write_json(
                    OUTPUT_ROOT / "raw_responses" / f"{rollout_id}.json",
                    {
                        "rollout_id": rollout_id,
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "transport": transport,
                        "attempt": attempt,
                        "response_text": raw_text,
                        "usage": usage,
                    },
                )
                write_jsonl(
                    OUTPUT_ROOT / "events" / f"{rollout_id}.jsonl",
                    events,
                )
                write_json(
                    OUTPUT_ROOT / "audits" / f"{rollout_id}.json",
                    audit,
                )
                write_json(
                    complete_path,
                    {
                        "rollout_id": rollout_id,
                        "seed": manifest["seed"],
                        "source_config_id": manifest["source_config_id"],
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "transport": transport,
                        "attempt": attempt,
                        "event_count": len(events),
                        "usage": usage,
                        "elapsed_seconds": time.time() - started,
                        "result_sha256": manifest["result_sha256"],
                        "interactions_sha256": manifest["interactions_sha256"],
                    },
                )
                return "completed"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                write_json(
                    OUTPUT_ROOT / "failures" / f"{rollout_id}_attempt{attempt}.json",
                    {
                        "rollout_id": rollout_id,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )
                if attempt < max_attempts:
                    await asyncio.sleep(min(60.0, (2 ** (attempt - 1)) + random.random()))
        raise RuntimeError(
            f"{rollout_id} failed after {max_attempts} attempts: {last_error}"
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()


async def async_main(args: argparse.Namespace) -> int:
    manifests = read_jsonl(MANIFEST_PATH)
    if args.seed is not None:
        manifests = [row for row in manifests if int(row["seed"]) == args.seed]
    if args.rollout_id:
        manifests = [row for row in manifests if row["rollout_id"] == args.rollout_id]
    manifests.sort(key=lambda row: (int(row["seed"]), int(row["source_config_id"])))
    if args.limit is not None:
        manifests = manifests[: args.limit]
    if not manifests:
        raise RuntimeError("No prepared rollouts match the requested filters")

    codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    codebook_by_code = {row["tag_code"]: row for row in codebook}
    api_key_env = (
        "JOIE_OPENAI_API_KEY_1"
        if args.transport == "openai_proxy"
        else "JOIE_OPENROUTER_API_KEY_1"
    )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    semaphore = asyncio.Semaphore(args.concurrency)
    counts = {
        "completed": 0,
        "already_complete": 0,
        "locked": 0,
        "failed": 0,
    }

    async def run(manifest: dict[str, Any]) -> None:
        async with semaphore:
            rollout_id = manifest["rollout_id"]
            try:
                status = await judge_one(
                    manifest,
                    codebook=codebook,
                    codebook_by_code=codebook_by_code,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    api_key=api_key,
                    transport=args.transport,
                    proxy_dir=args.proxy_dir,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    max_attempts=args.max_attempts,
                    stale_lock_seconds=args.stale_lock_seconds,
                )
                counts[status] += 1
                print(f"{status}: {rollout_id}", flush=True)
            except Exception as exc:
                counts["failed"] += 1
                print(
                    f"failed: {rollout_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    await asyncio.gather(*(run(manifest) for manifest in manifests))
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rollout-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="xhigh",
    )
    parser.add_argument(
        "--transport",
        choices=["direct", "proxy", "openai_proxy"],
        default="proxy",
    )
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65_536,
        help="Output-token ceiling, including hidden reasoning tokens.",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--stale-lock-seconds", type=int, default=21_600)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(async_main(parse_args())))


if __name__ == "__main__":
    main()
