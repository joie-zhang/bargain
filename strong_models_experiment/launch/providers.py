"""Explicit credentials and a single-attempt, audited HTTP boundary."""

import importlib.util
import os
from pathlib import Path
import re
import time
import uuid

from .plan import KEYS
from .schema import strict_json
from .store import atomic_json


class ProviderFailure(RuntimeError):
    pass


class UnknownRequestOutcome(ProviderFailure):
    pass


def load_credentials(env_file=None, environ=None):
    source = os.environ if environ is None else environ
    if env_file is not None:
        path = Path(env_file).expanduser().resolve()
        if path.stat().st_mode & 0o077:
            raise ValueError(f"Credential file must be private; run chmod 600 {path}")
        source = {}
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or key not in KEYS.values() or key in source:
                raise ValueError(f"Invalid, duplicate, or unsupported credential entry on line {number}")
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"Unclosed credential quote on line {number}")
                value = value[1:-1]
            source[key] = value
    # An explicit file is the sole source; never fill its gaps from the shell.
    result = {name: source[name] for name in KEYS.values() if source.get(name)}
    for name, value in result.items():
        if not isinstance(value, str) or any(c.isspace() for c in value) or value.startswith("your_"):
            raise ValueError(f"Invalid value for {name}")
    return result


def required_credentials(plan):
    return sorted({s["credential_env"] for run in plan["runs"] for s in run["seats"]})


def doctor(plan, credentials):
    missing = [name for name in required_credentials(plan) if not credentials.get(name)]
    dependencies = [name for name in ("numpy", "scipy", "aiohttp") if importlib.util.find_spec(name) is None]
    return {"ready_offline": not missing and not dependencies and os.name == "posix", "missing_keys": missing,
            "missing_dependencies": dependencies, "requested_runs": len(plan["runs"]),
            "supported_platform": os.name == "posix",
            "authentication_tested": False, "network_tested": False,
            "transport": "direct", "worker_count": 1}


def redact(value, credentials):
    if isinstance(value, str):
        for secret in credentials.values():
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
    if isinstance(value, dict):
        return {k: redact(v, credentials) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, credentials) for v in value]
    return value


class RequestJournal:
    def __init__(self, attempt, credentials, timeout):
        self.attempt = Path(attempt)
        self.directory = self.attempt / "requests"
        self.directory.mkdir(mode=0o700)
        self.credentials = credentials
        self.timeout = timeout
        self.failure = None
        self.session = None
        self.game_state = None
        self.game_environment = None

    def stop(self, error):
        if self.failure is None:
            self.failure = error

    async def close(self):
        if self.session is not None:
            await self.session.close()

    async def post(self, seat, payload, context_record):
        import aiohttp
        import asyncio

        if self.failure is not None:
            raise self.failure
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout), trust_env=False)
        provider = seat["provider"]
        key = self.credentials[seat["credential_env"]]
        headers = {"Content-Type": "application/json"}
        if provider == "anthropic":
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        else:
            headers["Authorization"] = "Bearer " + key
        path = self.directory / (uuid.uuid4().hex + ".json")
        record = {"state": "dispatching", "started_at": time.time(), "seat": seat,
                  "context": context_record, "payload": payload, "transport": "direct"}
        # A crash after this durable record is treated as an unknown paid outcome.
        atomic_json(path, redact(record, self.credentials))
        try:
            async with self.session.post(seat["endpoint"], headers=headers, json=payload, allow_redirects=False) as response:
                raw = await response.text()
                record.update({"http_status": response.status, "response_text": raw,
                               "provider_request_id": response.headers.get("x-request-id") or response.headers.get("request-id"),
                               "finished_at": time.time(), "state": "response_received"})
                atomic_json(path, redact(record, self.credentials))
                if response.status >= 500 or response.status in (408, 409):
                    raise UnknownRequestOutcome(f"{provider} HTTP {response.status}; remote execution status is unknown")
                if response.status != 200:
                    raise ProviderFailure(f"{provider} HTTP {response.status}; see request record {path}")
                data = strict_json(raw)
                if not isinstance(data, dict) or "error" in data:
                    raise ProviderFailure(f"{provider} returned an invalid response; see {path}")
                return data, str(path.relative_to(self.attempt))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = UnknownRequestOutcome(f"{provider} {type(exc).__name__}; remote execution status is unknown")
            self.stop(error)
            record.update(state="unknown_request_outcome", error=str(error))
            atomic_json(path, redact(record, self.credentials))
            raise error from None
        except Exception as exc:
            # No retry or alternate provider. The worker records the failed attempt.
            self.stop(exc)
            record.update(state="unknown_request_outcome" if isinstance(exc, UnknownRequestOutcome) else "failed", error=redact(str(exc), self.credentials))
            atomic_json(path, redact(record, self.credentials))
            raise


def build_payload(seat, messages, max_tokens):
    payload = {"model": seat["model_id"], **seat["parameters"]}
    if seat["provider"] == "anthropic":
        payload.update(system="\n".join(m["content"] for m in messages if m["role"] == "system"),
                       messages=[m for m in messages if m["role"] != "system"], max_tokens=max_tokens)
    else:
        payload["messages"] = messages
        field = "max_completion_tokens" if seat["provider"] == "openai" and "reasoning_effort" in payload else "max_tokens"
        payload[field] = max_tokens
    if seat["temperature"] is not None:
        payload["temperature"] = seat["temperature"]
    if seat["provider"] == "openai":
        payload["store"] = False
    return payload


def parse_response(provider, data):
    if provider == "anthropic":
        blocks = data.get("content", [])
        content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        finish = data.get("stop_reason")
        allowed = {"end_turn"}
    else:
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderFailure("Expected exactly one completion")
        content = choices[0].get("message", {}).get("content")
        finish = choices[0].get("finish_reason")
        allowed = {"stop"}
    # Reasoning remains in the raw response record; it is never used as an answer.
    if not isinstance(content, str) or not content.strip() or finish not in allowed:
        raise ProviderFailure(f"Missing final answer or incomplete generation (finish_reason={finish!r})")
    return content, {"finish_reason": finish, "usage": data.get("usage"),
                     "provider_reported_model": data.get("model"), "response_id": data.get("id")}
