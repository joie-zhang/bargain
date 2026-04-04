#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import requests


DEFAULT_MODELS = [
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-mini-128k-instruct",
    "microsoft/phi-3-small-128k-instruct:free",
    "microsoft/phi-3-small-128k-instruct",
    "microsoft/phi-3-medium-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct",
    "microsoft/phi-3.5-mini-128k-instruct:free",
    "microsoft/phi-3.5-mini-128k-instruct",
]

DEFAULT_PROMPT = "What is the color of the sky?"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _extract_error(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:500]

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if isinstance(error, str) and error.strip():
        return error.strip()
    return json.dumps(data)[:500]


def _extract_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    return ""


def probe_model(api_key: str, model: str, prompt: str, timeout: float) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/negotiation-research",
        "X-OpenRouter-Title": "Negotiation Research",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    started = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return {
            "model": model,
            "ok": False,
            "latency_s": round(time.time() - started, 2),
            "error": str(exc),
        }

    latency_s = round(time.time() - started, 2)

    if response.status_code != 200:
        return {
            "model": model,
            "ok": False,
            "status_code": response.status_code,
            "latency_s": latency_s,
            "error": _extract_error(response),
        }

    data = response.json()
    content = _extract_content(data)
    return {
        "model": model,
        "ok": bool(content),
        "status_code": response.status_code,
        "latency_s": latency_s,
        "content": content,
        "usage": data.get("usage") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Phi-3 128k models on OpenRouter.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    models = args.models or DEFAULT_MODELS
    any_success = False

    for model in models:
        result = probe_model(api_key=api_key, model=model, prompt=args.prompt, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=True))
        any_success = any_success or result.get("ok", False)

    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
