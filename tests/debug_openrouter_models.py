#!/usr/bin/env python3
"""
=============================================================================
Debug OpenRouter Model Responses
=============================================================================

Tests specific OpenRouter models to inspect their raw response format.
Useful for diagnosing empty response issues.

Usage:
    python tests/debug_openrouter_models.py

    # Test specific models
    python tests/debug_openrouter_models.py --models "z-ai/glm-4.7" "deepseek/deepseek-r1-0528"

=============================================================================
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path

# Models to test
PROBLEM_MODELS = [
    ("GLM-4.7", "z-ai/glm-4.7"),
    ("DeepSeek-R1-0528", "deepseek/deepseek-r1-0528"),
    ("DeepSeek-R1", "deepseek/deepseek-r1"),
    ("DeepSeek-V3", "deepseek/deepseek-chat"),
    ("Amazon-Nova-Micro", "amazon/nova-micro-v1"),  # Known working - for comparison
]

TEST_PROMPT = "Say 'Hello, I am working!' and nothing else."


async def test_model_direct(model_id: str, api_key: str) -> dict:
    """Test a model directly via OpenRouter API and return raw response."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/negotiation-research",
        "X-Title": "Negotiation Research Debug"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": TEST_PROMPT}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                status = response.status
                raw_text = await response.text()

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    data = {"raw_text": raw_text, "parse_error": True}

                return {
                    "status": status,
                    "data": data,
                    "error": None
                }
        except Exception as e:
            return {
                "status": None,
                "data": None,
                "error": f"{type(e).__name__}: {e}"
            }


def extract_content(data: dict) -> str:
    """Try to extract content from various response formats."""
    if not data or "choices" not in data:
        return None

    choices = data.get("choices", [])
    if not choices:
        return None

    choice = choices[0]
    message = choice.get("message", {})

    # Standard content field
    content = message.get("content")
    if content:
        return content

    # Some reasoning models put content elsewhere
    # Check for reasoning_content (DeepSeek-R1 style)
    reasoning = message.get("reasoning_content")
    if reasoning:
        return f"[REASONING]: {reasoning}"

    # Check for function_call or tool_calls
    if "function_call" in message:
        return f"[FUNCTION_CALL]: {message['function_call']}"

    if "tool_calls" in message:
        return f"[TOOL_CALLS]: {message['tool_calls']}"

    # Check delta for streaming
    delta = choice.get("delta", {})
    if delta.get("content"):
        return delta["content"]

    return None


def print_response_structure(data: dict, indent: int = 0):
    """Print the structure of the response for debugging."""
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print_response_structure(value, indent + 1)
            else:
                # Truncate long values
                str_val = str(value)
                if len(str_val) > 100:
                    str_val = str_val[:100] + "..."
                print(f"{prefix}{key}: {str_val}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{prefix}[{i}]:")
            print_response_structure(item, indent + 1)
    else:
        print(f"{prefix}{data}")


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("OpenRouter Model Response Debug")
    print("=" * 70)
    print(f"\nTest prompt: {TEST_PROMPT}\n")

    for name, model_id in PROBLEM_MODELS:
        print("-" * 70)
        print(f"Testing: {name}")
        print(f"Model ID: {model_id}")
        print("-" * 70)

        result = await test_model_direct(model_id, api_key)

        if result["error"]:
            print(f"ERROR: {result['error']}")
            continue

        print(f"HTTP Status: {result['status']}")

        data = result["data"]

        # Check for API errors
        if "error" in data:
            print(f"API Error: {data['error']}")
            continue

        # Print response structure
        print("\nResponse Structure:")
        print_response_structure(data)

        # Try to extract content
        content = extract_content(data)
        print(f"\nExtracted Content: {content}")

        if not content:
            print("WARNING: Could not extract content from response!")
            print("\nFull response JSON:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

        print()

    print("=" * 70)
    print("Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
