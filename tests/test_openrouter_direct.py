#!/usr/bin/env python3
"""Quick test of OpenRouter API with aiohttp."""
import os
import asyncio
import aiohttp

async def test():
    key = os.getenv("OPENROUTER_API_KEY")
    print(f"Using key: {key[:15]}...{key[-4:]}")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/negotiation-research",
        "X-Title": "Negotiation Research"
    }

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "Say hi"}],
        "temperature": 0.7,
        "max_tokens": 10
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            print(f"Status: {response.status}")
            text = await response.text()
            print(f"Response: {text[:500]}")

if __name__ == "__main__":
    asyncio.run(test())
