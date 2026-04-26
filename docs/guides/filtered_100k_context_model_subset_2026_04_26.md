# Filtered >=100K Context Model Subset

Source universe: `docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`.

Purpose: define the model pool for random multi-agent negotiation experiments where context pressure may become large for `n = 10`.

## Inclusion Rule

- Start from the active 30-model smooth Elo coverage list.
- Retain models with Arena Elo >= 1240.
- For OpenRouter-backed routes, require realized OpenRouter context >= 100,000 tokens.
- For native OpenAI, Anthropic, and Google routes, retain models whose source table records native/full context >= 100K.
- `amazon-nova-micro-v1.0` is the weakest retained model, with Arena Elo 1240 and OpenRouter context 128,000. No retained model has Elo below 1240.

## Retained Model Pool

| Model | Arena Elo | Route | Context Used For Filter |
|---|---:|---|---:|
| `claude-opus-4-6-thinking` | 1504 | Anthropic | 1M |
| `claude-opus-4-6` | 1499 | Anthropic | 1M |
| `gemini-3-pro` | 1486 | Google | 1M |
| `gpt-5.4-high` | 1484 | OpenAI | 1.1M |
| `gpt-5.2-chat-latest-20260210` | 1478 | OpenAI | 128K |
| `claude-opus-4-5-20251101-thinking-32k` | 1474 | Anthropic | 200K |
| `claude-opus-4-5-20251101` | 1468 | Anthropic | 200K |
| `gemini-2.5-pro` | 1448 | Google | 1M |
| `qwen3-max-preview` | 1435 | OpenRouter | 262,144 |
| `deepseek-r1-0528` | 1422 | OpenRouter | 163,840 |
| `claude-haiku-4-5-20251001` | 1407 | Anthropic | 200K |
| `claude-sonnet-4-20250514` | 1389 | Anthropic | 1M |
| `gemma-3-27b-it` | 1365 | OpenRouter | 131,072 |
| `o3-mini-high` | 1363 | OpenAI | 200K |
| `deepseek-v3` | 1358 | OpenRouter | 163,840 |
| `gpt-4o-2024-05-13` | 1345 | OpenAI | 128K |
| `gpt-5-nano-high` | 1337 | OpenAI | 128K |
| `qwq-32b` | 1336 | OpenRouter | 131,072 |
| `gpt-4.1-nano-2025-04-14` | 1322 | OpenAI | 1M |
| `llama-3.3-70b-instruct` | 1318 | OpenRouter | 131,072 |
| `gpt-4o-mini-2024-07-18` | 1317 | OpenAI | 128K |
| `amazon-nova-pro-v1.0` | 1290 | OpenRouter | 300,000 |
| `command-r-plus-08-2024` | 1276 | OpenRouter | 128,000 |
| `claude-3-haiku-20240307` | 1260 | Anthropic | 200K |
| `amazon-nova-micro-v1.0` | 1240 | OpenRouter | 128,000 |

Retained count: 25.

## Excluded From The 30-Model List

| Model | Arena Elo | Reason |
|---|---:|---|
| `deepseek-r1` | 1398 | OpenRouter context is 64,000, below the 100K threshold. |
| `qwen2.5-72b-instruct` | 1302 | OpenRouter context is 32,768, below the 100K threshold. |
| `llama-3.1-8b-instruct` | 1211 | OpenRouter context is 16,384, below the 100K threshold. |
| `llama-3.2-3b-instruct` | 1166 | OpenRouter context is 80,000 and the model has coherence concerns in small-`n` settings. |
| `llama-3.2-1b-instruct` | 1110 | OpenRouter context is 60,000 and the model has coherence concerns in small-`n` settings. |

## Sampling Sanity Check

The first Elo-spread sanity check uses this retained 25-model pool, samples distinct models without replacement, and evaluates `n in {2, 4, 6, 8, 10}` for `k = 1000` random draws per `n`.

The samples CSV records both Elo variance and Elo standard deviation. The histogram plot uses Elo standard deviation on the x-axis because it remains in ordinary Elo-point units.

Generated artifacts live under `analysis/elo_variance_sampling_100k_context/`.
