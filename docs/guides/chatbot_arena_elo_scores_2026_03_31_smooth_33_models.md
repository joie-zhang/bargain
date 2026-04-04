# Smooth Elo Coverage Model Set with Arena Context and Repo Support

Generated from [chatbot_arena_elo_scores_2026_03_31.md](/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31.md) for the active 32-model smooth-coverage list you are using. Relative to the prior smooth-coverage set, this version removes `mixtral-8x22b-instruct-v0.1` and swaps the two 8.2K-context slots to `amazon-nova-pro-v1.0` and `command-r-plus-08-2024` so the active roster stays on the longer-context routes you selected.

Support status reflects the current repo registry in [configs.py](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py) after the alias patch in this turn.

## 32-Model Set

| # | Model | Arena Elo | Arena Context | Arena Org | Repo Route | Model ID / Route | Key Needed | Support | Notes |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | `claude-opus-4-6-thinking` | 1504 | 1M | Anthropic | Anthropic | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | yes (inferred mapping) | Anthropic adaptive thinking alias; Arena does not explicitly document the API parameterization. |
| 2 | `claude-opus-4-6` | 1499 | 1M | Anthropic | Anthropic | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | yes |  |
| 3 | `gemini-3-pro` | 1486 | 1M | Google | Google | `gemini-3-pro-preview` | `GOOGLE_API_KEY` | yes |  |
| 4 | `gpt-5.4-high` | 1484 | 1.1M | OpenAI | OpenAI | `gpt-5.4` | `OPENAI_API_KEY` | yes |  |
| 5 | `gpt-5.2-chat-latest-20260210` | 1478 | 128K | OpenAI | OpenAI | `gpt-5.2-chat-latest` | `OPENAI_API_KEY` | yes |  |
| 6 | `claude-opus-4-5-20251101-thinking-32k` | 1474 | 200K | Anthropic | Anthropic | `claude-opus-4-5-20251101` | `ANTHROPIC_API_KEY` | yes |  |
| 7 | `claude-opus-4-5-20251101` | 1468 | 200K | Anthropic | Anthropic | `claude-opus-4-5-20251101` | `ANTHROPIC_API_KEY` | yes |  |
| 8 | `gemini-2.5-pro` | 1448 | 1M | Google | Google | `gemini-2.5-pro` | `GOOGLE_API_KEY` | yes |  |
| 9 | `qwen3-max-preview` | 1435 | 262.1K | Alibaba | OpenRouter | `qwen/qwen3-max` | `OPENROUTER_API_KEY` | yes (alias) | Routed to current OpenRouter `qwen/qwen3-max` model. |
| 10 | `deepseek-r1-0528` | 1422 | 163.8K | DeepSeek | OpenRouter | `deepseek/deepseek-r1-0528` | `OPENROUTER_API_KEY` | yes |  |
| 11 | `claude-haiku-4-5-20251001` | 1407 | 200K | Anthropic | Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | yes |  |
| 12 | `deepseek-r1` | 1398 | 128K | DeepSeek | OpenRouter | `deepseek/deepseek-r1` | `OPENROUTER_API_KEY` | yes |  |
| 13 | `claude-sonnet-4-20250514` | 1389 | 1M | Anthropic | Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | yes |  |
| 14 | `claude-3-5-sonnet-20241022` | 1372 | 200K | Anthropic | OpenRouter | `anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` | yes (best-effort) | Exact Anthropic dated snapshot is retired; routed to OpenRouter `anthropic/claude-3.5-sonnet`. |
| 15 | `gemma-3-27b-it` | 1365 | 131.1K | Google | OpenRouter | `google/gemma-3-27b-it` | `OPENROUTER_API_KEY` | yes |  |
| 16 | `o3-mini-high` | 1363 | 200K | OpenAI | OpenAI | `o3-mini-2025-01-31` | `OPENAI_API_KEY` | yes |  |
| 17 | `deepseek-v3` | 1358 | 128K | DeepSeek | OpenRouter | `deepseek/deepseek-chat` | `OPENROUTER_API_KEY` | yes |  |
| 18 | `gpt-4o-2024-05-13` | 1345 | 128K | OpenAI | OpenAI | `gpt-4o-2024-05-13` | `OPENAI_API_KEY` | yes |  |
| 19 | `qwq-32b` | 1336 | 131.1K | Alibaba | OpenRouter | `qwen/qwq-32b` | `OPENROUTER_API_KEY` | yes |  |
| 20 | `gpt-4.1-nano-2025-04-14` | 1322 | 1M | OpenAI | OpenAI | `gpt-4.1-nano-2025-04-14` | `OPENAI_API_KEY` | yes |  |
| 21 | `llama-3.3-70b-instruct` | 1318 | 128K | Meta | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 22 | `gpt-4o-mini-2024-07-18` | 1317 | 128K | OpenAI | OpenAI | `gpt-4o-mini-2024-07-18` | `OPENAI_API_KEY` | yes |  |
| 23 | `qwen2.5-72b-instruct` | 1302 | 131K | Alibaba | OpenRouter | `qwen/qwen-2.5-72b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 24 | `amazon-nova-pro-v1.0` | 1290 | 300K | Amazon | OpenRouter | `amazon/nova-pro-v1` | `OPENROUTER_API_KEY` | yes | Replaces the old 8.2K `gemma-2-27b-it` slot. |
| 25 | `command-r-plus-08-2024` | 1276 | 128K | Cohere | OpenRouter | `cohere/command-r-plus-08-2024` | `OPENROUTER_API_KEY` | yes | Replaces the old 8.2K `llama-3-70b-instruct` slot. |
| 26 | `claude-3-haiku-20240307` | 1260 | 200K | Anthropic | Anthropic | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` | yes |  |
| 27 | `amazon-nova-micro-v1.0` | 1240 | 128K | Amazon | OpenRouter | `amazon/nova-micro-v1` | `OPENROUTER_API_KEY` | yes |  |
| 28 | `llama-3.1-8b-instruct` | 1211 | 128K | Meta | OpenRouter | `meta-llama/llama-3.1-8b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 29 | `llama-3.2-3b-instruct` | 1166 | 128K | Meta | OpenRouter | `meta-llama/llama-3.2-3b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 30 | `qwq-32b-preview` | 1156 | 131.1K | Alibaba | OpenRouter | `qwen/qwq-32b` | `OPENROUTER_API_KEY` | yes (best-effort alias) | OpenRouter does not currently expose a separate `qwq-32b-preview` model ID; this is routed to the currently available `qwen/qwq-32b`. |
| 31 | `phi-3-mini-128k-instruct` | 1128 | 128K | Microsoft | Princeton Cluster | `/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct` | GPU job | yes | Uses local Princeton-cluster weights; run on a dedicated GPU allocation. |
| 32 | `llama-3.2-1b-instruct` | 1110 | 128K | Meta | OpenRouter | `meta-llama/llama-3.2-1b-instruct` | `OPENROUTER_API_KEY` | yes |  |

## Special Setup

| Model | Why It Needs Special Setup | Best Current Path | Keys / Setup |
|---|---|---|---|
| `phi-3-mini-128k-instruct` | Supported via local Princeton-cluster weights instead of a hosted API route. | Use the local path in `strong_models_experiment/configs.py` and submit as a GPU job | `/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct` |

## Important Interpretation Notes

- `claude-opus-4-6-thinking` is modeled here as `claude-opus-4-6` plus Anthropic adaptive thinking. That is a strong inference from Arena naming plus Anthropic docs, not an Arena-certified API recipe.
- `claude-3-5-sonnet-20241022` is only available here as a best-effort OpenRouter alias because the exact dated Anthropic snapshot is retired.
- `qwq-32b-preview` is only available here as a best-effort alias to the currently available `qwen/qwq-32b` OpenRouter route.
- `phi-3-mini-128k-instruct` is supported here via local Princeton-cluster weights, not via OpenRouter.
- The active smooth-coverage roster now uses `amazon-nova-pro-v1.0` and `command-r-plus-08-2024` in place of the older 8.2K-context `gemma-2-27b-it` and `llama-3-70b-instruct` slots.
- `mixtral-8x22b-instruct-v0.1` remains excluded because its 65.5K context window still falls below the >100K context target for this roster.
- In this repo, `gpt-5.2-high` is pinned to `gpt-5.2-2025-12-11` with `reasoning_effort="high"`; the Arena label itself does not explicitly say `2025-12-11`.
