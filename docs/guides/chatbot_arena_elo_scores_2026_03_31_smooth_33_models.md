# Smooth Elo Coverage Model Set with Arena Context and Repo Support

Generated from [chatbot_arena_elo_scores_2026_03_31.md](/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31.md) for the 33-model smooth-coverage list you are using (the original 30-model set plus `gemini-3-pro`, `gpt-5.2-chat-latest-20260210`, and `gpt-5.4-high`).

Support status reflects the current repo registry in [configs.py](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py) after the alias patch in this turn.

## 33-Model Set

| # | Model | Arena Elo | Arena Context | Arena Org | Repo Route | Model ID / Route | Key Needed | Support | Notes |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | `claude-opus-4-6-thinking` | 1504 | 1M | Anthropic | Anthropic | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | yes (inferred mapping) | Anthropic adaptive thinking alias; Arena does not explicitly document the API parameterization. |
| 2 | `claude-opus-4-6` | 1499 | 1M | Anthropic | Anthropic | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | yes |  |
| 3 | `claude-opus-4-5-20251101-thinking-32k` | 1474 | 200K | Anthropic | Anthropic | `claude-opus-4-5-20251101` | `ANTHROPIC_API_KEY` | yes |  |
| 4 | `claude-opus-4-5-20251101` | 1468 | 200K | Anthropic | Anthropic | `claude-opus-4-5-20251101` | `ANTHROPIC_API_KEY` | yes |  |
| 5 | `gemini-2.5-pro` | 1448 | 1M | Google | Google | `gemini-2.5-pro` | `GOOGLE_API_KEY` | yes |  |
| 6 | `qwen3-max-preview` | 1435 | 262.1K | Alibaba | OpenRouter | `qwen/qwen3-max` | `OPENROUTER_API_KEY` | yes (alias) | Routed to current OpenRouter `qwen/qwen3-max` model. |
| 7 | `deepseek-r1-0528` | 1422 | 163.8K | DeepSeek | OpenRouter | `deepseek/deepseek-r1-0528` | `OPENROUTER_API_KEY` | yes |  |
| 8 | `claude-haiku-4-5-20251001` | 1407 | 200K | Anthropic | Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | yes |  |
| 9 | `deepseek-r1` | 1398 | 128K | DeepSeek | OpenRouter | `deepseek/deepseek-r1` | `OPENROUTER_API_KEY` | yes |  |
| 10 | `claude-sonnet-4-20250514` | 1389 | 1M | Anthropic | Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | yes |  |
| 11 | `claude-3-5-sonnet-20241022` | 1372 | 200K | Anthropic | OpenRouter | `anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` | yes (best-effort) | Exact Anthropic dated snapshot is retired; routed to OpenRouter `anthropic/claude-3.5-sonnet`. |
| 12 | `gemma-3-27b-it` | 1365 | 131.1K | Google | OpenRouter | `google/gemma-3-27b-it` | `OPENROUTER_API_KEY` | yes |  |
| 13 | `o3-mini-high` | 1363 | 200K | OpenAI | OpenAI | `o3-mini-2025-01-31` | `OPENAI_API_KEY` | yes |  |
| 14 | `deepseek-v3` | 1358 | 128K | DeepSeek | OpenRouter | `deepseek/deepseek-chat` | `OPENROUTER_API_KEY` | yes |  |
| 15 | `gpt-4o-2024-05-13` | 1345 | 128K | OpenAI | OpenAI | `gpt-4o-2024-05-13` | `OPENAI_API_KEY` | yes |  |
| 16 | `qwq-32b` | 1336 | 131.1K | Alibaba | OpenRouter | `qwen/qwq-32b` | `OPENROUTER_API_KEY` | yes |  |
| 17 | `gpt-4.1-nano-2025-04-14` | 1322 | 1M | OpenAI | OpenAI | `gpt-4.1-nano-2025-04-14` | `OPENAI_API_KEY` | yes |  |
| 18 | `llama-3.3-70b-instruct` | 1318 | 128K | Meta | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 19 | `gpt-4o-mini-2024-07-18` | 1317 | 128K | OpenAI | OpenAI | `gpt-4o-mini-2024-07-18` | `OPENAI_API_KEY` | yes |  |
| 20 | `qwen2.5-72b-instruct` | 1302 | 131K | Alibaba | OpenRouter | `qwen/qwen-2.5-72b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 21 | `gemma-2-27b-it` | 1288 | 8.2K | Google | OpenRouter | `google/gemma-2-27b-it` | `OPENROUTER_API_KEY` | yes |  |
| 22 | `llama-3-70b-instruct` | 1275 | 8.2K | Meta | OpenRouter | `meta-llama/llama-3-70b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 23 | `claude-3-haiku-20240307` | 1260 | 200K | Anthropic | Anthropic | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` | yes |  |
| 24 | `amazon-nova-micro-v1.0` | 1240 | 128K | Amazon | OpenRouter | `amazon/nova-micro-v1` | `OPENROUTER_API_KEY` | yes |  |
| 25 | `mixtral-8x22b-instruct-v0.1` | 1228 | 65.5K | Mistral | OpenRouter | `mistralai/mixtral-8x22b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 26 | `llama-3.1-8b-instruct` | 1211 | 128K | Meta | OpenRouter | `meta-llama/llama-3.1-8b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 27 | `mixtral-8x7b-instruct-v0.1` | 1196 | 32K | Mistral | OpenRouter | `mistralai/mixtral-8x7b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 28 | `mistral-7b-instruct-v0.2` | 1149 | 32.8K | Mistral | Together AI or self-hosted TGI/vLLM | `—` | `TOGETHER_API_KEY` or none for self-hosted | no | Not wired in repo. OpenRouter does not currently expose the exact v0.2 model ID we checked. |
| 29 | `phi-3-mini-128k-instruct` | 1128 | 128K | Microsoft | Azure AI Foundry / self-hosted | `—` | `AZURE_AI_API_KEY` + endpoint, or none for self-hosted | no | Not wired in repo. OpenRouter does not currently expose the exact model ID we checked. |
| 30 | `llama-3.2-1b-instruct` | 1110 | 128K | Meta | OpenRouter | `meta-llama/llama-3.2-1b-instruct` | `OPENROUTER_API_KEY` | yes |  |
| 31 | `gemini-3-pro` | 1486 | 1M | Google | Google | `gemini-3-pro-preview` | `GOOGLE_API_KEY` | yes |  |
| 32 | `gpt-5.2-chat-latest-20260210` | 1478 | 128K | OpenAI | OpenAI | `gpt-5.2-chat-latest` | `OPENAI_API_KEY` | yes |  |
| 33 | `gpt-5.4-high` | 1484 | 1.1M | OpenAI | OpenAI | `gpt-5.4` | `OPENAI_API_KEY` | yes |  |

## Exact Models Still Not Wired in This Repo

| Model | Why It Is Still Unsupported | Best Current Path | Keys / Setup |
|---|---|---|---|
| `mistral-7b-instruct-v0.2` | Not wired in repo. OpenRouter does not currently expose the exact v0.2 model ID we checked. | Together AI or self-hosted TGI/vLLM | `TOGETHER_API_KEY` or none for self-hosted |
| `phi-3-mini-128k-instruct` | Not wired in repo. OpenRouter does not currently expose the exact model ID we checked. | Azure AI Foundry / self-hosted | `AZURE_AI_API_KEY` + endpoint, or none for self-hosted |

## Important Interpretation Notes

- `claude-opus-4-6-thinking` is modeled here as `claude-opus-4-6` plus Anthropic adaptive thinking. That is a strong inference from Arena naming plus Anthropic docs, not an Arena-certified API recipe.
- `claude-3-5-sonnet-20241022` is only available here as a best-effort OpenRouter alias because the exact dated Anthropic snapshot is retired.
- In this repo, `gpt-5.2-high` is pinned to `gpt-5.2-2025-12-11` with `reasoning_effort="high"`; the Arena label itself does not explicitly say `2025-12-11`.
