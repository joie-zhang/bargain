# OpenRouter Context Window Audit For The Final 32-Model Slate

Generated: 2026-04-06 06:27:56 EDT

This compares the repo's canonical native/full context window for each OpenRouter-backed model in the final 32-model slate against the current context window reported by OpenRouter.

Sources:
- Native/full context reference: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`
- Current OpenRouter model metadata: `https://openrouter.ai/api/v1/models`
- Fallback when a route is missing from the live OpenRouter model index: the public model page at `https://openrouter.ai/<route>`

Summary:
- OpenRouter-backed entries in the 32-model slate: 16
- Same context on OpenRouter: 4
- Shorter context on OpenRouter: 8
- Missing from live OpenRouter models API and resolved from page HTML: 1
- Missing OpenRouter context even after fallback: 0

| Rank | Model | OpenRouter Route | Native / Full Context | OpenRouter Context Now | OpenRouter vs Native | OpenRouter / Native | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | `qwen3-max-preview` | [`qwen/qwen3-max`](https://openrouter.ai/qwen/qwen3-max) | 262.1K (262,100) | 262,144 | +44 | 100.0% | repo alias to current `qwen/qwen3-max` route |
| 10 | `deepseek-r1-0528` | [`deepseek/deepseek-r1-0528`](https://openrouter.ai/deepseek/deepseek-r1-0528) | 163.8K (163,800) | 163,840 | +40 | 100.0% |  |
| 12 | `deepseek-r1` | [`deepseek/deepseek-r1`](https://openrouter.ai/deepseek/deepseek-r1) | 128K (128,000) | 64,000 | -64,000 | 50.0% |  |
| 14 | `claude-3-5-sonnet-20241022` | [`anthropic/claude-3.5-sonnet`](https://openrouter.ai/anthropic/claude-3.5-sonnet) | 200K (200,000) | 200,000 | same | 100.0% | dated Anthropic snapshot is retired; repo uses best-effort OpenRouter route; not present in current OpenRouter models API; context parsed from route page HTML |
| 15 | `gemma-3-27b-it` | [`google/gemma-3-27b-it`](https://openrouter.ai/google/gemma-3-27b-it) | 131.1K (131,100) | 131,072 | -28 | 100.0% |  |
| 17 | `deepseek-v3` | [`deepseek/deepseek-chat`](https://openrouter.ai/deepseek/deepseek-chat) | 128K (128,000) | 163,840 | +35,840 | 128.0% |  |
| 19 | `qwq-32b` | [`qwen/qwq-32b`](https://openrouter.ai/qwen/qwq-32b) | 131.1K (131,100) | 131,072 | -28 | 100.0% |  |
| 21 | `llama-3.3-70b-instruct` | [`meta-llama/llama-3.3-70b-instruct`](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct) | 128K (128,000) | 131,072 | +3,072 | 102.4% |  |
| 23 | `qwen2.5-72b-instruct` | [`qwen/qwen-2.5-72b-instruct`](https://openrouter.ai/qwen/qwen-2.5-72b-instruct) | 131K (131,000) | 32,768 | -98,232 | 25.0% |  |
| 24 | `amazon-nova-pro-v1.0` | [`amazon/nova-pro-v1`](https://openrouter.ai/amazon/nova-pro-v1) | 300K (300,000) | 300,000 | same | 100.0% |  |
| 25 | `command-r-plus-08-2024` | [`cohere/command-r-plus-08-2024`](https://openrouter.ai/cohere/command-r-plus-08-2024) | 128K (128,000) | 128,000 | same | 100.0% |  |
| 27 | `amazon-nova-micro-v1.0` | [`amazon/nova-micro-v1`](https://openrouter.ai/amazon/nova-micro-v1) | 128K (128,000) | 128,000 | same | 100.0% |  |
| 28 | `llama-3.1-8b-instruct` | [`meta-llama/llama-3.1-8b-instruct`](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct) | 128K (128,000) | 16,384 | -111,616 | 12.8% |  |
| 29 | `llama-3.2-3b-instruct` | [`meta-llama/llama-3.2-3b-instruct`](https://openrouter.ai/meta-llama/llama-3.2-3b-instruct) | 128K (128,000) | 80,000 | -48,000 | 62.5% |  |
| 30 | `qwq-32b-preview` | [`qwen/qwq-32b`](https://openrouter.ai/qwen/qwq-32b) | 131.1K (131,100) | 131,072 | -28 | 100.0% | alias of `qwq-32b` |
| 32 | `llama-3.2-1b-instruct` | [`meta-llama/llama-3.2-1b-instruct`](https://openrouter.ai/meta-llama/llama-3.2-1b-instruct) | 128K (128,000) | 60,000 | -68,000 | 46.9% |  |

Canonical slate rows consulted:
- `qwen3-max-preview`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L20`
- `deepseek-r1-0528`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L21`
- `deepseek-r1`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L23`
- `claude-3-5-sonnet-20241022`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L25`
- `gemma-3-27b-it`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L26`
- `deepseek-v3`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L28`
- `qwq-32b`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L30`
- `llama-3.3-70b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L32`
- `qwen2.5-72b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L34`
- `amazon-nova-pro-v1.0`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L35`
- `command-r-plus-08-2024`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L36`
- `amazon-nova-micro-v1.0`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L38`
- `llama-3.1-8b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L39`
- `llama-3.2-3b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L40`
- `qwq-32b-preview`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L30`
- `llama-3.2-1b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L41`
