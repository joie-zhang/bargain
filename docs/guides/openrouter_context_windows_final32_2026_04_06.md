# OpenRouter Context Window Audit For The Final 32-Model Slate

Generated: 2026-04-11 18:46:34 EDT

This compares the repo's canonical native/full context window for each retained OpenRouter-backed model in the final 32-model slate against the current context window reported by OpenRouter.

Sources:

- Native/full context reference: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`
- Current OpenRouter model metadata: `https://openrouter.ai/api/v1/models`
- Fallback when a route is missing from the live OpenRouter model index: the public model page at `https://openrouter.ai/<route>`

Summary:

- OpenRouter-backed entries in the 32-model slate: 14
- Same context on OpenRouter: 3
- Shorter context on OpenRouter: 7
- Missing from live OpenRouter models API and resolved from page HTML: 0
- Missing OpenRouter context even after fallback: 0
- Excluded from this guide: retired `claude-3-5-sonnet-20241022` alias


| Elo Order | Final 32 Rank | Model                    | Arena Elo | OpenRouter Route                                                                               | Native / Full Context | OpenRouter Context Now | OpenRouter vs Native | OpenRouter / Native | Notes                                        |
| --------- | ------------- | ------------------------ | --------- | ---------------------------------------------------------------------------------------------- | --------------------- | ---------------------- | -------------------- | ------------------- | -------------------------------------------- |
| 1         | 9             | `qwen3-max-preview`      | 1435      | `[qwen/qwen3-max](https://openrouter.ai/qwen/qwen3-max)`                                       | 262.1K (262,100)      | 262,144                | +44                  | 100.0%              | repo alias to current `qwen/qwen3-max` route |
| 2         | 10            | `deepseek-r1-0528`       | 1422      | `[deepseek/deepseek-r1-0528](https://openrouter.ai/deepseek/deepseek-r1-0528)`                 | 163.8K (163,800)      | 163,840                | +40                  | 100.0%              |                                              |
| 3         | 12            | `deepseek-r1`            | 1398      | `[deepseek/deepseek-r1](https://openrouter.ai/deepseek/deepseek-r1)`                           | 128K (128,000)        | 64,000                 | -64,000              | 50.0%               |                                              |
| 4         | 15            | `gemma-3-27b-it`         | 1365      | `[google/gemma-3-27b-it](https://openrouter.ai/google/gemma-3-27b-it)`                         | 131.1K (131,100)      | 131,072                | -28                  | 100.0%              |                                              |
| 5         | 17            | `deepseek-v3`            | 1358      | `[deepseek/deepseek-chat](https://openrouter.ai/deepseek/deepseek-chat)`                       | 128K (128,000)        | 163,840                | +35,840              | 128.0%              |                                              |
| 6         | 19            | `qwq-32b`                | 1336      | `[qwen/qwq-32b](https://openrouter.ai/qwen/qwq-32b)`                                           | 131.1K (131,100)      | 131,072                | -28                  | 100.0%              |                                              |
| 7         | 21            | `llama-3.3-70b-instruct` | 1318      | `[meta-llama/llama-3.3-70b-instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct)` | 128K (128,000)        | 131,072                | +3,072               | 102.4%              |                                              |
| 8         | 23            | `qwen2.5-72b-instruct`   | 1302      | `[qwen/qwen-2.5-72b-instruct](https://openrouter.ai/qwen/qwen-2.5-72b-instruct)`               | 131K (131,000)        | 32,768                 | -98,232              | 25.0%               |                                              |
| 9         | 24            | `amazon-nova-pro-v1.0`   | 1290      | `[amazon/nova-pro-v1](https://openrouter.ai/amazon/nova-pro-v1)`                               | 300K (300,000)        | 300,000                | same                 | 100.0%              |                                              |
| 10        | 25            | `command-r-plus-08-2024` | 1276      | `[cohere/command-r-plus-08-2024](https://openrouter.ai/cohere/command-r-plus-08-2024)`         | 128K (128,000)        | 128,000                | same                 | 100.0%              |                                              |
| 11        | 27            | `amazon-nova-micro-v1.0` | 1240      | `[amazon/nova-micro-v1](https://openrouter.ai/amazon/nova-micro-v1)`                           | 128K (128,000)        | 128,000                | same                 | 100.0%              |                                              |
| 12        | 28            | `llama-3.1-8b-instruct`  | 1211      | `[meta-llama/llama-3.1-8b-instruct](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct)`   | 128K (128,000)        | 16,384                 | -111,616             | 12.8%               |                                              |
| 13        | 29            | `llama-3.2-3b-instruct`  | 1166      | `[meta-llama/llama-3.2-3b-instruct](https://openrouter.ai/meta-llama/llama-3.2-3b-instruct)`   | 128K (128,000)        | 80,000                 | -48,000              | 62.5%               |                                              |
| 14        | 32            | `llama-3.2-1b-instruct`  | 1110      | `[meta-llama/llama-3.2-1b-instruct](https://openrouter.ai/meta-llama/llama-3.2-1b-instruct)`   | 128K (128,000)        | 60,000                 | -68,000              | 46.9%               |                                              |


Canonical context rows consulted:

- `qwen3-max-preview`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L20`
- `deepseek-r1-0528`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L21`
- `deepseek-r1`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L23`
- `gemma-3-27b-it`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L25`
- `deepseek-v3`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L27`
- `qwq-32b`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L30`
- `llama-3.3-70b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L32`
- `qwen2.5-72b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L34`
- `amazon-nova-pro-v1.0`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L35`
- `command-r-plus-08-2024`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L36`
- `amazon-nova-micro-v1.0`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L38`
- `llama-3.1-8b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L39`
- `llama-3.2-3b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L40`
- `llama-3.2-1b-instruct`: `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md#L41`

