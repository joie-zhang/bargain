# Lewis Slides Plot Inputs (Mar 1)

This report is generated from CSV only (no raw JSON dependency).

- Input CSV: `/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/gpt5_nano_full_data.csv`
- Model inclusion rule for plotted lines: `successful runs >= 10`
- Main plots are generated with: compression + smoothing

## Generated Plot Files

- `MAIN_PLOT_1_BASELINE_PAYOFF.png`
- `MAIN_LOT_2_ADVERSARY_PAYOFF.png`

## Models Used Per Plot

### `MAIN_PLOT_1_BASELINE_PAYOFF.png`

- Number of models: `32`
- Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-70B-Instruct, Phi-3-mini-128k-instruct, QwQ-32B, Qwen2.5-72B-Instruct, amazon-nova-micro, claude-3-haiku, claude-3.5-sonnet, claude-haiku-4-5, claude-opus-4-5, claude-opus-4-5-thinking-32k, claude-sonnet-4, claude-sonnet-4-5, deepseek-r1, deepseek-r1-0528, deepseek-v3, gemini-3-flash, gemini-3-pro, gemma-2-27b-it, gemma-3-27b-it, glm-4.7, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, grok-4, llama-3.3-70b-instruct, mixtral-8x22b-instruct-v0.1, mixtral-8x7b-instruct-v0.1, o3-mini-high, phi-4, qwen3-max

### `MAIN_LOT_2_ADVERSARY_PAYOFF.png`

- Number of models: `32`
- Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-70B-Instruct, Phi-3-mini-128k-instruct, QwQ-32B, Qwen2.5-72B-Instruct, amazon-nova-micro, claude-3-haiku, claude-3.5-sonnet, claude-haiku-4-5, claude-opus-4-5, claude-opus-4-5-thinking-32k, claude-sonnet-4, claude-sonnet-4-5, deepseek-r1, deepseek-r1-0528, deepseek-v3, gemini-3-flash, gemini-3-pro, gemma-2-27b-it, gemma-3-27b-it, glm-4.7, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, grok-4, llama-3.3-70b-instruct, mixtral-8x22b-instruct-v0.1, mixtral-8x7b-instruct-v0.1, o3-mini-high, phi-4, qwen3-max

## Successful Runs Per Model Per Competition Level

| adversary_model | elo | total_successful_runs | comp_0.0 | comp_0.1 | comp_0.2 | comp_0.3 | comp_0.4 | comp_0.5 | comp_0.6 | comp_0.7 | comp_0.8 | comp_0.9 | comp_1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini-3-pro | 1490 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| gemini-3-flash | 1472 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| claude-opus-4-5-thinking-32k | 1470 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| claude-opus-4-5 | 1467 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| claude-sonnet-4-5 | 1450 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| glm-4.7 | 1441 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| gpt-5.2-high | 1436 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| qwen3-max | 1434 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| deepseek-r1-0528 | 1418 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| grok-4 | 1409 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| claude-haiku-4-5 | 1403 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| deepseek-r1 | 1397 | 20 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| claude-sonnet-4 | 1390 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| claude-3.5-sonnet | 1373 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| gemma-3-27b-it | 1365 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| o3-mini-high | 1364 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| deepseek-v3 | 1358 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| gpt-4o | 1346 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| QwQ-32B | 1336 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| llama-3.3-70b-instruct | 1320 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Qwen2.5-72B-Instruct | 1303 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| gemma-2-27b-it | 1288 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| Meta-Llama-3-70B-Instruct | 1277 | 20 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 2 |
| claude-3-haiku | 1262 | 20 | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 |
| phi-4 | 1256 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| amazon-nova-micro | 1241 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| mixtral-8x22b-instruct-v0.1 | 1231 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 |
| gpt-3.5-turbo-0125 | 1225 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| llama-3.1-8b-instruct | 1212 | 7 | 0 | 0 | 1 | 1 | 1 | 2 | 1 | 0 | 0 | 1 | 0 |
| mixtral-8x7b-instruct-v0.1 | 1198 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Llama-3.2-3B-Instruct | 1167 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Mistral-7B-Instruct-v0.2 | 1151 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Phi-3-mini-128k-instruct | 1130 | 22 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Llama-3.2-1B-Instruct | 1112 | 21 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 |
