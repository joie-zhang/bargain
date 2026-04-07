# Game 2 Batch PNG Export

- Results root: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215`
- Total configs: `540`
- Completed configs: `532`
- Missing-result configs: `8`

## Coverage

| Adversary | Model Order | Completed | Missing Result |
|---|---|---:|---:|
| Nova Micro | strong_first | 9 | 0 |
| Nova Micro | weak_first | 9 | 0 |
| Nova Pro | strong_first | 9 | 0 |
| Nova Pro | weak_first | 9 | 0 |
| Claude 3 Haiku | strong_first | 9 | 0 |
| Claude 3 Haiku | weak_first | 9 | 0 |
| Haiku 4.5 | strong_first | 9 | 0 |
| Haiku 4.5 | weak_first | 9 | 0 |
| Opus 4.5 | strong_first | 9 | 0 |
| Opus 4.5 | weak_first | 9 | 0 |
| Opus 4.5 Thinking | strong_first | 9 | 0 |
| Opus 4.5 Thinking | weak_first | 9 | 0 |
| Opus 4.6 | strong_first | 9 | 0 |
| Opus 4.6 | weak_first | 9 | 0 |
| Opus 4.6 Thinking | strong_first | 9 | 0 |
| Opus 4.6 Thinking | weak_first | 9 | 0 |
| Sonnet 4 | strong_first | 9 | 0 |
| Sonnet 4 | weak_first | 9 | 0 |
| Command R+ | strong_first | 9 | 0 |
| Command R+ | weak_first | 9 | 0 |
| DeepSeek R1 | strong_first | 9 | 0 |
| DeepSeek R1 | weak_first | 9 | 0 |
| DeepSeek R1-0528 | strong_first | 9 | 0 |
| DeepSeek R1-0528 | weak_first | 9 | 0 |
| DeepSeek V3 | strong_first | 9 | 0 |
| DeepSeek V3 | weak_first | 9 | 0 |
| Gemini 2.5 Pro | strong_first | 9 | 0 |
| Gemini 2.5 Pro | weak_first | 9 | 0 |
| Gemini 3 Pro | strong_first | 9 | 0 |
| Gemini 3 Pro | weak_first | 9 | 0 |
| Gemma 3 27B | strong_first | 9 | 0 |
| Gemma 3 27B | weak_first | 9 | 0 |
| GPT-4.1 nano | strong_first | 9 | 0 |
| GPT-4.1 nano | weak_first | 9 | 0 |
| GPT-4o | strong_first | 9 | 0 |
| GPT-4o | weak_first | 9 | 0 |
| GPT-4o mini | strong_first | 9 | 0 |
| GPT-4o mini | weak_first | 9 | 0 |
| GPT-5-nano | strong_first | 9 | 0 |
| GPT-5-nano | weak_first | 9 | 0 |
| GPT-5.2 Chat | strong_first | 9 | 0 |
| GPT-5.2 Chat | weak_first | 9 | 0 |
| GPT-5.4 High | strong_first | 9 | 0 |
| GPT-5.4 High | weak_first | 9 | 0 |
| Llama 3.1 8B | strong_first | 9 | 0 |
| Llama 3.1 8B | weak_first | 9 | 0 |
| Llama 3.2 1B | strong_first | 1 | 8 |
| Llama 3.2 1B | weak_first | 9 | 0 |
| Llama 3.2 3B | strong_first | 9 | 0 |
| Llama 3.2 3B | weak_first | 9 | 0 |
| Llama 3.3 70B | strong_first | 9 | 0 |
| Llama 3.3 70B | weak_first | 9 | 0 |
| o3-mini-high | strong_first | 9 | 0 |
| o3-mini-high | weak_first | 9 | 0 |
| Qwen2.5 72B | strong_first | 9 | 0 |
| Qwen2.5 72B | weak_first | 9 | 0 |
| Qwen3 Max | strong_first | 9 | 0 |
| Qwen3 Max | weak_first | 9 | 0 |
| QwQ-32B | strong_first | 9 | 0 |
| QwQ-32B | weak_first | 9 | 0 |

## Files

- `utility_vs_elo_overall.png`
- `utility_vs_elo_by_competition_index.png`
- `utility_vs_elo_by_rho_theta.png`
- `utility_vs_theta.png`
- `utility_vs_rho.png`
- `utility_vs_competition_index.png`
- `utility_rho_theta_heatmap.png`
- `social_welfare_rho_theta_heatmap.png`

## Notes

- The Elo plots use adversary-model utility, not GPT-5-nano baseline utility.
- `utility_vs_elo_overall.png` averages over all completed `(rho, theta, model_order)` configs for each adversary model.
- `utility_vs_elo_by_rho_theta.png` uses the native Game 2 parameters directly: panels fix `theta`, and curves fix `rho`.
- `utility_vs_elo_by_competition_index.png` remains as a derived 1D summary using `CI2 = theta * (1 - rho) / 2`.
