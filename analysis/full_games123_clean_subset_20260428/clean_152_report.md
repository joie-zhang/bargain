# Full Games 1-3 Clean-Subset Preliminary Analysis

- Results root: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_20260427_040554`
- Audit source: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_20260427_040554/monitoring/salvageability_audit_20260427_2118.json`
- Mode: `clean_152_subset=True`
- Exclude synthetic proposals: `False`
- Exclude controls: `False`
- Total successful result rows loaded: `429`
- Exported subset run rows: `158`
- Plot-relevant non-control rows: `154`
- Rows with zero proposal fallback markers: `118`
- Rows with proposal fallback markers: `40`

## Composition

| game_label   | experiment_family     |   n_agents | competition_key   |   run_count |
|:-------------|:----------------------|-----------:|:------------------|------------:|
| game1        | heterogeneous_random  |          2 | comp=0.00         |          18 |
| game1        | heterogeneous_random  |          2 | comp=0.25         |           8 |
| game1        | heterogeneous_random  |          2 | comp=0.50         |          12 |
| game1        | heterogeneous_random  |          2 | comp=0.75         |          12 |
| game1        | heterogeneous_random  |          2 | comp=1.00         |          14 |
| game1        | heterogeneous_random  |          4 | comp=0.00         |           6 |
| game1        | heterogeneous_random  |          4 | comp=0.25         |           5 |
| game1        | heterogeneous_random  |          4 | comp=0.50         |           7 |
| game1        | heterogeneous_random  |          4 | comp=0.75         |           7 |
| game1        | heterogeneous_random  |          4 | comp=1.00         |           6 |
| game1        | heterogeneous_random  |          6 | comp=0.00         |           5 |
| game1        | heterogeneous_random  |          6 | comp=0.25         |           5 |
| game1        | heterogeneous_random  |          6 | comp=0.50         |           3 |
| game1        | homogeneous_adversary |          2 | comp=0.00         |          11 |
| game1        | homogeneous_adversary |          2 | comp=0.25         |           5 |
| game1        | homogeneous_adversary |          2 | comp=0.50         |           8 |
| game1        | homogeneous_adversary |          2 | comp=0.75         |           9 |
| game1        | homogeneous_adversary |          2 | comp=1.00         |          11 |
| game1        | homogeneous_adversary |          4 | comp=0.00         |           2 |
| game1        | homogeneous_control   |          2 | comp=0.00         |           2 |
| game1        | homogeneous_control   |          2 | comp=0.50         |           1 |
| game1        | homogeneous_control   |          2 | comp=0.75         |           1 |

## Model Appearances

| experiment_family     | model                                 | model_short       |   elo |   agent_appearances |
|:----------------------|:--------------------------------------|:------------------|------:|--------------------:|
| heterogeneous_random  | amazon-nova-micro-v1.0                | Nova Micro        |  1240 |                  14 |
| heterogeneous_random  | claude-3-haiku-20240307               | Claude 3 Haiku    |  1260 |                  10 |
| heterogeneous_random  | command-r-plus-08-2024                | Command R+        |  1276 |                  10 |
| heterogeneous_random  | amazon-nova-pro-v1.0                  | Nova Pro          |  1290 |                  18 |
| heterogeneous_random  | gpt-4o-mini-2024-07-18                | GPT-4o mini       |  1317 |                  20 |
| heterogeneous_random  | llama-3.3-70b-instruct                | Llama 3.3 70B     |  1318 |                  20 |
| heterogeneous_random  | gpt-4.1-nano-2025-04-14               | GPT-4.1 nano      |  1322 |                  18 |
| heterogeneous_random  | gpt-4o-2024-05-13                     | GPT-4o            |  1345 |                  21 |
| heterogeneous_random  | deepseek-v3                           | DeepSeek V3       |  1358 |                  20 |
| heterogeneous_random  | o3-mini-high                          | o3-mini-high      |  1363 |                  26 |
| heterogeneous_random  | gemma-3-27b-it                        | Gemma 3 27B       |  1365 |                  23 |
| heterogeneous_random  | claude-sonnet-4-20250514              | Sonnet 4          |  1389 |                  21 |
| heterogeneous_random  | claude-haiku-4-5-20251001             | Haiku 4.5         |  1407 |                  10 |
| heterogeneous_random  | deepseek-r1-0528                      | DeepSeek R1-0528  |  1422 |                   6 |
| heterogeneous_random  | qwen3-max-preview                     | Qwen3 Max         |  1435 |                  18 |
| heterogeneous_random  | gemini-2.5-pro                        | Gemini 2.5 Pro    |  1448 |                   2 |
| heterogeneous_random  | claude-opus-4-5-20251101              | Opus 4.5          |  1468 |                  17 |
| heterogeneous_random  | claude-opus-4-5-20251101-thinking-32k | Opus 4.5 Thinking |  1474 |                  13 |
| heterogeneous_random  | gpt-5.2-chat-latest-20260210          | GPT-5.2 Chat      |  1478 |                   8 |
| heterogeneous_random  | gpt-5.4-high                          | GPT-5.4 High      |  1484 |                   6 |
| heterogeneous_random  | gemini-3-pro                          | Gemini 3 Pro      |  1486 |                   1 |
| heterogeneous_random  | claude-opus-4-6                       | Opus 4.6          |  1499 |                  17 |
| heterogeneous_random  | claude-opus-4-6-thinking              | Opus 4.6 Thinking |  1504 |                  11 |
| homogeneous_adversary | amazon-nova-micro-v1.0                | Nova Micro        |  1240 |                  14 |
| homogeneous_adversary | gpt-4o-mini-2024-07-18                | GPT-4o mini       |  1317 |                   7 |
| homogeneous_adversary | gpt-5-nano                            | GPT-5-nano        |  1337 |                  50 |
| homogeneous_adversary | claude-sonnet-4-20250514              | Sonnet 4          |  1389 |                  15 |
| homogeneous_adversary | gemini-2.5-pro                        | Gemini 2.5 Pro    |  1448 |                   3 |
| homogeneous_adversary | claude-opus-4-6-thinking              | Opus 4.6 Thinking |  1504 |                   7 |
| homogeneous_control   | gpt-5-nano                            | GPT-5-nano        |  1337 |                   8 |

## Plots

- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_gini_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_gini_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_gini_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_gini_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_mean_gini_vs_mean_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_mean_gini_vs_mean_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_mean_gini_vs_mean_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_mean_gini_vs_mean_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_adversary_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_clean_subset_20260428/clean_152_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_and_competition.png`
