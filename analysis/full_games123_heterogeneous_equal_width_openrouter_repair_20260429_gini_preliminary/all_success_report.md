# Full Games 1-3 All-Success Preliminary Analysis

- Results root: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848`
- Exclude synthetic proposals: `False`
- Exclude controls: `False`
- Total successful result rows loaded: `624`
- Exported subset run rows: `624`
- Plot-relevant non-control rows: `624`
- Rows with zero proposal fallback markers: `510`
- Rows with proposal fallback markers: `114`
- Gini small-N correction: `raw_gini * N/(N-1)` for complete runs with `N < 8`; raw and correction factor columns are also exported.
- Payoff-vs-Elo slope: per-run OLS fit of `final_utility ~ elo`; signed and absolute slopes are exported per 100 Elo points.

## Composition

| game_label   | experiment_family    |   n_agents | competition_key        |   run_count |
|:-------------|:---------------------|-----------:|:-----------------------|------------:|
| game1        | heterogeneous_random |          2 | comp=0.00              |          19 |
| game1        | heterogeneous_random |          2 | comp=0.25              |          20 |
| game1        | heterogeneous_random |          2 | comp=0.50              |          20 |
| game1        | heterogeneous_random |          2 | comp=0.75              |          20 |
| game1        | heterogeneous_random |          2 | comp=1.00              |          19 |
| game1        | heterogeneous_random |          4 | comp=0.00              |          13 |
| game1        | heterogeneous_random |          4 | comp=0.25              |          19 |
| game1        | heterogeneous_random |          4 | comp=0.50              |          18 |
| game1        | heterogeneous_random |          4 | comp=0.75              |          16 |
| game1        | heterogeneous_random |          4 | comp=1.00              |          19 |
| game1        | heterogeneous_random |          6 | comp=0.00              |          10 |
| game1        | heterogeneous_random |          6 | comp=0.25              |          13 |
| game1        | heterogeneous_random |          6 | comp=0.50              |          13 |
| game1        | heterogeneous_random |          6 | comp=0.75              |          14 |
| game1        | heterogeneous_random |          6 | comp=1.00              |          10 |
| game1        | heterogeneous_random |          8 | comp=0.00              |           2 |
| game1        | heterogeneous_random |          8 | comp=0.25              |           5 |
| game1        | heterogeneous_random |          8 | comp=0.50              |           4 |
| game1        | heterogeneous_random |          8 | comp=0.75              |           2 |
| game1        | heterogeneous_random |          8 | comp=1.00              |           4 |
| game1        | heterogeneous_random |         10 | comp=0.00              |           4 |
| game1        | heterogeneous_random |         10 | comp=0.25              |           1 |
| game1        | heterogeneous_random |         10 | comp=0.50              |           3 |
| game1        | heterogeneous_random |         10 | comp=0.75              |           1 |
| game2        | heterogeneous_random |          2 | rho=-1.000, theta=0.20 |          20 |
| game2        | heterogeneous_random |          2 | rho=-1.000, theta=0.80 |          20 |
| game2        | heterogeneous_random |          2 | rho=0.900, theta=0.20  |          20 |
| game2        | heterogeneous_random |          2 | rho=0.900, theta=0.80  |          19 |
| game2        | heterogeneous_random |          4 | rho=-0.320, theta=0.20 |          18 |
| game2        | heterogeneous_random |          4 | rho=-0.320, theta=0.80 |          18 |
| game2        | heterogeneous_random |          4 | rho=0.900, theta=0.20  |          19 |
| game2        | heterogeneous_random |          4 | rho=0.900, theta=0.80  |          18 |
| game2        | heterogeneous_random |          6 | rho=-0.191, theta=0.20 |          17 |
| game2        | heterogeneous_random |          6 | rho=-0.191, theta=0.80 |          18 |
| game2        | heterogeneous_random |          6 | rho=0.900, theta=0.20  |          17 |
| game2        | heterogeneous_random |          6 | rho=0.900, theta=0.80  |          18 |
| game2        | heterogeneous_random |          8 | rho=-0.137, theta=0.20 |          14 |
| game2        | heterogeneous_random |          8 | rho=-0.137, theta=0.80 |           6 |
| game2        | heterogeneous_random |          8 | rho=0.900, theta=0.20  |          13 |
| game2        | heterogeneous_random |          8 | rho=0.900, theta=0.80  |          12 |
| game2        | heterogeneous_random |         10 | rho=-0.106, theta=0.20 |           9 |
| game2        | heterogeneous_random |         10 | rho=-0.106, theta=0.80 |           2 |
| game2        | heterogeneous_random |         10 | rho=0.900, theta=0.20  |          14 |
| game2        | heterogeneous_random |         10 | rho=0.900, theta=0.80  |          12 |
| game3        | heterogeneous_random |          2 | sigma=0.20, alpha=0.20 |          18 |
| game3        | heterogeneous_random |          2 | sigma=0.20, alpha=0.80 |          17 |
| game3        | heterogeneous_random |          2 | sigma=0.50, alpha=0.20 |          10 |
| game3        | heterogeneous_random |          2 | sigma=0.50, alpha=0.80 |           6 |

## Model Appearances

| experiment_family    | model                                 | model_short       |   elo |   agent_appearances |
|:---------------------|:--------------------------------------|:------------------|------:|--------------------:|
| heterogeneous_random | amazon-nova-micro-v1.0                | Nova Micro        |  1240 |                 158 |
| heterogeneous_random | claude-3-haiku-20240307               | Claude 3 Haiku    |  1260 |                 133 |
| heterogeneous_random | command-r-plus-08-2024                | Command R+        |  1276 |                 123 |
| heterogeneous_random | amazon-nova-pro-v1.0                  | Nova Pro          |  1290 |                 106 |
| heterogeneous_random | gpt-4o-mini-2024-07-18                | GPT-4o mini       |  1317 |                 102 |
| heterogeneous_random | llama-3.3-70b-instruct                | Llama 3.3 70B     |  1318 |                 116 |
| heterogeneous_random | gpt-4.1-nano-2025-04-14               | GPT-4.1 nano      |  1322 |                 105 |
| heterogeneous_random | gpt-5-nano-high                       | GPT-5-nano        |  1337 |                  71 |
| heterogeneous_random | gpt-4o-2024-05-13                     | GPT-4o            |  1345 |                 124 |
| heterogeneous_random | deepseek-v3                           | DeepSeek V3       |  1358 |                 111 |
| heterogeneous_random | o3-mini-high                          | o3-mini-high      |  1363 |                 122 |
| heterogeneous_random | gemma-3-27b-it                        | Gemma 3 27B       |  1365 |                  88 |
| heterogeneous_random | claude-sonnet-4-20250514              | Sonnet 4          |  1389 |                 136 |
| heterogeneous_random | claude-haiku-4-5-20251001             | Haiku 4.5         |  1407 |                 109 |
| heterogeneous_random | deepseek-r1-0528                      | DeepSeek R1-0528  |  1422 |                 111 |
| heterogeneous_random | qwen3-max-preview                     | Qwen3 Max         |  1435 |                 116 |
| heterogeneous_random | gemini-2.5-pro                        | Gemini 2.5 Pro    |  1448 |                 122 |
| heterogeneous_random | claude-opus-4-5-20251101              | Opus 4.5          |  1468 |                 112 |
| heterogeneous_random | claude-opus-4-5-20251101-thinking-32k | Opus 4.5 Thinking |  1474 |                 125 |
| heterogeneous_random | gpt-5.2-chat-latest-20260210          | GPT-5.2 Chat      |  1478 |                 127 |
| heterogeneous_random | gpt-5.4-high                          | GPT-5.4 High      |  1484 |                 117 |
| heterogeneous_random | gemini-3.1-pro                        | gemini-3.1-pro    |  1494 |                 124 |
| heterogeneous_random | claude-opus-4-6                       | Opus 4.6          |  1499 |                 130 |
| heterogeneous_random | claude-opus-4-6-thinking              | Opus 4.6 Thinking |  1504 |                 136 |

## Plots

- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_gini_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_utility_std_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_gini_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_utility_std_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_gini_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_utility_std_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_stddev_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_variance.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_payoff_elo_abs_slope_vs_elo_variance_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_gini_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_utility_std_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_stddev_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_by_n_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_mean_payoff_elo_abs_slope_vs_elo_variance_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game1_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game2_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary/all_success_game3_heterogeneous_agent_payoff_vs_elo_mean_avg_over_n_and_competition.png`
