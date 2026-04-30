# Full Games 1-3 Clean-Subset Preliminary Analysis

- Results root: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255`
- Audit source: none
- Mode: `clean_152_subset=False`
- Exclude synthetic proposals: `False`
- Exclude controls: `False`
- Total successful result rows loaded: `1166`
- Exported subset run rows: `1166`
- Plot-relevant non-control rows: `1039`
- Rows with zero proposal fallback markers: `1001`
- Rows with proposal fallback markers: `165`

## Composition

| game_label   | experiment_family     |   n_agents | competition_key        |   run_count |
|:-------------|:----------------------|-----------:|:-----------------------|------------:|
| game1        | homogeneous_adversary |          2 | comp=0.00              |          20 |
| game1        | homogeneous_adversary |          2 | comp=0.25              |          20 |
| game1        | homogeneous_adversary |          2 | comp=0.50              |          20 |
| game1        | homogeneous_adversary |          2 | comp=0.75              |          20 |
| game1        | homogeneous_adversary |          2 | comp=1.00              |          20 |
| game1        | homogeneous_adversary |          4 | comp=0.00              |          20 |
| game1        | homogeneous_adversary |          4 | comp=0.25              |          19 |
| game1        | homogeneous_adversary |          4 | comp=0.50              |          16 |
| game1        | homogeneous_adversary |          4 | comp=0.75              |          14 |
| game1        | homogeneous_adversary |          4 | comp=1.00              |          17 |
| game1        | homogeneous_adversary |          6 | comp=0.00              |          18 |
| game1        | homogeneous_adversary |          6 | comp=0.25              |          15 |
| game1        | homogeneous_adversary |          6 | comp=0.50              |          15 |
| game1        | homogeneous_adversary |          6 | comp=0.75              |          13 |
| game1        | homogeneous_adversary |          6 | comp=1.00              |          16 |
| game1        | homogeneous_adversary |          8 | comp=0.00              |          16 |
| game1        | homogeneous_adversary |          8 | comp=0.25              |          10 |
| game1        | homogeneous_adversary |          8 | comp=0.50              |          14 |
| game1        | homogeneous_adversary |          8 | comp=0.75              |          12 |
| game1        | homogeneous_adversary |          8 | comp=1.00              |          10 |
| game1        | homogeneous_adversary |         10 | comp=0.00              |          17 |
| game1        | homogeneous_adversary |         10 | comp=0.25              |          14 |
| game1        | homogeneous_adversary |         10 | comp=0.50              |          11 |
| game1        | homogeneous_adversary |         10 | comp=0.75              |          14 |
| game1        | homogeneous_adversary |         10 | comp=1.00              |          15 |
| game1        | homogeneous_control   |          2 | comp=0.00              |           2 |
| game1        | homogeneous_control   |          2 | comp=0.25              |           2 |
| game1        | homogeneous_control   |          2 | comp=0.50              |           2 |
| game1        | homogeneous_control   |          2 | comp=0.75              |           2 |
| game1        | homogeneous_control   |          2 | comp=1.00              |           2 |
| game1        | homogeneous_control   |          4 | comp=0.00              |           2 |
| game1        | homogeneous_control   |          4 | comp=0.25              |           2 |
| game1        | homogeneous_control   |          4 | comp=0.50              |           2 |
| game1        | homogeneous_control   |          4 | comp=0.75              |           2 |
| game1        | homogeneous_control   |          4 | comp=1.00              |           2 |
| game1        | homogeneous_control   |          6 | comp=0.00              |           2 |
| game1        | homogeneous_control   |          6 | comp=0.25              |           2 |
| game1        | homogeneous_control   |          6 | comp=0.50              |           2 |
| game1        | homogeneous_control   |          6 | comp=0.75              |           2 |
| game1        | homogeneous_control   |          6 | comp=1.00              |           2 |
| game1        | homogeneous_control   |          8 | comp=0.00              |           2 |
| game1        | homogeneous_control   |          8 | comp=0.25              |           2 |
| game1        | homogeneous_control   |          8 | comp=0.50              |           2 |
| game1        | homogeneous_control   |          8 | comp=0.75              |           2 |
| game1        | homogeneous_control   |          8 | comp=1.00              |           2 |
| game1        | homogeneous_control   |         10 | comp=0.00              |           1 |
| game1        | homogeneous_control   |         10 | comp=0.25              |           1 |
| game1        | homogeneous_control   |         10 | comp=0.50              |           2 |
| game1        | homogeneous_control   |         10 | comp=0.75              |           2 |
| game1        | homogeneous_control   |         10 | comp=1.00              |           2 |
| game2        | homogeneous_adversary |          2 | rho=-1.000, theta=0.20 |          20 |
| game2        | homogeneous_adversary |          2 | rho=-1.000, theta=0.80 |          17 |
| game2        | homogeneous_adversary |          2 | rho=0.900, theta=0.20  |          20 |
| game2        | homogeneous_adversary |          2 | rho=0.900, theta=0.80  |          19 |
| game2        | homogeneous_adversary |          4 | rho=-0.320, theta=0.20 |          17 |
| game2        | homogeneous_adversary |          4 | rho=-0.320, theta=0.80 |          16 |
| game2        | homogeneous_adversary |          4 | rho=0.900, theta=0.20  |          17 |
| game2        | homogeneous_adversary |          4 | rho=0.900, theta=0.80  |          17 |
| game2        | homogeneous_adversary |          6 | rho=-0.191, theta=0.20 |          16 |
| game2        | homogeneous_adversary |          6 | rho=-0.191, theta=0.80 |          16 |
| game2        | homogeneous_adversary |          6 | rho=0.900, theta=0.20  |          16 |
| game2        | homogeneous_adversary |          6 | rho=0.900, theta=0.80  |          17 |
| game2        | homogeneous_adversary |          8 | rho=-0.137, theta=0.20 |          16 |
| game2        | homogeneous_adversary |          8 | rho=-0.137, theta=0.80 |          16 |
| game2        | homogeneous_adversary |          8 | rho=0.900, theta=0.20  |          16 |
| game2        | homogeneous_adversary |          8 | rho=0.900, theta=0.80  |          15 |
| game2        | homogeneous_adversary |         10 | rho=-0.106, theta=0.20 |          16 |
| game2        | homogeneous_adversary |         10 | rho=-0.106, theta=0.80 |          16 |
| game2        | homogeneous_adversary |         10 | rho=0.900, theta=0.20  |          16 |
| game2        | homogeneous_adversary |         10 | rho=0.900, theta=0.80  |          17 |
| game2        | homogeneous_control   |          2 | rho=-1.000, theta=0.20 |           2 |
| game2        | homogeneous_control   |          2 | rho=-1.000, theta=0.80 |           2 |
| game2        | homogeneous_control   |          2 | rho=0.900, theta=0.20  |           2 |
| game2        | homogeneous_control   |          2 | rho=0.900, theta=0.80  |           2 |
| game2        | homogeneous_control   |          4 | rho=-0.320, theta=0.20 |           2 |
| game2        | homogeneous_control   |          4 | rho=-0.320, theta=0.80 |           2 |
| game2        | homogeneous_control   |          4 | rho=0.900, theta=0.20  |           2 |
| game2        | homogeneous_control   |          4 | rho=0.900, theta=0.80  |           2 |
| game2        | homogeneous_control   |          6 | rho=-0.191, theta=0.20 |           2 |
| game2        | homogeneous_control   |          6 | rho=-0.191, theta=0.80 |           2 |

## Model Appearances

| experiment_family     | model                    | model_short    |   elo |   agent_appearances |
|:----------------------|:-------------------------|:---------------|------:|--------------------:|
| homogeneous_adversary | amazon-nova-micro-v1.0   | Nova Micro     |  1240 |                 228 |
| homogeneous_adversary | gpt-4o-mini-2024-07-18   | GPT-4o mini    |  1317 |                 246 |
| homogeneous_adversary | gpt-5-nano               | GPT-5-nano     |  1337 |                4833 |
| homogeneous_adversary | claude-sonnet-4-20250514 | Sonnet 4       |  1389 |                 223 |
| homogeneous_adversary | gemini-2.5-pro           | Gemini 2.5 Pro |  1448 |                 257 |
| homogeneous_adversary | gpt-5.4-high             | GPT-5.4 High   |  1484 |                  85 |
| homogeneous_control   | gpt-5-nano               | GPT-5-nano     |  1337 |                 752 |

## Plots

- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_raw.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_raw_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_raw_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_faceted_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_competition_by_n.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_by_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_adversary_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_adversary_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game2_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_adversary_payoff_vs_elo_mean_avg_over_n_and_competition.png`
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/full_games123_production_20260428_085255_plots_20260429/all_success_game3_hom_baseline_payoff_vs_adversary_elo_mean_avg_over_n_and_competition.png`
