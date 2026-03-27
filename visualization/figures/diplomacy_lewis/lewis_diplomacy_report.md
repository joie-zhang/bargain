# Lewis Slides Diplomacy Report

- Experiment dir: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260316_071347`
- Model inclusion rule: `runs >= 1`
- rho values: [np.float64(-1.0), np.float64(1.0)]
- theta values: [np.float64(0.0), np.float64(1.0)]
- Competition index: CI = theta × (1 − rho) / 2

## Generated Plots

- `MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png`
- `MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png`
- `MAIN_PLOT_3_BASELINE_PAYOFF_CI.png`
- `MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png`

## Models Used Per Plot

### `MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png`
- Count: 1
- Models: gpt-5-nano

### `MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png`
- Count: 1
- Models: gpt-5-nano

### `MAIN_PLOT_3_BASELINE_PAYOFF_CI.png`
- Count: 1
- Models: gpt-5-nano

### `MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png`
- Count: 1
- Models: gpt-5-nano

## Run Counts Per Adversary Model

| model | elo | total_runs |
|---|---:|---:|
| gpt-5-nano | 1338 | 4 |

## Grid Coverage (runs per rho × theta cell)

|   rho |   0.0 |   1.0 |
|------:|------:|------:|
|    -1 |     0 |     2 |
|     1 |     2 |     0 |
