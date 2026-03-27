# Lewis Slides Co-Funding (Game 3) Report

- Experiment dir: `experiments/results/cofunding_20260316_071259; experiments/results/cofunding_20260316_071418`
- Model inclusion rule: `runs >= 1`
- alpha values: [np.float64(0.0), np.float64(1.0)]
- sigma values: [np.float64(0.3), np.float64(1.0)]
- Competition index: CI₃ = (1−α)·(1−σ)

## Generated Plots

- `MAIN_PLOT_1_FOCAL_PAYOFF_ALPHA.png`
- `MAIN_PLOT_2_REFERENCE_PAYOFF_ALPHA.png`
- `MAIN_PLOT_3_FOCAL_PAYOFF_CI.png`
- `MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png`

## Models Used Per Plot

### `MAIN_PLOT_1_FOCAL_PAYOFF_ALPHA.png`
- Count: 2
- Models: claude-opus-4-6, gpt-5-nano

### `MAIN_PLOT_2_REFERENCE_PAYOFF_ALPHA.png`
- Count: 2
- Models: claude-opus-4-6, gpt-5-nano

### `MAIN_PLOT_3_FOCAL_PAYOFF_CI.png`
- Count: 2
- Models: claude-opus-4-6, gpt-5-nano

### `MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png`
- Count: 2
- Models: claude-opus-4-6, gpt-5-nano

## Run Counts Per Focal Model

| model | elo | total_runs |
|---|---:|---:|
| claude-opus-4-6 | 1475 | 8 |
| gpt-5-nano | 1338 | 8 |

## Grid Coverage (runs per alpha × sigma cell)

|   alpha |   0.3 |   1.0 |
|--------:|------:|------:|
|       0 |     8 |     0 |
|       1 |     0 |     8 |
