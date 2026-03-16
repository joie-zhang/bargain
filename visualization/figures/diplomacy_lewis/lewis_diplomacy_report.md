# Lewis Slides Diplomacy Report

- Experiment dir: `experiments/results/diplomacy_20260223_032204`
- Model inclusion rule: `runs >= 5`
- rho values: [np.float64(-1.0), np.float64(-0.5), np.float64(0.0), np.float64(0.5), np.float64(1.0)]
- theta values: [np.float64(0.0), np.float64(0.25), np.float64(0.5), np.float64(0.75), np.float64(1.0)]
- Competition index: CI = theta × (1 − rho) / 2

## Generated Plots

- `MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png`
- `MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png`
- `MAIN_PLOT_3_BASELINE_PAYOFF_CI.png`
- `MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png`

## Models Used Per Plot

### `MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png`
- Count: 8
- Models: amazon-nova-micro, claude-haiku-4-5, claude-sonnet-4-5, gemini-3-pro, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, o3-mini-high

### `MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png`
- Count: 8
- Models: amazon-nova-micro, claude-haiku-4-5, claude-sonnet-4-5, gemini-3-pro, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, o3-mini-high

### `MAIN_PLOT_3_BASELINE_PAYOFF_CI.png`
- Count: 8
- Models: amazon-nova-micro, claude-haiku-4-5, claude-sonnet-4-5, gemini-3-pro, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, o3-mini-high

### `MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png`
- Count: 8
- Models: amazon-nova-micro, claude-haiku-4-5, claude-sonnet-4-5, gemini-3-pro, gpt-3.5-turbo-0125, gpt-4o, gpt-5.2-high, o3-mini-high

## Run Counts Per Adversary Model

| model | elo | total_runs |
|---|---:|---:|
| amazon-nova-micro | 1220 | 150 |
| claude-haiku-4-5 | 1403 | 150 |
| claude-sonnet-4-5 | 1450 | 150 |
| gemini-3-pro | 1490 | 150 |
| gpt-3.5-turbo-0125 | 1105 | 150 |
| gpt-4o | 1346 | 150 |
| gpt-5.2-high | 1436 | 150 |
| o3-mini-high | 1364 | 150 |

## Grid Coverage (runs per rho × theta cell)

|   rho |   0.0 |   0.25 |   0.5 |   0.75 |   1.0 |
|------:|------:|-------:|------:|-------:|------:|
|  -1   |    48 |     48 |    48 |     48 |    48 |
|  -0.5 |    48 |     48 |    48 |     48 |    48 |
|   0   |    48 |     48 |    48 |     48 |    48 |
|   0.5 |    48 |     48 |    48 |     48 |    48 |
|   1   |    48 |     48 |    48 |     48 |    48 |
