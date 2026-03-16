# Lewis Slides Co-Funding (Game 3) Report

- Experiment dir: `experiments/results/cofunding_latest`
- Model inclusion rule: `runs >= 3`
- alpha values: [np.float64(0.0), np.float64(0.5), np.float64(1.0)]
- sigma values: [np.float64(0.2), np.float64(0.6), np.float64(1.0)]
- Competition index: CI₃ = (1−α)·(1−σ)

## Generated Plots

- `MAIN_PLOT_3_FOCAL_PAYOFF_CI.png`
- `MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png`

## Models Used Per Plot

### `MAIN_PLOT_3_FOCAL_PAYOFF_CI.png`
- Count: 4
- Models: claude-haiku-4-5, claude-opus-4-6, llama-3.1-8b-instruct, qwen3-32b

### `MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png`
- Count: 4
- Models: claude-haiku-4-5, claude-opus-4-6, llama-3.1-8b-instruct, qwen3-32b

## Run Counts Per Focal Model

| model | elo | total_runs |
|---|---:|---:|
| claude-haiku-4-5 | 1403 | 36 |
| claude-opus-4-6 | 1475 | 36 |
| llama-3.1-8b-instruct | 1180 | 6 |
| qwen3-32b | 1360 | 4 |

## Grid Coverage (runs per alpha × sigma cell)

|   alpha |   0.2 |   0.6 |   1.0 |
|--------:|------:|------:|------:|
|     0   |     8 |     8 |    10 |
|     0.5 |     8 |    12 |    12 |
|     1   |     8 |     8 |     8 |
