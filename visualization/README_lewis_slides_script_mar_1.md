# Lewis Slides Script (Mar 1)

This README is for:

- `visualization/lewis_slides_script_mar_1.py`

It is a CSV-only plotting script (no raw JSON dependency).

## What It Reads

- `visualization/figures/gpt5_nano_full_data.csv` (by default)

## What It Produces

- Main slide plots (always):
  - `MAIN_PLOT_1_BASELINE_PAYOFF.png`
  - `MAIN_PLOT_2_ADVERSARY_PAYOFF.png`
- Markdown report:
  - `lewis_slides_plot_report_mar_1.md` (default name)

The report includes:

- Which models were used for each generated plot
- Successful run counts per model per competition level

## Quick Start (uv)

Run from repo root.

```bash
uv venv
source .venv/bin/activate
uv pip install pandas numpy matplotlib
python visualization/lewis_slides_script_mar_1.py
```

If you prefer not to activate a venv:

```bash
uv run --with pandas --with numpy --with matplotlib \
  python visualization/lewis_slides_script_mar_1.py
```

## Default Behavior

`python visualization/lewis_slides_script_mar_1.py`

- Uses `figures/gpt5_nano_full_data.csv` relative to the script directory
- Writes outputs to `figures/` relative to the script directory
- Filters out models with `< 10` successful runs
- Main plots use: compression + smoothing (default smoothing method: `ewm`)

## Generate All Plot Variants

To generate raw, compressed, smoothed, and compressed+smoothed variants:

```bash
python visualization/lewis_slides_script_mar_1.py --make-all-variants
```

## Useful CLI Examples

Use moving-average smoothing instead of EWM:

```bash
python visualization/lewis_slides_script_mar_1.py \
  --smoothing-method moving_average \
  --smoothing-window 3
```

Tune EWM smoothing strength:

```bash
python visualization/lewis_slides_script_mar_1.py \
  --smoothing-method ewm \
  --smoothing-alpha 0.25
```

Disable model run-count filtering:

```bash
python visualization/lewis_slides_script_mar_1.py --min-runs-per-model 0
```

Write outputs to a custom folder:

```bash
python visualization/lewis_slides_script_mar_1.py \
  --output-dir figures/lewis_mar_1_custom
```

Use a different CSV file:

```bash
python visualization/lewis_slides_script_mar_1.py \
  --input-csv figures/another_full_data.csv
```

## Argument Reference

```text
--input-csv
--output-dir
--min-runs-per-model
--smoothing-method {ewm,moving_average}
--smoothing-alpha
--smoothing-window
--make-all-variants
--report-name
```

Show full help:

```bash
python visualization/lewis_slides_script_mar_1.py --help
```
