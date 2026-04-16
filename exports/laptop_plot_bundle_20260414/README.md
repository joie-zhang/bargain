# Laptop Plot Bundle (CSV-first)

This bundle is set up so you can tweak figure aesthetics on your laptop without raw experiment logs.

## Included

- `data/n2/game_1/*.csv`
- `data/n2/game_2/*.csv`
- `data/n2/game_3/*.csv`
- `data/n_gt_2/game1_multiagent_full_20260413_045538/*.csv`
- `scripts/plot_game1_from_csv.py`
- `scripts/plot_game2_from_csv.py`
- `scripts/plot_game3_from_csv.py`
- `scripts/plot_game1_multiagent_from_csv.py`
- `reference_figures/*.png` (snapshot targets for visual comparison)

## Quick Start

```bash
cd laptop_plot_bundle_20260414
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p outputs
python scripts/plot_game1_from_csv.py --input-dir data/n2/game_1 --output-dir outputs/game_1
python scripts/plot_game2_from_csv.py --input-dir data/n2/game_2 --output-dir outputs/game_2
python scripts/plot_game3_from_csv.py --input-dir data/n2/game_3 --output-dir outputs/game_3
python scripts/plot_game1_multiagent_from_csv.py \
  --summary-csv data/n_gt_2/game1_multiagent_full_20260413_045538/summary_by_model.csv \
  --output-dir outputs/n_gt_2
```

## Notes

- All scripts are CSV-driven and standalone (no repo-specific imports).
- Game 2 CSVs were exported from `diplomacy_20260405_082215` to match your current figure snapshot.
- For n>2, the included summary is from `game1_multiagent_full_20260413_045538`.
