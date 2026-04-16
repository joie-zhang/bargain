MODELS=$(python - <<'PY'
from strong_models_experiment.analysis.active_model_roster import ACTIVE_ADVERSARY_MODELS
print(" ".join(ACTIVE_ADVERSARY_MODELS))
PY
)

python scripts/plot_scaling_utility_vs_elo.py \
  --results-root experiments/results/scaling_experiment_20260404_064451 \
  --elo-markdown docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md \
  --models $MODELS \
  --output-dir experiments/results/scaling_experiment_20260404_064451/analysis \
  --by-competition-level

python visualization/export_game2_batch_pngs.py \
  --results-dir experiments/results/diplomacy_20260405_082215 \
  --output-dir visualization/figures/diplomacy_20260405_082215_summary

python scripts/plot_game3_utility_vs_elo.py \
  --results-root experiments/results/cofunding_20260405_083548 \
  --output-dir experiments/results/cofunding_20260405_083548/analysis

python scripts/game1_multiagent_full_batch.py summary \
  --results-root experiments/results/game1_multiagent_full_20260413_045538 --json

python scripts/game1_multiagent_full_batch.py analyze \
  --results-root experiments/results/game1_multiagent_full_20260413_045538

python scripts/sync_thesis_figures.py