# E15: Game 3 homogeneous-adversary scaling

## Scope

Current appendix lines 582–604, labels `fig:appendix_multiagent_hom_payoff_full`, `fig:appendix_multiagent_hom_competition`, and `fig:appendix_multiagent_dilution`. This audit covers the Game 3 portion only. Figure numbering differs from the September 5 recreation: current assets correspond to recreation figures 24–26.

## Verified chain

- Batch generator/runner is `scripts/full_games123_multiagent_batch.py`; line 1535 builds a subprocess command to `run_strong_models_experiment.py` using saved config fields, Game 3 project/cost/discount options, seeds and model order.
- The retained batch's `slurm/run_full_games123.sbatch` calls `run-one`, supplies selected config IDs or an array offset, sources an external credential environment, and selects shared OpenRouter file-proxy transport. It uses the project virtual environment and CPU partition.
- Batch root is `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255`. Manifest gives master seed 20260427, five n values, five adversary models, sigma 0.2/0.5, alpha 0.2/0.8, two orders and two seed replicates. Game 3 project count is int(2.5n), not fixed across n.
- Selection is exactly configs with `experiment_family=homogeneous_adversary` and `game_label=game3`: 400 runs, including 42 without consensus. No new inference runs were performed.
- I independently opened all 400 selected source configs and results and checked family, game, role membership, and recomputed each payoff advantage from final utility and agent role maps against the retained raw metrics CSV.
- Endpoint average gap is 3.0110177185000007 at n=2 and 8.014440791666667 at n=10, reproducing the +5.003423073166666 change. Each endpoint includes 80 runs. The comparison does not isolate group size because project counts also change.
- Raw reader `scripts/plot_full_games123_clean_subset.py:355` reads results, status, interactions and config logs; the analysis in `scripts/analyze_n2_plus_multiagent_comparison.py` aggregates model/n/competition means and adversary-minus-baseline gaps.
- Current by-n rendering source is `lh_review_20260829/02_agent_n/candidates/fig24/reproduce.py`; current competition renderer is `scripts/paper_figures/plot_figure24_hom_adversary_competition_clean.py`, with latest review source under `lh_review_20260829/02_agent_n/candidates/fig25/recreate.py`.
- Dilution renderer `scripts/paper_figures/plot_figure30_dilution_top_only_clean.py` reads `reproduction_audit/fig30_multiagent_dilution/output/pooled_plot_points.csv` by default. This CSV is not safely disposable unless that default is changed and verified. Its producer is `reproduction_audit/fig30_multiagent_dilution/reproduce.py`.
- Competition renderer likewise defaults to `reproduction_audit/fig24_hom_adversary_competition/aggregated_plot_data.csv`. These old-looking directories contain live inputs.

## Historical evidence

Used the codex-search skill script for the repository with batch and dilution anchors. Original April/May execution session was not recovered. Manually inspected original full-history records at `/home/jz4391/.codex/sessions/2026/08/02/rollout-2026-08-02T01-57-18-019fc10c-5259-76a0-8efa-0e41696b8ab5.jsonl:1632` (creation patch and live CSV default) and `/home/jz4391/.codex/sessions/2026/08/29/rollout-2026-08-29T20-38-40-01a0501a-ae58-7fc3-93a1-917ae569c417.jsonl:363` (review renderer patch batch). September recreation reports were leads rather than sole evidence.

## Required files and boundaries

The companion JSON names 1,600 concrete selected config/result/status/interaction paths plus inspected launch, current import, config, analysis and provenance files. Current runtime imports include experiment, phase handlers, prompt generator, agent factory, model configs, utility/file manager, provider clients/key rotation/context compaction/JSON repair, and Game 3 environment. The package imports Game 1 and Game 2 environments too, so their files are import dependencies even for this Game 3 workflow. Model Elo lookup reads the dated markdown under docs/guides. A current source import is not proof that the same exact bytes ran historically.

The common launcher can use `.env` and an external credential file and needs the shared `/home/jz4391/openrouter_proxy` service. Credentials were not opened. Provider route defaults and retry attempts must remain auditable. Retain selected run logs and old attempts pending a complete per-attempt inventory; this report does not falsely certify their absence as irrelevance.

## Limits and removal decisions

- No file is certified removable by this result audit. Lack of use in Game 3 is not evidence of global irrelevance.
- Current generator has roster/sampling changes relative to April manifest, and no immutable original execution commit was found in inspected metadata. Preserve saved resolved configurations; do not regenerate the batch from current defaults and assume equivalence.
- September audit reports 46 Game 3 transcripts with proposal-recovery markers. Structured vote integrity passing does not establish absence of all proposal fallback behavior. Failed attempts and recovery logs remain provenance, not garbage.
- Reported slope counts in prior independent figure audit are 57/65 positive overall and six negative Game 3 fits; these counts were not recalculated in this audit, whereas the 400 raw gap calculations were.
- Numerical reproduction from saved utilities is not independent provider verification or full utility-from-proposal recomputation. Exhaustive runtime dependency closure and safe deletion precision cannot be guaranteed from static inspection alone.
- No code, paper, environment, Git state or data was changed. Only these audit reports were added.
