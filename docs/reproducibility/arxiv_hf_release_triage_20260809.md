# arXiv + Hugging Face Release Triage

Date: 2026-08-09
Paper of record: `overleaf/icml_aiwild_template/` (compiled `icml_aiwild_2026.pdf`, 2026-08-09 07:38)

This file updates `experiment_data_retention_audit.md` (2026-07-20) with everything that
landed after that audit: the 9-seed TTC replication, the GPT-5.4 coordinated-team batch,
the Codex qualitative adjudication runs, and the two rebuttal ablations.

Working tree is about **130 GB**. About **100 GB of that is deletable today** with no
effect on any paper figure, table, or number.

---

## Tier 0 — Critical. Never delete. Ships with arXiv.

### Engine (the code that produced every run)

| Path | Note |
| --- | --- |
| `run_strong_models_experiment.py` | Single entry point for all batches |
| `strong_models_experiment/` | Orchestrator + phase handlers |
| `game_environments/` | Games 1/2/3 + metrics |
| `negotiation/` | LLM agents, OpenRouter client, context compaction |
| `tests/` (40 files) | Behavioral tests for the engine |
| `requirements.txt`, `pytest.ini`, `LICENSE`, `README.md`, `CLAUDE.md` | Release metadata |

### Config generators (the definition of each grid)

`scripts/generate_configs_both_orders.sh` (G1 bilateral) ·
`scripts/generate_diplomacy_configs.sh` (G2) ·
`scripts/generate_cofunding_configs.sh` (G3) ·
`scripts/generate_appendix_llama33_baseline_configs.py` ·
`scripts/full_games123_multiagent_batch.py` (homogeneous + heterogeneous + monoculture) ·
`scripts/generate_ttc_native_scaling_jobs.py` ·
`scripts/generate_ttc_seed_replication_jobs.py` (new — the 9 seeds) ·
`scripts/generate_game1_gpt54_team_coordination.py` + `scripts/lock_game1_team_preferences.py` (new) ·
`scripts/generate_discount_factor_ablation.py` (new) ·
`scripts/context_compaction_pilot.py` (new)

### Figure + table pipeline

`scripts/paper_figures/` — 34 of the 39 plot scripts are live; `verify_all.py` is the
regression harness. Top-level analysis scripts that build the intermediate tables the
plotters read: `analyze_n2_baseline_comparison.py`, `analyze_n2_plus_multiagent_comparison.py`,
`analyze_neurips_revision_stats.py`, `analyze_nash_lindahl_fairness.py`,
`analyze_appendix_llama33_baseline_500.py`, `plot_gpt5_nano_baseline_vs_elo_all_games.py`,
`plot_exploitation_vs_elo.py`, `plot_full_games123_clean_subset.py`,
`plot_game3_utility_vs_elo.py`, `export_game2_batch_pngs.py`,
`generate_all_prompts_reference.py`, plus the new `analyze_ttc_ten_seeds.py`,
`analyze_ttc_hidden_reasoning_tokens.py`, `analyze_game1_gpt54_team_coordination.py`,
`analyze_game1_team_ceiling_normalization.py`, `analyze_discount_factor_ablation.py`,
`analyze_context_compaction_pilot.py`.

### Provenance manifests

All of `docs/reproducibility/`. `paper_experiment_data_manifest.csv` (4 MB) is the
run-level index that selects the exact config/result/rollout triples out of the raw
roots — it is what makes the HF dataset buildable. `scripts/build_paper_experiment_data_manifest.py --check`
validates it.

### Derived tables the plotters read directly

`experiments/results/n2_baseline_comparison_analysis_20260505` ·
`n2_plus_multiagent_comparison_analysis_20260505` ·
`n2_ttc_multiagent_comparison_analysis_20260505` ·
`appendix_llama33_baseline_analysis_20260503` ·
`figure_iteration_20260507` · `figure_iteration_20260626` · `figure_iteration_20260802` ·
`analysis/nash_lindahl_fairness_20260505` · `analysis/neurips_revision_20260504` ·
`analysis/ttc_group_intensity_turn_dedup_verification_20260701`

### Paper source

`overleaf/icml_aiwild_template/` — the 7 `.tex` files, `refs_joie.bib`, the ICML style
files, and the **34 graphics actually referenced by `\includegraphics`**.

### Reproduction evidence

`reproduction_audit/` (79 MB) — an independent per-figure lineage audit for all 23
appendix figures with recreated assets and pixel-diff verdicts. This is the strongest
reproducibility artifact in the repo; keep it and consider linking it from the README.

---

## Tier 1 — Raw data. The Hugging Face payload.

### Paper cohort (5,691 runs, already validated)

| Root | Runs | Size |
| --- | ---: | ---: |
| `scaling_experiment_20260404_064451` | 840 | 2.2 GB |
| `diplomacy_20260405_082215` | 540 | 1.8 GB |
| `cofunding_20260405_083548` | 540 | 4.4 GB |
| `appendix_llama33_baseline_game{1,2,3}_202605` | 140/180/180 | 186 MB |
| `full_games123_multiagent_production_20260428_085255` | 1,430 | 3.6 GB |
| `full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848` | 1,300 | 4.5 GB |
| `full_games123_random_monoculture_control_20260628_014357` | **325** | 1.1 GB |
| `ttc_native_scaling_20260502_212943` | 216 | 107 MB |

Do not upload the roots wholesale — they carry retry logs and archive copies. Filter
through `paper_experiment_data_manifest.csv` (≈ 442 MiB results + 3,819 MiB rollouts).

### Post-audit additions (verified complete today)

| Root | Runs found | Size | Status |
| --- | ---: | ---: | --- |
| `ttc_native_scaling_seed984_20260725_025700` | 216 | 98 MB | complete |
| `ttc_native_scaling_seed526_20260725_181400` | 216 | 98 MB | complete |
| `ttc_native_scaling_seed423_20260725_211500` | 216 | 93 MB | complete |
| `ttc_native_scaling_seed1024_20260725_211500` | 216 | 117 MB | complete |
| `ttc_native_scaling_seed128_20260727_043613` | 216 | 126 MB | complete |
| `ttc_native_scaling_seed256_20260727_043613` | 216 | 127 MB | complete |
| `ttc_native_scaling_seed2048_20260727_043613` | 216 | 128 MB | complete |
| `ttc_native_scaling_seed4096_20260727_043613` | 216 | 141 MB | complete |
| `ttc_native_scaling_seed612_20260727_043613` | **215** | 107 MB | **one run short** |
| `game1_gpt54_team_coordination_20260809_055844` | 101 result files | 246 MB | needs a 100-run reconcile |
| `discount_factor_ablation_game1_20260725` | — | 86 MB | rebuttal ablation |
| `discount_factor_ablation_game1_gamma_0p5_20260725` | — | 42 MB | rebuttal ablation |
| `context_compaction_pilot_20260726_210417` | — | 237 MB | rebuttal |
| `context_compaction_followup_20260727_050618` | — | 264 MB | rebuttal |

Qualitative annotation outputs to ship alongside:
`analysis/ttc_claude_seed_qualitative_adjudication_20260728` (64 MB) and
`analysis/ttc_claude_nine_seed_codex_adjudication_20260728` (50 MB), plus the
`analysis/llm_strategic_tag_*` and `analysis/ttc_*20260629` families listed in the
July audit.

**Two things to settle before upload.** `seed612` has 215 of 216 runs — either rerun the
missing config or document the exclusion. `game1_gpt54_team_coordination` has 101 result
files against a stated 100 runs; one is likely a retry duplicate and needs the same
dedup treatment the monoculture batch got.

---

## Tier 2 — Delete now. ~100 GB, zero paper impact.

### 1. `experiments/results/game1_multiagent_full_20260413_045538` — **97 GB**

This alone is 80% of the repo. It is the pre-production April Game-1 multi-agent design
with different group sizes, explicitly outside the 5,691-run paper cohort. I grepped the
whole tree: **no `.tex` file and no `.py` file references it.** The only mentions are four
prose documents (`experiment_data_retention_audit.md`, `cleanup_and_release_plan_2026_06.md`,
`experiment_result_root_classification.csv`, `docs/analysis/QUALITATIVE_FINDINGS_ELO_MECHANISMS_2026_04_16.md`).

The July audit put it on hold because `analysis/game1_position_qualitative` (237 MB)
derives from it. That derived directory is also uncited by the paper. Delete the derived
directory alongside the raw root, or keep the 237 MB derived summary and drop the 97 GB
of raw rollouts behind it — the second option is what I'd do.

### 2. `experiments/results/TO_DELETE/` — **3.1 GB**

Already approved and staged on 2026-07-19: the obsolete `full_games123_multiagent_20260427_040554`
root, superseded reference slates, cluster fallback batches, and partial plot-report assets.
See its own `README.md`. This has been sitting for three weeks; nothing has needed it.

### 3. Aborted TTC batch `*_20260727_021100` — 22 MB, 5 roots

`ttc_native_scaling_seed{128,256,612,2048,4096}_20260727_021100` hold 12–15 completed runs
each against the 216 expected. They were relaunched 1.5 hours later as `_20260727_043613`,
which are complete. The partial roots are pure noise and will confuse anyone reading the
seed inventory.

### 4. Unused paper graphics — 48 MB, 140 files

`overleaf/icml_aiwild_template/graphics/` holds 174 images; only 34 are referenced by
`\includegraphics`. The dead weight concentrates in `n_gt_2_report/` (38 files),
`n2_gpt5_nano/` (20), `qualitative_ttc/` (16), `n2_llama33/` (12). Three whole
directories are entirely unused: `n_gt_2_game1_multiagent/`, `n_gt_2_multiagent/`,
`appendix_multiagent/`.

Prune these before the arXiv upload — arXiv rejects unreferenced bulk anyway.

### 5. Build and cache debris

- 98 `__pycache__/` directories outside `.venv` — 9.7 MB
- `.pytest_cache/` — 60 KB
- `slurm/` — **12,375** job log files, 6.9 MB (gitignored; nothing reads them)
- `logs/cluster_pre_feb_23/` — 19 MB, predates every paper experiment
- `logs/openrouter_proxy_monitor.log` — 6.5 MB
- `overleaf/icml_aiwild_template/`: `check_icml_dims.{aux,log,out,pdf}`, `blah.tex`,
  `example_paper.tex`, `example_paper.bib`, `.aux/.log/.blg/.out`
- root: `texput.log`, `__pycache__/`

### 6. Loose root files

- `26631_Scaling_Laws_for_Strateg (3).pdf` — **25 MB**, a downloaded submission PDF
- `NeurIPS 2026 Rebuttals - ....pdf` and `... (1).pdf` — 240 KB, superseded by the tex
- `README-old.md` — 23 KB, replaced 2026-07-29
- `bargain/` — a stray nested directory containing only an empty `models/Phi-3-mini-128k-instruct`
  and a `.claude/logs` stub. Phi-3 was dropped from the paper entirely.
- `strategic_tag_review_final.json` — 19 KB, belongs under `analysis/` if still live

### 7. Dead script

`scripts/plot_scaling_utility_vs_elo.py` reads `experiments/results/scaling_experiment_20260403_051515`,
which does not exist. It cannot run. Nothing else references it.

---

## Tier 3 — Your call. I did not classify these.

| Path | Size | Question |
| --- | ---: | --- |
| `overleaf/neurips/`, `NExT_Game_2026_style{,_new}/`, `2026_iclr/`, `2026_icml/`, `2025_neurips/` | **1.7 GB** | Superseded paper generations. Irrelevant to arXiv, but this is your writing history. Move outside the repo rather than delete? |
| `.venv/` | **7.6 GB** | Gitignored. Exclude from the arXiv tarball; keep locally. |
| `paper_revision_proposals/` | 5.6 MB | 24 `issue_*` dirs + a `PROPOSED_PAPER.patch`. Are the patches all landed? |
| `paper_correctness_audit/`, `overleaf/analysis/*_20260809` | 154 MB | Very recent (Aug 8–9). Still active? |
| `readable_papers/` | 40 MB | Related-work PDFs. Not yours to redistribute — must not go to arXiv or HF. |
| `recreated_poster/` | 16 MB | Workshop poster, last touched Jul 18. |
| `subagent_outputs/` | 7.6 MB | Last touched Jun 29. Superseded by `analysis/` dirs? |
| `legacy/`, `ui/`, `notebooks/`, `visualization/`, `external_codebases/` | 920 KB | Small. Do any support a paper claim, or is this all dev scaffolding? |
| `experiments/results/excluded_from_paper_20260720/` (104 MB), `superseded_invalid_20260720/` (100 MB), `reruns_20260720/` (8 MB) | 212 MB | Deliberate provenance archives (the Phi exclusion, the `config_0412` rerun). Small and defensible — I'd keep them and cite them in the dataset card. |

---

## Gaps I found while checking

**The figure manifest has drifted from the compiled paper.** `icml_figure_manifest.csv`
lists 38 graphic paths; the current `.tex` files reference 34, and several have been
renamed or recombined since the manifest was written:

| Manifest says | Paper now uses |
| --- | --- |
| `appendix/bilateral_order_diagnostics_4x3.png` | `bilateral_order_diagnostics_gpt5_nano_order_1x3.png` |
| `multiagent_gini/heterogeneous_vs_homogeneous_gini_bars.png` (appendix) | `heterogeneous_vs_homogeneous_payoff_std.png` |
| `07_rounds_to_consensus_overall.png` + `08_..._by_competition.png` | merged `07_08_rounds_to_consensus_combined.png` |
| `hom_adversary_dilution_advantage_vs_n.png` | `..._vs_n_top_only_clean.png` |
| `homogeneous_adversary_gini_and_role_payoff.png` (appendix) | `homogeneous_adversary_payoff_std_and_role_payoff.png` |
| `multiagent_fairness_distance_vs_n.png`, `..._social_welfare_efficiency_vs_n.png`, `..._utility_gini_vs_n.png` | merged `multiagent_fairness_inequality_efficiency_3x3.png` |
| `qualitative_n2/n2_qualitative_combined_smooth5_outlier_removed_aligned.png` | `n2_qualitative_combined_category_matched_colors.png` |
| `11_fairness_distance_three_game_curves.png` | `..._smaller_30pct.png` |

Regenerate the manifest against the compiled PDF before you freeze the release, or the
provenance chain will point at the wrong assets.

**The paper text does not yet describe four of the batches you want to release.**
Grepping the current `.tex`: the 325-run monoculture control is described (`4_analysis.tex:88`),
but there is no mention of the 9-seed TTC replication (only the original 216-run sweep),
the GPT-5.4 coordinated team, the discount-factor ablation, or context compaction. If
those are meant to be in the camera-ready, the text needs to catch up; if they are
NeurIPS-rebuttal-only, the HF dataset card should say so explicitly so the run counts
reconcile against the paper.
