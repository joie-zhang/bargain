# KEEP / DELETE Manifest — Data Directories

Date: 2026-08-09
Companion to `arxiv_hf_release_triage_20260809.md`.

Every directory under `experiments/results/` and `analysis/` is classified below.
Nothing is missing from these tables — the KEEP list is complete, so the DELETE list
is exactly its complement. Deletion frees **~100 GB**.

Method: I resolved every `experiments/results/*` and `analysis/*` path string
referenced by (a) the paper `.tex` sources, (b) `scripts/paper_figures/*.py`,
(c) the top-level KEEP analysis scripts, and (d) `scripts/retained_analysis/`.
A directory is KEEP if some retained script or the paper reads it, or if it holds
raw runs in a paper or release cohort.

---

## Part 1 — `experiments/results/` KEEP (32 entries)

### Raw paper cohort (5,691 runs)

| Directory | Why |
| --- | --- |
| `scaling_experiment_20260404_064451` | G1 bilateral, 840 runs |
| `diplomacy_20260405_082215` | G2 bilateral, 540 runs |
| `cofunding_20260405_083548` | G3 bilateral, 540 runs |
| `appendix_llama33_baseline_game1_202605` | Llama baseline, 140 |
| `appendix_llama33_baseline_game2_202605` | Llama baseline, 180 |
| `appendix_llama33_baseline_game3_202605` | Llama baseline, 180 |
| `full_games123_multiagent_production_20260428_085255` | Homogeneous + nano control, 1,430 |
| `full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848` | Heterogeneous, 1,300 |
| `full_games123_random_monoculture_control_20260628_014357` | Monoculture, 325 |
| `ttc_native_scaling_20260502_212943` | TTC, 216 |

### Raw post-audit cohort (release, not yet in paper text)

| Directory | Why |
| --- | --- |
| `ttc_native_scaling_seed984_20260725_025700` | TTC seed replication |
| `ttc_native_scaling_seed526_20260725_181400` | TTC seed replication |
| `ttc_native_scaling_seed423_20260725_211500` | TTC seed replication |
| `ttc_native_scaling_seed1024_20260725_211500` | TTC seed replication |
| `ttc_native_scaling_seed128_20260727_043613` | TTC seed replication |
| `ttc_native_scaling_seed256_20260727_043613` | TTC seed replication |
| `ttc_native_scaling_seed612_20260727_043613` | TTC seed replication (backfill in flight) |
| `ttc_native_scaling_seed2048_20260727_043613` | TTC seed replication |
| `ttc_native_scaling_seed4096_20260727_043613` | TTC seed replication |
| `game1_gpt54_team_coordination_20260809_055844` | GPT-5.4 coordinated team, 100/100 valid |
| `discount_factor_ablation_game1_20260725` | Rebuttal ablation |
| `discount_factor_ablation_game1_gamma_0p5_20260725` | Rebuttal ablation |
| `context_compaction_pilot_20260726_210417` | Rebuttal |
| `context_compaction_followup_20260727_050618` | Rebuttal |

### Derived tables read by retained scripts

| Directory | Read by |
| --- | --- |
| `n2_baseline_comparison_analysis_20260505` | 24 references across `paper_figures/` |
| `n2_plus_multiagent_comparison_analysis_20260505` | 22 references across `paper_figures/` |
| `n2_ttc_multiagent_comparison_analysis_20260505` | TTC comparison figures |
| `appendix_llama33_baseline_analysis_20260503` | `analyze_appendix_llama33_baseline_500.py` |
| `figure_iteration_20260507` | 3 figure scripts |
| `figure_iteration_20260626` | 2 figure scripts |
| `figure_iteration_20260802` | 2 figure scripts |
| `capability_payoff_scaling_20260505` | `retained_analysis/analyze_capability_payoff_scaling_20260505.py` |

### Provenance archives — small, and they are the audit trail

| Directory | Why |
| --- | --- |
| `excluded_from_paper_20260720` | The 28 Phi-3 exclusions. Cite in the dataset card. |
| `superseded_invalid_20260720` | `config_0412` and `config_0262` pre-repair states |
| `superseded_invalid_20260809` | **New today** — `config_0142` seed612 partial |
| `reruns_20260720` | July rerun staging |
| `provider_failure_reports` | 8.5 KB, cross-batch provider error record |

### Loose files at the root of `experiments/results/` — keep

`multiagent_experiment_completion_report_20260504.md` (the 2,730/2,730 report),
`provider_failures.{jsonl,md,lock}`.

### Symlinks — keep, they cost nothing

`cofunding_latest`, `diplomacy_latest`, `scaling_experiment` all point at KEEP roots.
`game1_multiagent_full_latest` points at a partially-deleted root; see below.

---

## Part 2 — `analysis/` KEEP (24 entries)

### Read directly by retained scripts

`full_games123_all_success_preliminary_20260428` — **despite the name, this is live.**
Read by `analyze_neurips_revision_stats.py` and `plot_full_games123_clean_subset.py`.
Do not delete on the strength of the word "preliminary".

`full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_20260503_rerun` — read by `analyze_neurips_revision_stats.py`
`nash_lindahl_fairness_20260505` · `neurips_revision_20260504` ·
`ttc_group_intensity_turn_dedup_verification_20260701` ·
`llm_strategic_tag_adjudication_n2_gpt5_20260629` ·
`llm_strategic_tag_elo_exploration_n2_gpt5_intensity_20260629` ·
`recreated_figures` (5 references) · `paper_figure_verification` (read by `verify_all.py`)

### Qualitative cohort — the release payload for the annotation work

`llm_strategic_tag_adjudication_20260628` (538 MB) ·
`llm_strategic_tag_adjudication_random_monoculture_20260629` ·
`llm_strategic_tag_elo_exploration_20260629` ·
`llm_strategic_tag_elo_exploration_competition_20260629` ·
`llm_strategic_tag_elo_exploration_hot_20260629` ·
`llm_strategic_tag_elo_exploration_intensity_20260629` ·
`llm_strategic_tag_elo_exploration_lines_payoff_20260629` ·
`llm_strategic_tag_elo_exploration_random_monoculture_intensity_20260629` ·
`strategic_qualitative_tags_20260628` · `qualitative_rollout_dynamics_20260628` ·
`qualitative_dynamics_trends_20260628` · `homogeneous_adversary_redline_elo_20260628` ·
`homogeneous_adversary_tag_mechanism_20260629` · `ttc_hot_strategic_tags_20260629` ·
`ttc_llm_strategic_tag_adjudication_20260629` ·
`ttc_strategic_tag_exploration_lines_payoff_20260629` ·
`ttc_strategic_tag_intensity_lines_payoff_20260629` ·
`ttc_claude_seed_qualitative_adjudication_20260728` ·
`ttc_claude_nine_seed_codex_adjudication_20260728`

### Recent rebuttal / paper work

`reviewer_item8_n_slope_20260725` · `figure7_endpoint_fairness_comparison_20260729` ·
`icml_aiwild_report_20260808`

### Loose files

`__init__.py`, `multiagent_rating.py`, `ttc_qualitative_synthesis_20260628.md`

---

## Part 3 — DELETE. Everything not listed above.

| # | Path | Size | Why it is safe |
| --: | --- | ---: | --- |
| 1 | `experiments/results/game1_multiagent_full_20260413_045538/proposal1_invasion` | **97 GB** | Superseded April "invasion" design, 752 runs, zero code/tex references |
| 2 | `experiments/results/game1_multiagent_full_20260413_045538/logs` | 61 MB | Job logs for the above |
| 3 | `experiments/results/TO_DELETE` | 3.1 GB | Approved and staged 2026-07-19, untouched since |
| 4 | `analysis/game1_position_qualitative` | 237 MB | Only consumer of the April root; uncited by the paper |
| 5 | `analysis/full_games123_heterogeneous_..._gini_preliminary` | 42 MB | Superseded by `_20260503_rerun`, 0 references |
| 6 | `analysis/full_games123_production_20260428_085255_plots_20260429` | 24 MB | Superseded by `_20260503_rerun`, 0 references |
| 7 | `analysis/full_games123_production_20260428_085255_plots_20260503_rerun` | 27 MB | 0 references — **verify before deleting**, see caveat |
| 8 | `analysis/json_parse_errors_20260502` | 3.3 MB | One-off debugging record, 0 references |
| 9 | `analysis/elo_variance_sampling_100k_context` | 2.8 MB | Pre-production sampling probe, 0 references |
| 10 | `analysis/game1_qualitative_review` | 3.1 MB | Superseded by the June qualitative cohort |
| 11 | `experiments/results/ttc_native_scaling_seed{128,256,612,2048,4096}_20260727_021100` | 22 MB | Aborted batch, 12–15 of 216 runs each; relaunched 1.5 h later as `_043613` |
| 12 | `experiments/results/ttc_reasoning_capture_derisk_seed128_20260727_042424` | 901 KB | De-risk probe, 0 references |
| 13 | `slurm/` | 6.9 MB | 12,375 loose job logs, gitignored |
| 14 | `logs/cluster_pre_feb_23` | 19 MB | Predates every paper experiment |
| 15 | `readable_papers/` | 40 MB | Third-party PDFs — must not ship to arXiv or HF anyway |
| 16 | `recreated_poster/` | 16 MB | Workshop poster, superseded |
| 17 | `subagent_outputs/` | 7.6 MB | Superseded by the `analysis/` dirs |
| 18 | `bargain/` | 2.5 MB | Stray nested dir; only an empty Phi-3 model folder. Phi-3 was cut from the paper. |
| 19 | `legacy/` | 490 KB | Pre-refactor code |
| | **Total** | **~100 GB** | |

**Caveat on #7.** `full_games123_production_20260428_085255_plots_20260503_rerun` has no
code reference, but it is the rerun that supersedes #6, and reruns are usually the live
copy. I could not find a consumer. Stage it, do not delete it, until you have rebuilt the
multi-agent figures once from a clean checkout.

**Note on #1.** Delete only `proposal1_invasion/` and `logs/`, not the whole root. Keeping
`configs/`, `analysis/`, `analysis_extended/`, `status/`, and `manifest.json` (~26 MB)
preserves what the April exploration actually concluded, at 0.03% of the storage. The
97 GB is pathological: 5,660 per-agent interaction JSONs averaging 17 MB each, because
that design replayed full conversation history per agent per turn. After this, update or
drop the `game1_multiagent_full_latest` symlink.

### Also delete (non-directory debris)

98 `__pycache__/` dirs outside `.venv` (9.7 MB) · `.pytest_cache/` ·
`logs/openrouter_proxy_monitor.log` (6.5 MB) · root `texput.log` · `README-old.md` ·
`26631_Scaling_Laws_for_Strateg (3).pdf` (25 MB) · the two `NeurIPS 2026 Rebuttals*.pdf` ·
`strategic_tag_review_final.json` (move to `analysis/` if still live) ·
`scripts/plot_scaling_utility_vs_elo.py` (reads a root that does not exist) ·
the 140 unreferenced files in `overleaf/icml_aiwild_template/graphics/` (48 MB) ·
LaTeX build debris in the template dir.

---

## Part 4 — Staging commands

These **move** into `TO_DELETE_20260809/`. Nothing is destroyed; you run the final
`rm -rf` yourself once you are satisfied.

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
S=experiments/results/TO_DELETE_20260809
mkdir -p $S/{experiments_results,analysis,repo_root,logs}
```

```bash
# 1-2. The 97 GB April root: move the bulk, keep the derived summaries in place
mv experiments/results/game1_multiagent_full_20260413_045538/proposal1_invasion \
   $S/experiments_results/game1_multiagent_full_20260413_proposal1_invasion
mv experiments/results/game1_multiagent_full_20260413_045538/logs \
   $S/experiments_results/game1_multiagent_full_20260413_logs
```

```bash
# 3. Previously approved staging area, folded into this one
mv experiments/results/TO_DELETE $S/experiments_results/TO_DELETE_20260719
```

```bash
# 4-10. Superseded analysis directories
mv analysis/game1_position_qualitative \
   analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary \
   analysis/full_games123_production_20260428_085255_plots_20260429 \
   analysis/json_parse_errors_20260502 \
   analysis/elo_variance_sampling_100k_context \
   analysis/game1_qualitative_review \
   $S/analysis/
```

```bash
# 11-12. Aborted TTC batch and the de-risk probe
mv experiments/results/ttc_native_scaling_seed128_20260727_021100 \
   experiments/results/ttc_native_scaling_seed256_20260727_021100 \
   experiments/results/ttc_native_scaling_seed612_20260727_021100 \
   experiments/results/ttc_native_scaling_seed2048_20260727_021100 \
   experiments/results/ttc_native_scaling_seed4096_20260727_021100 \
   experiments/results/ttc_reasoning_capture_derisk_seed128_20260727_042424 \
   $S/experiments_results/
```

```bash
# 13-14. Logs
mv slurm $S/logs/slurm_loose_job_logs
mv logs/cluster_pre_feb_23 $S/logs/
mv logs/openrouter_proxy_monitor.log $S/logs/
```

```bash
# 15-19. Repo-root directories
mv readable_papers recreated_poster subagent_outputs bargain legacy $S/repo_root/
```

```bash
# Loose files
mv README-old.md texput.log strategic_tag_review_final.json \
   "26631_Scaling_Laws_for_Strateg (3).pdf" \
   "NeurIPS 2026 Rebuttals - Scaling Laws for Strategic Interactions.pdf" \
   "NeurIPS 2026 Rebuttals - Scaling Laws for Strategic Interactions (1).pdf" \
   scripts/plot_scaling_utility_vs_elo.py \
   $S/repo_root/
```

```bash
# Caches — these regenerate, no staging needed
find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
rm -rf .pytest_cache
```

```bash
# Stage #7 separately — the one I could not fully clear
mkdir -p $S/verify_before_deleting
mv analysis/full_games123_production_20260428_085255_plots_20260503_rerun \
   $S/verify_before_deleting/
```

### Verify before you delete

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain && source .venv/bin/activate && python scripts/build_paper_experiment_data_manifest.py --check && python scripts/paper_figures/verify_all.py
```

If both pass with the staging directory in place, nothing retained depended on what
you moved. Then, and only then:

```bash
du -sh /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/TO_DELETE_20260809
```

```bash
rm -rf /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/TO_DELETE_20260809
```
