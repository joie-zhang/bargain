# Bargain — Cleanup, Preservation & Release Plan (June 2026)

**Hard constraint:** Folder access on `/scratch/gpfs/DANQIC/jz4391/bargain` is lost in ~1 month (~early July 2026).
**Other deadlines:** NeurIPS rebuttals expected late July 2026; arXiv submission of the paper.
**Operating mode:** Parallel workstreams. Curated paper-final subset → HuggingFace. Claude traces & inventories; **user does all deletion manually.** Nothing destructive runs until keepers are verified off-cluster.

---

## Situation (from initial scan, 2026-06-03)

- **217G total**, of which **207G is `experiments/results/`** (166,388 files). Everything else is small.
- `.gitignore` already excludes `experiments/`, `analysis/`, `Figures/`, `exports/`, `overleaf`, `logs/`, `slurm` → git footprint is tiny. Only 11 uncommitted items (mostly stray screenshots + 2 plot scripts).
- Clutter in `results/` root: hundreds of `monitor_822_snapshot_openrouter_*.json` and duplicate `multiagent_experiment_completion_report_*.md`.

## Paper-final data dependency map (traced from figure/analysis code)

> **CAVEAT — verify on disk first.** Code references `scaling_experiment_20260404_064451` and `appendix_llama33_baseline_game1_202605`, which were below the initial `du` cutoff. Step A0 reconciles code-expected paths vs. actual directories before anything is uploaded or flagged.

### KEEP — referenced by paper code (candidate HF subset)

| Directory | Size | Feeds |
|---|---|---|
| `game1_multiagent_full_20260413_045538` | **97G** | Game 1 position/role effects, qualitative exemplars (dominates the subset) |
| `cofunding_20260405_083548` | 4.4G | Game 3 N=2 baseline / capability-payoff |
| `full_games123_multiagent_production_20260428_085255` | 3.6G | Homogeneous N>2 multiagent + fairness |
| `diplomacy_20260405_082215` | 1.8G | Game 2 N=2 baseline / capability-payoff |
| `full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848` | 1.7G | Heterogeneous N>2 multiagent (arena Elo) |
| `ttc_native_scaling_20260502_212943` | 107M | TTC effort-vs-payoff / compute plots |
| `scaling_experiment_20260404_064451` *(verify)* | ~147M | Game 1 N=2 baseline (canonical) |
| `appendix_llama33_baseline_game{1,2,3}_202605` | ~230M | Appendix Llama-3.3 baselines |

**Generated analysis outputs (keep, but reproducible):** `n2_baseline_comparison_analysis_20260505`, `n2_plus_multiagent_comparison_analysis_20260505`, `n2_ttc_multiagent_comparison_analysis_20260505` (final report bundle), `capability_payoff_scaling_20260505`.

→ Curated subset ≈ **~110G, dominated by the single 97G Game 1 multiagent run.** Decision point: does the full 97G need uploading, or a per-config subsample? (see A2.)

### STALE — no code reference (candidate to delete; user decides)

- **Biggest wins:** `cofunding_20260223_032239` (24G), `diplomacy_20260223_032204` (20G) → ~44G, unreferenced.
- Older dated dups: all `scaling_experiment_*` except 20260404; all `diplomacy_*` except 20260405; all `cofunding_*` except 20260405; all `ttc_scaling*` except `ttc_native_scaling_20260502`.
- Smoke/sample/derisk: `game1_multiagent_smoke_*`, `*_matrix_sample_*`, `game2_samples_*`, `game3_multiagent_sample_*`, `derisk_15_samples_*`, `context_json_repair_derisk_*`, `backfill_proposals`, `gpt-5-*-effort_vs_*` comparisons, `full_games123_multiagent_20260427` (pre-production).
- Root clutter: `monitor_822_snapshot_*.json`, duplicate completion reports.

---

## Workstreams (run in parallel)

### WS-A — Data preservation → HuggingFace  *(critical path; gated by deadline)*
- **A0. Reconcile keep-list vs disk.** Full `du` of every `results/` dir; confirm each KEEP dir exists; resolve the `20260404_064451` / `appendix_game1` discrepancy. Produce a machine-readable `keep_manifest.csv` (dir, size, status, feeds).
- **A1. User confirms** the keep-list (mix-of-both: Claude proposes, user corrects).
- **A2. Decide 97G handling** — upload full vs. subsample the Game 1 multiagent run.
- **A3. Create HF dataset repo(s)** + dataset card; structure by game / N / run.
- **A4. Upload curated subset** via `huggingface_hub` (resumable; chunk the 97G). Verify checksums/row counts after upload.
- **A5. Record HF URLs** in README + paper data-availability statement.

### WS-B — Codebase cleanup & runnability  *(for arXiv public release)*
- **B0. Git hygiene (quick win):** triage uncommitted items — remove stray screenshots and commit or relocate the plot scripts.
- **B1. Repro path:** confirm `requirements.txt`/env, `.env.example`, and a clean `run_strong_models_experiment.py` entrypoint work from scratch.
- **B2. Header/docs pass** on the canonical scripts (the analyze/plot/build scripts that generate paper figures) — what they read, what they produce.
- **B3. Prune dead/legacy code** (`legacy/`, stale scripts) — inventory only, user confirms.
- **B4. README:** quickstart, data-availability (HF links), figure-reproduction recipe.

### WS-C — arXiv paper prep
- **C0. Resolved:** `overleaf/` is canonical; the stale May 7 merge-prep snapshot was removed after its changes were verified against Git history and the retained stash.
- **C1. Figure-reproduction recipe:** map each paper figure → script → HF data (extends the trace above).
- **C2. arXiv packaging:** strip aux/build artifacts, ensure figures embed, anonymity/licensing check (`.gitmodules` points to an anonymized GovSim URL).

### WS-D — Stale inventory  *(feeds A & frees space; user deletes)*
- **D0.** Categorized inventory: KEEP / STALE / UNSURE with sizes and reasons (CSV + summary).
- **D1.** Generate a reviewable delete-candidate list + a ready-to-run (but not executed) removal script the user runs themselves.

### WS-E — NeurIPS rebuttal readiness
- Overlaps WS-B: keeping the experiment harness runnable = rebuttal-ready. Ensure API keys/config documented and a small smoke run works before access ends.

---

## Suggested first moves
1. **A0 + D0 together** — one full disk inventory pass produces both the verified keep-manifest and the stale inventory. (read-only, safe)
2. **B0** — clear the 11 uncommitted items (5-min quick win).
3. Then **A1** (user confirms keep-list) unblocks the upload, the real deadline item.

## Open decisions for user
- 97G Game 1 run: upload whole vs. subsample? (A2)
- HF layout: one combined dataset vs. per-game datasets? curated-only vs. curated + raw-archive?
