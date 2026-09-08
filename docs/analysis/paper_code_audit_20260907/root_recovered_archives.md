# Recovered replacement-run records

- The first experiment audits could not find some replacement control directories at their original paths.
- The root audit found their moved copies inside an earlier cleanup archive.
- These records must remain available to explain selected primary Game 2 and Game 3 results.
- The companion JSON lists 27 concrete files, not the entire archive.

## What was checked

- Parsed both manifests and all 17 saved configurations.
- Verified that the configuration IDs match the manifest selections.
- Verified that each saved output directory exists.
- Read both worker scripts and checked their calls to `run_strong_models_experiment.py`.
- Listed the associated Slurm submission scripts as historical launch records.
- Did not execute the scripts, which contain paths from before the archive move.

## Recovered locations

- Game 3: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/TO_DELETE_20260809/experiments_results/TO_DELETE_20260719/cofunding_20260405_083548_cluster_backfill_pli_20260409`.
  - The manifest selects Qwen configuration IDs 396, 397, 404, 408, 409, 412 and Llama IDs 522, 523, 528.
  - The configurations use in-place replacement output paths in the retained primary Game 3 root.
- Game 2: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/TO_DELETE_20260809/experiments_results/TO_DELETE_20260719/diplomacy_20260405_082215_llama32_1b_inplace_plicp_20260411`.
  - The manifest selects IDs 523, 525, 527, 529, 531, 535, 537, 539.
  - It records a Llama 3.2 1B local-GPU replacement on the `pli-cp` queue.

## How this changes the audit

- The missing-directory statements in E02 and E03 mean missing at the original location, not deleted from the whole workspace.
- The recovered records narrow those uncertainties.
- They do not prove which source revision ran on every attempt.
- A directory name such as `TO_DELETE` is not evidence that its contents are unnecessary.
- No files were deleted or moved.

## History checked

- Used the bundled codex-search script for prior cleanup history.
- Manually read session `019f7c50-0d37-7273-ba6e-e8da81d9f1ca`, lines 1962, 2167 and 3218, in `/home/jz4391/.codex/sessions/2026/07/19/rollout-2026-07-19T17-37-28-019f7c50-0d37-7273-ba6e-e8da81d9f1ca.jsonl`.
- The old discussion used earlier paper cohorts and retention decisions.
- The current archive checks, not those old decisions, support the 27-file keep list.
