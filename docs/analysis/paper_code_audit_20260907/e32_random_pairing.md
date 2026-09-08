# E32: Fixed baseline versus heterogeneous two-player opponents

- Source is /scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:497-508.
- Scope covers all three game slopes, the 260-run comparison, model means, error bars, model pool and pairing procedure.
- No source, data or paper files changed; no deletion candidates established.

## Verified result

| Game | Fixed runs/models | Heterogeneous runs/models | Fixed slope/100 Elo | Heterogeneous slope/100 Elo |
| --- | --- | --- | --- | --- |
| 1 | 420/30 | 100/24 | 6.7532680262 | 7.4898533752 |
| 2 | 540/30 | 80/24 | 6.7586007082 | 4.7897057885 |
| 3 | 540/30 | 80/24 | 7.4005666676 | 5.5896685766 |

- Independently read all 260 selected heterogeneous terminal JSON files and checked 520 model identities/payoffs against the actual renderer input.
- Payoff comparison uses absolute tolerance 1e-12 for CSV float conversion.
- All selected configurations have n=2 and two distinct models.
- Each point averages model appearances; appearance counts range 3-17 in Game 1 and 2-12 in Games 2-3.
- The regression weights each model mean equally; it does not weight by appearance count.
- SEM uses sample standard deviation divided by square root of appearances.
- Failed agreements remain included: 2, 0 and 6 heterogeneous runs in Games 1-3.
- All six slopes were independently recalculated from current input CSVs.
- Current ICLR image and prior raw recreation both have SHA-256 457b6b1c238c23f276196ea46856392942d3a4c6b898c2af1003c2dcdcf90c15.

## Generation and execution

- /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:397-499 enumerates distinct subsets and divides population Elo standard deviation into five equal-width intervals.
- At n=2, these intervals stratify pairwise Elo separation.
- Lines 901-979 sample four pairs per interval per game-setting cell, with replacement across draws, then shuffle seat order.
- These are stratified random opponents, not unrestricted uniform random pairings from the whole pool.
- Master seed is 20260427; environment, roster draw and seat order each have separate stable seeds.
- An observed sample has environment seed 2106547656, draw seed 598402729 and order seed 1198855750.
- Game 1 uses five competition cells; Games 2 and 3 use four cells each, with 20 negotiations per cell.
- Game 2 has rho -1 or 0.9 and theta 0.2 or 0.8.
- Game 3 has sigma 0.2 or 0.5 and alpha 0.2 or 0.8.
- Fixed-baseline runs use a different roster, preference draws and game grids; slope differences do not isolate partner effects.
- The saved Slurm file calls the batch script run-one, which builds the command for /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py.
- Saved Slurm configuration routes OpenRouter through /home/jz4391/openrouter_proxy and sources an external key environment file without exposing its contents.
- Shared runtime imports are routed through StrongModelsExperiment; dedicated runtime agents cover deeper phase/provider dependencies.

## Analysis and historical evidence

- The renderer directly reads /scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig10_bilateral_adversary_payoff/generated/primary_run_lineage.csv and /scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig21_fixed_vs_random_pairings/heterogeneous_agent_rows_from_raw.csv.
- /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_20/reproduce.py reconstructs those inputs from raw results and the historical heterogeneous parser at Git commit 4ef9d4ffee0ea16d23ef757f62a1ca1f33b2bc9e.
- Its concrete raw/config hash manifest must remain available; fixed raw cohort validation is handled by E01-E03.
- The archived parser snapshot is retained because current exact reproduction depends on its historical behavior.
- Codex-search ran against local repo-linked histories; top matches repeated the current conversation and were not accepted as original evidence.
- Manually inspected /home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T01-38-21-01a00914-034d-7071-b138-37d972ac2a87.jsonl:258-259, which records the August 16 switch from 1,941 historical fixed runs to the 1,500 primary cohort and updates the source.
- The prior May-origin history is only a lead here, not independently established original launch proof.

## Cleanup implications and limits

- The JSON report lists 539 existing concrete paths, including 260 selected result files and 260 corresponding launch configurations.
- Raw result contents were checked; saved external config existence was checked, while the actual embedded result configuration supplied model and seed verification.
- The historical source hash manifest supplies the larger shared fixed-cohort selection; it does not make all files in an experiment root needed by this figure.
- Current CSV diagnostics report 19 synthetic proposal markers; preserve selected interaction files, config logs and statuses for the independent integrity review.
- Do not delete reproduction_audit inputs or the dated recreation parser merely because their names refer to older figure numbers.
- Default renderer output still names ICML; current ICLR asset is correct by hash, but the default command does not directly update it.
- No file is safe to delete based on this result alone.
