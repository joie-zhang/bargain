# E25 per-tag statistics and codebook audit

## Result

- Current appendix labels are `tab:appendix_tag_level`, `tab:appendix_codebook`, and `sec:appendix_bilateral_capability_mechanisms`.
- The codebook contains 23 selected tags in six categories.
- The statistics table contains 23 Elo correlations, 23 intensity slopes, and 20 payoff correlations, for 66 printed numbers.
- All 66 numbers match a dated **provisional, pre-repair** analysis at two-decimal precision.
- No files are certified for deletion in this result audit.

## Exact observed checks

- I imported the exporter and called its read-only statistic functions, without running its writing main function.
- The selected manifest has 1,500 adversary records and 30 models.
- All 1,500 raw result paths exist.
- The selected original annotation records contain 4,805 tag events before turn deduplication and 4,777 afterward.
- The recomputed 23 rows match the saved provisional CSV to a maximum absolute difference of 9.8e-17.
- The same numbers match all 66 printed numeric cells in the current ICLR appendix.
- The diagnostic summary names 15 unannotated replacements, consisting of 14 Game 1 Nano-high runs and one Game 3 Qwen run.
- The old payoff sidecar supplies 1,486 primary utilities, while the rule-normalized input supplies all 1,500.
- Later Figure 4 repair files contain 776 annotations for those 15 transcripts, but the per-tag producer does not load that overlay.

## Verified production chain

1. The original 50-label scaffold is defined by `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_strategic_qualitative_tags.py`.
2. It uses the separate 2,730-run qualitative corpus and exports the source codebook.
3. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/prepare_llm_strategic_tag_adjudication_n2_gpt5.py` creates N2 manifests and judge instructions from the baseline analysis table and codebook.
4. Historical Codex workers write semantic annotation chunks, which the validator aggregates.
5. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/export_n2_qualitative_dedup.py` defines the selected 23 codes and statistic calculations.
6. `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/analysis/bilateral_1500_migration_20260809/workstreams/nonfigure_analysis_map/compute_primary_qualitative_diagnostics.py` imports that exporter, restricts the manifest to the primary 1,500 runs, and supplies canonical zero-on-no-agreement payoffs.
7. Its `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/analysis/bilateral_1500_migration_20260809/workstreams/nonfigure_analysis_map/primary_1500_tag_mechanism_rule_normalized_provisional.csv` is the exact numerical source for the current table.
8. The current table is manually entered in `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex`, rather than included from a generated TeX fragment.

## Statistical definitions checked

- Deduplication uses speaker identity, round, discussion turn, phase, and tag code.
- Each model's intensity is total deduplicated events divided by its 50 primary runs.
- Missing selected-tag records are filled with zero counts.
- Elo correlation is Spearman correlation across 30 model means.
- Slope is unweighted linear regression of intensity on Elo, multiplied by 100.
- Payoff correlation is Spearman correlation between model mean intensity and model mean utility.
- The exporter calculates payoff correlations for all 23 tags, while the paper manually blanks the three outcome-defined structural tags.
- The codebook is not an additional 23 experiments, and the 66 coefficients share one data-generation and aggregation chain.

## Historical evidence

- I read the codex-search skill and ran its bundled repo-linked search.
- The broad query returned current-session forks, which are not original experiment evidence.
- I then searched the exact producer name in older local session logs and manually inspected the original migration parent.
- `/home/jz4391/.codex/sessions/2026/08/09/rollout-2026-08-09T06-18-15-019fe607-c21a-7662-bdf6-4723be814ec6.jsonl:632` records the completed nonfigure diagnostic report.
- Lines 779–780 of that session inventory the generated migration artifacts.
- `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/analysis/minimal_fix_review_20260809/issue_068/report.md` independently records the exact table/provisional-CSV match and explains the unmet annotation coverage gate.
- History access was limited to locally retained logs.

## Preservation implications

- The dated migration directory is needed even though its name makes it look temporary.
- The old 1,920-row annotation manifest is needed to explain the primary-cohort filter and original labeling assignments.
- The old payoff sidecar is needed to explain why normalization replaced 14 absent utilities.
- The missing-15 repair files are needed to explain why main Figure 4 and the per-tag table now use different annotation coverage.
- The original scaffold remains needed as codebook provenance, but its regex output is not the final semantic classifier.
- Raw experiment runtime and saved configurations overlap E01–E03 and must also be retained.
- E24 supplies the detailed worker/judge and main Figure 4 repair trace.

## Limits

- This audit verifies current numerical provenance, not the correctness of every semantic label.
- I did not rerun paid judges or negotiation experiments.
- I did not certify an exact historical runtime commit for each primary run.
- I found no safe removal candidate from this result alone.
- The adjacent JSON provides concrete needed files, reasons, evidence, and explicit directory-selection limits.

Only these two audit report files were authored; code, paper, configurations, and data were not changed.
