# E24. Six behavior categories

## Result and scope

- Current paper source is `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/4_analysis.tex`, Figure `fig:n2_qualitative` and the mechanisms paragraph.
- Appendix source is `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:651`, including `tab:appendix_codebook`.
- The upper panel relates category event counts to adversary payoff across 1,500 runs.
- The lower panel relates category event frequency to model Elo across 30 models.
- E25 handles the separate per-tag table; E01–E03 handle the negotiation launch and runtime chain.

## Verified source chain

1. The historical scaffold script is `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_refined_qualitative_dynamics.py`.
   - Its two source roots are the production multi-agent batch and the heterogeneous repair batch.
   - Its selected coding CSV contains 2,730 rows, independently counted in this audit.
   - This is provenance for codebook development, not the current 1,500-run plotted denominator.
2. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_strategic_qualitative_tags.py` reads that coding CSV and its prior codebook and defines the 50-label scaffold.
3. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/prepare_llm_strategic_tag_adjudication_n2_gpt5.py` reads the all-runs comparison CSV and the 50-label CSV.
   - It selects the GPT-5-nano baseline and excludes Phi-3, yielding an asserted 1,920-run manifest.
   - It emits ten-transcript chunk manifests, a codebook, and exact semantic judge instructions.
   - It is preparation code, not an API-based annotation runner.
4. Original Codex worker launches supplied each chunk and the saved instructions to independent agents.
   - The inspected worker context records `gpt-5.5` and `high` effort.
   - The instruction requires semantic classification of all relevant messages and outcomes and explicitly forbids using regex as the classifier.
   - Workers save event JSONL and audit Markdown, including reviewed transcripts with no detected events.
5. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/validate_llm_strategic_tag_adjudication.py` validates and merges the worker files.
   - Original history executes its previous path with `--out-dir analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629 --write-aggregate --strict`.
   - The current saved aggregate contains 21,863 event rows.
6. The current primary cohort selects 1,500 negotiation results, of which 1,485 have original annotation rows.
   - The remaining 15 transcripts were separately annotated by `gpt-5.6-sol` with `high` effort.
   - The repair contributes 776 event rows and preserves a completion ledger, worker outputs, audits, and validation report.
7. The final cohort migration is `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/analysis/bilateral_1500_migration_20260809/workstreams/figure_04_qualitative/rebuild_figure4_1500.py`.
   - It reads the original events and the repair overlay and checks repair completion.
   - It selects the adversary agent and 23 retained tags, then deduplicates by rollout, speaker, round, discussion turn, phase, and category.
   - Zero-event reviewed runs remain in denominators.
8. The latest stacked renderer is `/scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260829/08_figure4/make_halfwidth_stacked_candidate.py`.
   - It dynamically imports `/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/icml_aiwild_current_20260829/fig04_qualitative/reproduce.py`.
   - It reads the candidate correlation and intensity CSV files in the same dated review directory.
   - It removes one model point per category using deviation from a centered five-model rolling median and then smooths the remaining points with a centered five-model rolling mean.
   - This means the dated review directory and reproduction-audit source contain actual dependencies, despite their names.

## Checks performed now

- Parsed the current primary cohort and independently counted 1,500 selected runs.
- Parsed the original annotation manifest and counted 1,920 rows.
- Parsed original and repair annotation files and counted 21,863 and 776 events.
- Applied the current 23-tag and adversary selection and counted 5,031 event rows.
- Independently constructed category-turn deduplication keys and counted 4,410 events.
- Counted 161 current cohort runs with no selected adversary event.
- Verified that all 206 worker-output files referenced by selected events exist.
- Checked all 15 repair raw-result hashes, output hashes, and audit hashes against the completion ledger.
- Inspected code paths and local imports without invoking write-producing analysis or plotting entry points.
- No paid model calls, annotations, code edits, or data edits were made.

## Original history verified

- The bundled codex-search script found session `019f0d3e-accb-7f62-b5a2-06d8b3f0b79c` as its highest match.
- `/home/jz4391/.codex/sessions/2026/06/28/rollout-2026-06-28T04-00-39-019f0d3e-accb-7f62-b5a2-06d8b3f0b79c.jsonl:447` contains original scaffold creation.
- The same session at line 5246 contains user approval for the annotation launch.
- The same session at line 5338 contains an actual semantic worker launch and exact required outputs.
- The same session at line 7286 contains the strict aggregate command.
- `/home/jz4391/.codex/sessions/2026/06/29/rollout-2026-06-29T05-37-39-019f12bd-d875-7193-8d8e-9101378ce90f.jsonl:5` records the inspected original worker model and effort.
- `/home/jz4391/.codex/sessions/2026/07/01/rollout-2026-07-01T02-32-17-019f1c60-dc4f-7830-97ef-5fe2932bf500.jsonl:594` records the user's reduction to 23 labels.
- `/home/jz4391/.codex/sessions/2026/07/01/rollout-2026-07-01T03-02-16-019f1c7c-4d8c-7661-86e0-efa40d82c543.jsonl:7831` records the user's request to remove one extreme outlier per curve.
- History coverage is limited to this profile's locally available sessions.

## Needed files and deletion decision

- The companion JSON lists 35 concrete paths and five directory entries with explicit selection rules.
- The primary raw results are exactly the selected `result_path` values in the primary cohort CSV; E01–E03 provide their launch/runtime dependencies.
- The historical 1,920 annotation manifest and 2,730-row scaffold retain additional source references that cannot be dropped merely because they are outside the current 1,500 plotted runs.
- Keep the original semantic worker responses and audits, including zero-event reviews, because they distinguish absent behavior from absent review.
- Keep repair data and its source hashes because regenerated model labels would not reproduce the observed annotation decisions.
- Keep the independent audit scripts and individual annotation-verification records because they document unresolved source issues.
- No file is proposed for deletion by this result audit.
- Directory entries in the JSON are selected subsets, not approval to retain or delete every other file in those trees.

## Limits and inconsistencies

- Current main text says all transcripts were reannotated with a 23-label codebook by `GPT-5.5-xhigh` judges.
   - The observed original worker uses the 50-label instructions with `gpt-5.5/high`, followed by a selected-label filter and the separate 15-transcript `gpt-5.6-sol/high` repair.
   - The appendix already describes the original high-effort annotation stage more accurately, but does not describe the repair model.
- The exact brainstorming worker effort and completeness of human/model reading over all 2,730 scaffold transcripts were not independently established here.
- The prior figure audit reports 516 mechanically unresolved quotes and 168 formal-outcome annotations requiring semantic review.
   - Those observations justify retaining the original logs, worker records, and verification records.
   - This task did not perform a new semantic judgment of every annotation.
- Original history calls `scripts/report_llm_strategic_tag_progress.py`, but an exact filename search did not find that source in the current tree.
- The older exporter at `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/export_n2_qualitative_dedup.py` is historical analysis code and is not interchangeable with the corrected migration producer.
   - E25 independently reports that the current per-tag table matches the original-event-only analysis and omits the 15-transcript repair used by Figure 4.
- Absolute paths and a dynamic import into a dated audit directory make relocation unsafe without separate testing.
- A complete semantic-validity guarantee or a codebase-wide deletion guarantee does not follow from these checks.
