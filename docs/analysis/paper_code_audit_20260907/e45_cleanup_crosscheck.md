# E45 cleanup cross-check

## Conclusion

- No experiment source file is approved for deletion by this audit.
- Two standalone TTC preview renderers are retirement candidates, subject to a decision to stop reproducing their old previews.
- The July and August cleanup manifests are historical records, not current deletion instructions.
- Absence from a current paper dependency list does not prove that a file has no remaining use.

## Scope and checks

- Read the current TTC paragraph and appendix table description in `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/4_analysis.tex:38` and `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:467`.
- The paper now reports all 2,160 runs across ten seeds.
- Read the two old retention manifests, the retained-analysis README, TTC preview scripts, their shared analyzer imports, and samples of archived legacy runtime code.
- Cross-checked candidate paths against all E01–E44 JSON needed lists after those reports existed, including exact files and ancestor directory selectors.
- Neither candidate below occurs in those needed lists.
- Searched visible and ignored Python, shell, Slurm and report files across scripts, tests, analysis, dated review directories, reproduction audits, documentation, experiment results and the current paper.
- Searched result manifests and saved Claude tool logs as an additional dynamic-path check.
- Result manifests mention the five-seed renderer only in recorded Git status, not as an execution dependency.
- Both candidate files are tracked and have no current local changes.
- The two candidate files total 9,659 bytes, so deleting them would have little storage benefit.
- Other dirty or untracked user work remains protected.
- This is a bounded code and provenance review, not proof that every byte in every ignored data directory has no consumer.

## Retirement candidates, not deletion approval

| Absolute path | Specific purpose | Evidence and remaining condition |
| --- | --- | --- |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_ttc_average_target_utility.py` | Standalone three-panel TTC preview with a hard-coded statement that Claude max uses eight available seeds. | Complete source inspection shows CSV input and PNG/PDF/SVG output only; line 109 fixes the incomplete-seed annotation; the current paper uses the complete ten-seed result; no execution consumer was found in the inspected code and metadata; retain it unless the historical incomplete preview is explicitly retired. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_ttc_five_seed_portrait.py` | Standalone five-seed portrait-style plot variant with narrow panels and y-limits around the mean. | Complete source inspection shows two five-seed CSV inputs and PNG/PDF output only; original July 26 history confirms that these were user-requested display variants; the default output PNG/PDF still exist; no current execution consumer was found; retain it unless reproduction of that historical preview is explicitly retired. |

Neither script runs experiments or writes raw experiment results. Neither is an exact duplicate of the current renderer, so deleting it would remove the ability to regenerate that specific display variant directly. The historical previews are outside the current paper, but that is a scope distinction, not automatic permission to delete them.

## Files that look obsolete but remain needed

- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_three_seeds.py` supplies endpoint and seed-agreement functions called by the ten-seed analyzer at lines 122–123.
- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_five_seeds.py` supplies confidence intervals, plot helpers, reports and recovery-ID handling to the ten-seed analyzer at lines 124–125, 144–201 and 231.
- `/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:8` also imports the five-seed analyzer.
- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_ten_seeds_partial.py` explains the historical incomplete-seed estimator discussed in `/scratch/gpfs/DANQIC/jz4391/bargain/paper_revision_proposals/items/item_087.md:42`.
- Its main routine permits incomplete grids and writes missing-config metadata; retaining it preserves the meaning of the historical diagnostic, even though it must not be substituted for the final complete analysis.
- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_strategic_qualitative_tags.py` appears in E24's verified chain for the current qualitative analysis.
- The instruction to delete the entire retained-analysis directory in its README is therefore unsafe for the current paper.

## Why the old cleanup manifests cannot authorize deletion

- `/scratch/gpfs/DANQIC/jz4391/bargain/docs/reproducibility/script_retention_manifest.md` records a July 19 cleanup of an earlier 188-file script set.
- Its deleted-launcher list includes Game 2 and Game 3 backfill generators whose historical provenance matters to current results.
- `/scratch/gpfs/DANQIC/jz4391/bargain/docs/reproducibility/keep_delete_manifest_20260809.md` treats its KEEP list as complete and the complement as removable.
- The current root audit recovered 27 needed backfill manifest, configuration and launcher files from that manifest's deletion-staging tree.
- Their exact paths are recorded in `/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/root_recovered_archives.json`.
- I checked the recovered-file list and confirmed the two replacement families and selected configuration IDs recorded there.
- The August manifest also recommends deleting Slurm logs because they are ignored; current result audits use launch and recovery logs as provenance.
- Passing a figure verifier with an archive present or absent cannot prove that historical run-generation records are unnecessary.
- Keep the old manifests as evidence of previous moves, but do not execute their cleanup commands.

## Archived legacy code

- `/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/__init__.py:8` explicitly says the old negotiation modules were archived.
- The surviving archive contains 20 Python files under `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/TO_DELETE_20260809/repo_root/legacy`.
- The archived negotiation runner uses older imports and a different protocol description.
- The archived diplomacy implementation samples a clipped Gaussian and mixes Dirichlet weights, which differs from the current copula method.
- No import of those legacy implementations was found in the inspected current runtime, scripts, tests or analysis Python files.
- Their relationship to all historical exploratory results was not resolved file by file, so this audit does not approve those 20 files for deletion.

## Codex-search evidence

- Ran the bundled codex-search script for cleanup and for the two TTC renderer names.
- Manually read `/home/jz4391/.codex/sessions/2026/07/19/rollout-2026-07-19T01-24-35-019f78d5-5654-7e72-a948-45e06660770c.jsonl:526`, where the user explicitly retained qualitative and TTC analysis work for then-current revisions.
- Line 1390 describes the approved script reorganization, and line 2499 records the completed earlier cleanup.
- Manually read `/home/jz4391/.codex/sessions/2026/07/26/rollout-2026-07-26T21-15-19-019fa124-00be-73c0-9b56-414f329077f0.jsonl:431` and line 470, which describe the completed five-seed data and user-requested plot variants.
- These are locally available histories, not a complete record of all launches or all user activity.

## Unresolved boundary

- Retiring old exploratory plots requires a scope decision about retaining historical outputs, not another zero-reference search alone.
- Historical source-version identity remains incomplete for some experiment runs.
- External consumers, command-line invocations and notebooks can name paths indirectly.
- Do not delete unclassified code, complete archive trees, logs, failed runs or configs on the strength of this report.
- No source, data, configuration, paper or Git state was changed; only this audit's report files were added.
