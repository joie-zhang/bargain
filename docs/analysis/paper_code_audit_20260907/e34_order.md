# E34 Speaking-order diagnostic

## Result and verification

The current appendix at lines524-532 compares intended adversary-first and baseline-first groups. I independently opened all1,500 selected configurations and result JSON files, checked every plotted adversary payoff against raw final_utilities with the established failure-zero rule, and recomputed all six unweighted fits from30 model means per order. The crossings are1393.869265,1364.588719 and1345.988648 for Games1-3. Game1/2 favor adversary-second below the crossing and adversary-first above it; Game3 reverses direction. The current paper PNG and prior raw recreation have SHA256 6c47c988361367e5add0c1d9d6eb180c1828b59ef8406eafa41b8c8afbe8854a.

## Provenance chain

Saved launchers in the three primary result roots invoke run_strong_models_experiment.py. E01-E03 cover complete launch/runtime/provider dependencies and historical replacements. The index and config selections supply420/540/540 runs. scripts/analyze_n2_baseline_comparison.py extracts raw utilities and assigns roles using saved configurations, then produces primary_runs_with_metrics.csv. scripts/paper_figures/plot_figure23_gpt5_nano_order_clean.py selects the1,500 Nano-baseline rows and groups by game/model/intended order. Its normal input is shared with Llama analyses; deleting those inputs would break the generic upstream generator. The raw-only recreation offers a separate exact selection route, not proof that shared inputs are unnecessary.

The code at phase_handlers.py:1473 iterates public speakers in agent order. Proposal contexts at2178 use shared prior public context and separate private state; dispatch at2602 uses the common task executor. This is a speaking-order comparison, not sequential observation of another same-round proposal.

## History

Used the bundled codex-search query bilateral_order_diagnostics crossover first second1394. Top results were current forked conversation copies, not original creation evidence. I manually inspected the original August2 session019fc0ec-87ac-7be0-9159-a5402a2016ad at lines1361,1400,1405,1406,1413. It records the user layout request, creation of the current renderer, successful execution with180 aggregate rows from1,500 runs, and file synchronization. Exact full history path is in the JSON. Earlier April launch session availability remains incomplete as recorded by E01-E03.

## Identity and interpretation limits

The11 transcript-confirmed Game1 reversals recorded by E01 also affect this figure because it uses the same configured attribution. The earlier figure02 actual_order_audit.json says zero mismatches, but that check compared result metadata, not the independent transcript evidence. Preserve transcript_role_discrepancies.json and its11 exact transcript paths. Do not treat the zero-mismatch metadata file as proof of correct runtime identity.

The raw audit also includes two Game3 Qwen72 records with only the baseline present; intended adversary utility is zero-filled. Twelve other rows have absent preference/utility fields from failures. A naive model-string comparison is not valid because GPT-5-nano-high shares a runtime model name with the baseline. My preliminary substring check returned61 matches, which is not a valid mismatch count and is not used as evidence of61 reversals.

The existing preference-pair audit records750 pairs,737 with complete adversary preferences and all737 different across orders;13 lack complete evidence. Seeds match across Game1 orders but differ across Game2/3 orders. Thus the crossing is descriptive, not an isolated causal estimate of turn order. Historical code revisions and replacement records remain necessary provenance, even if current extraction no longer reads them.

## Cleanup decision

No deletion candidates established. JSON lists concrete source dependencies and exact selection rules for large raw cohorts; it does not approve retention of every file in those roots. Unlisted files remain unaudited, not unnecessary. Only these report files were created; no code, paper, data or Git changes were made.
