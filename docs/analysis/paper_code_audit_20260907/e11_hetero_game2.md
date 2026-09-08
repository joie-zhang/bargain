# E11: Heterogeneous Game 2 experiment dependencies

## Scope
Current appendix lines 550–574 covers labels fig:appendix_multiagent_hetero_payoff_full, fig:multiagent_hetero_buckets, and fig:appendix_multiagent_hetero_competition. This task covers their Game 2 portions only. Current figure numbering differs from September 5 reconstruction: reconstruction figure_27 is by-n scaling, figure_28 is pooled Elo buckets, figure_29 is competition detail.

## Verified chain
- Original generation/launch entry is scripts/full_games123_multiagent_batch.py. Run-root RUN_NOTES.md records generate → select heterogeneous_all → submit-selection commands. Saved slurm/run_full_games123.sbatch invokes its run-one entry, which constructs run_strong_models_experiment.py CLI at lines 1535 onward.
- The saved full manifest has 2730 generated configs, but heterogeneous_all_config_ids.txt selects 1300. Do not equate generation count with retained run count.
- Reread all selected status/result payloads, filtering embedded config.game_label=game2. There are exactly 400 Game 2 runs: n=2,4,6,8,10 × two rho values × theta=.2,.8 × 20 sampled rosters. Rho negative limits are -1, -.319802274228682, -.19130568257555955, -.1365347919384111, -.10615795122401363; positive rho=.9. Ten issues, ten-round cap, gamma=.9.
- The 24-model pool and per-run model/seat/Elo maps are embedded in result configs. The sampler first chooses an unordered distinct-model subset, stratified by five equal-width within-roster Elo standard deviation intervals, then randomizes order; reuse across runs is allowed. Four draws per stratum produce 20 per exact cell. Preserve RNG states, maps and configuration seeds.
- Current runtime imports StrongModelsExperiment → agents, phases, prompts, analysis and utility modules. Game2 routes through game_environments/diplomatic_treaty.py:314 (state generation), 371 (copula positions), 412 (weights), 894 (utility). The package initializer eagerly imports other-game classes, so these modules are still load-time dependencies even for Game2.
- Provider transport uses negotiation/openrouter_client.py and negotiation/llm_agents.py, key rotation and context compaction; Slurm exports OPENROUTER_TRANSPORT=proxy and the external /home/jz4391/openrouter_proxy queue. No credentials or external monitor contents were read.
- Original multiagent analysis scripts/analyze_n2_plus_multiagent_comparison.py imports plot_full_games123_clean_subset.py and analyze_nash_lindahl_fairness.py. Game2 plotting index is theta*(1-rho)/2 (line 425), not the older rho*theta helper present in clean_subset.
- Retained by-n reconstruction reads selected statuses and raw results, derives agent mean payoff by model/game/n, adds SEM and fits unweighted OLS to the 24 model means. Preserved fit_summary.csv has Game2 slopes per 100 Elo 4.7897, 4.5339, 3.1984, 1.6208, 3.8658 for n=2..10.
- Competition reconstruction figure_29/scripts/recreate.py reads raw final utilities and embedded Elo maps, computes exact competition indices, and uses retained aggregate_het_agents and plot_figure25_heterogeneous_competition_clean.py. Twenty Game2 fits contribute to the cross-game count.
- Elo buckets figure_28/scripts/producer_snapshot.py:190–214 first averages agent appearances by game/n/bucket, then equally averages the three game means. Game2 is one-third of each populated all-game mean, not an independently reported Game2-only bucket chart.

## History evidence
Used codex-search bundled search script with query heterogeneous Game 2 1300 stratified. Top results were current September 7 conversation forks and were rejected as original provenance. Manually read original history.jsonl lines 1559 and 1563: session 019df655-b70c-7491-9b65-410505d06180 requested all-three-game heterogeneous scaling across n and then error bars. Its May full session is absent according to retained history coverage, so only the user-request record is independently established.

Manually inspected full primary session /home/jz4391/.codex/sessions/2026/08/29/rollout-2026-08-29T20-38-40-01a0501a-ae58-7fc3-93a1-917ae569c417.jsonl lines 363, 383, 436. These contain the retained fig27 renderer patch, command execution, and SHA256 58eb1c971c5b76d0cac70c59589dce49eca5fd80ac931ebfa8a63d41017c6284. The requested September 5 session exists at /home/jz4391/.codex/sessions/2026/09/05/rollout-2026-09-05T23-13-06-01a074b4-92a4-78c3-96f8-c42ae5db663d.jsonl; its reconstruction artifacts were used as leads and independently checked against raw files, not accepted as original launch proof.

## Integrity and deletion risks
Reread raw payloads independently. Eight selected Game2 results record contaminated voting: configs 1250,1412,1541,1584,1626,1676,1705,1790. Six contain synthetic_proposal=true: 1250,1295,1412,1500,1790,1798. Union is 11 affected Game2 runs. These records remain in the original plotted cohort. This audit does not recompute a clean-cohort result or remove anything. RUN_NOTES initially claims no silent synthetic fallback; current phase_handlers and actual payload markers document synthetic actions, so preserve both notes and raw/recovery history to explain the chronology.

Exact code revision for each historical attempt is unresolved. RUN_NOTES records live client patches and task retries; a current runtime file is not proof it was the exact historical source. Current launch writes phase token caps and metadata that may differ from first-attempt configs. Failed logs and apparently superseded artifacts are not safe to delete.

No deletion candidates are certified. Files not used by this one result may serve another result, tests, current workflows, or provenance. Sidecar paths, conditional runtime branches, external services and historical code snapshots still require wider review. Tests were protected, not executed; no figures or experiments were rerun.

## Deliverables
e11_hetero_game2.json contains concrete existing needed file paths with reasons and evidence, plus directory selections only for large cohort material. Every listed path exists. Directory entries are not blanket per-file necessity claims. No code, data, paper, Git or environment changes were made.

