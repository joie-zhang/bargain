# E07: GPT-5 test-time effort payoff scaling

## Scope and observed result

Current paper Figure 5 (fig:ttc_scatter), main-text scaling paragraph, and appendix tab:appendix_ttc_overall report the GPT-5 endpoint gain and its approximate Elo equivalent. Source: /scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/4_analysis.tex:32-38 and appendix.tex:237,467-485.

This audit independently opened all 720 GPT-5 selected configs, terminal results, and terminal transcripts. There are 180 observations at each requested effort level. Direct means are minimal 63.6239748167, low 66.3700032, medium 67.4058570373, high 67.0751848062. High minus minimal is 3.4512099895. Dividing by the paper's 6.97 payoff/100Elo gives49.515Elo, rounded50. Baseline slope provenance is a separate result audit.

## Verified generation and selection chain

1. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:19-60 defines gpt-5-2025-08-07 under four effort aliases; baseline is gpt-5-nano, low effort.
2. /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943/configs/config_0001.json through config_0072.json are the original GPT-5 selected configs. Full generator produces216 configs across all model families.
3. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:70-150 clones all216 source configs, changing seed/output paths and recording source IDs. Seeds are42,984,526,423,1024,128,256,612,2048,4096. This audit independently selected the72 GPT-5 rows per seed; JSON contains exact paths, not blanket-directory protection.
4. Saved slurm/run_one.sbatch files call /scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py, which invokes /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py using selected config models, seed, game parameters, seat ordering, round/phase settings, output cap, and output directory. Generated launchers activate .venv, load cluster proxy module, and may source externally managed API key environment; do not read/delete secrets.
5. Runtime entrypoint imports StrongModelsExperiment, whose factory reads /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:201-241. /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:445-452 passes reasoning_effort into OpenAIAgent custom parameters. /scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2820-2883 sends these with max_completion_tokens through OpenAI chat completions or its shared proxy path.
6. Experiment phase handler invokes PromptGenerator and game environments, then /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:841 computes final outcome utilities, assigning zero after failed agreement. Final selected terminal artifacts are run_1_experiment_results.json and run_1_all_interactions.json.
7. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:65-170 selects config.output_dir/run_1_experiment_results.json, maps target_position to Agent1/2, takes stored final_utilities, averages paired orders within nine game cells, then averages cells. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py is the current complete-panel producer; it imports this helper.
8. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_icml_ttc_main_figures.py reads /scratch/gpfs/DANQIC/jz4391/bargain/analysis/ttc_complete_family_seed_panels_20260810/family_effort_complete_seed_ci95.csv. The actual half-width asset producer is /scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260829/07_halfwidth_footnote/make_candidate.py. Its historical input is /scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/icml_aiwild_current_20260829/fig05_ttc/derived_summary.csv. This review-directory producer must not be presumed unused.
9. /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/scripts/recreate.py and raw_source_snapshot.py supply the separately retained raw-data aggregation route. I read this code and independently parsed the720 GPT-5 rows rather than accepting its older report as proof.

All GPT-5 conditions have two agents, ten-round cap, two discussion turns and gamma0.9. Nine cells comprise Game1 competition0,.5,1; Game2 rho-1,0,1 with theta1; Game3 three alpha/sigma combinations, each in both seat orders. Config files, rather than an unrelated generic YAML, define the executed cohort.

## Recovery is part of provenance

Three retained high-effort GPT-5 results have effective16384 cap while original configs say10500: seed984 configs68 and72; seed526 config68. All have matching cap16384_attempt2 recovery configs. Seed984 submitted_jobs.tsv lines77/81 and218/219 directly record original/recovery submissions. The recovery helper /scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py refuses to replace terminal results, moves failed output to recovery/failed_attempts, and launches adjusted configs.

The failed-attempt transcripts, progress files and per-agent logs for these three cases are explicitly protected in JSON, alongside recovery configs/logs and submission ledgers. The stated recovery reason is empty output at the output-token limit. All high-effort output caps therefore are not identical to low-effort caps; interpreting the result as an isolated effort effect requires qualification.

## Local history checked

Codex-search script searched locally available repository-linked history; returned2175 session records and42 history-only IDs. Top match was session019fa81f-23b2-7d71-bc1d-b06f1965c20e. I manually read its original lines10551,10580,10743: they describe the partial2106-run predecessor, not the final2160 cohort.

I also manually verified:
- /home/jz4391/.codex/sessions/2026/08/14/rollout-2026-08-14T04-28-46-019fff63-4eac-7e31-8963-368d5f16f095.jsonl:5,42,73: user explicitly requests2160 inclusion, cohort patched, figure producer invoked.
- /home/jz4391/.codex/sessions/2026/08/29/rollout-2026-08-29T20-37-36-01a05019-b13e-7720-8b5c-473e30522a76.jsonl:955,961,981: halfwidth producer and styling updated, re-executed, asset copied.

The supplied September5 audit/session is useful provenance context, but raw reads above are the numerical evidence.

## Current runtime dependency protections

The JSON lists2266 concrete existing paths:2160 selected config/result/transcript paths, three recovery configs,52 cohort/launch/recovery provenance files, and51 inspected code/support dependencies.

Shared runtime modules include experiment/data models/configs, agent factory, phase handler, prompt generator, analyzer/qualitative schemas, file utilities, all three game engines and metrics, negotiation preferences/random-vector generator, provider clients/key rotation, JSON repair and context compaction. Package __init__.py imports are included. Even OpenRouter client is imported by the shared agent factory despite the GPT-5 condition naming OpenAI.

Context compaction dynamically reads /scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md. This seemingly historical documentation file is a live runtime input. TTC regression tests import seed replication/cap helpers. requirements.txt and externally managed .venv/provider environment need protection.

/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_ten_seeds.py imports analyze_ttc_five_seeds.py and analyze_ttc_three_seeds.py. Their older names do not establish nonuse.

## Limits and removals

No file is recommended for deletion from this result. A file outside this dependency list may support another experiment, test, interface, or historical explanation.

Current launchers invoke the live worktree; I did not recover frozen exact runtime bytes for every historical invocation. Saved provider/model IDs plus present code do not prove every historical wire request. No API calls, experiments, new figures, or broad utility recomputation were run. Exact dependency completeness is not proven for external SDKs, dynamic routes, archived code versions, or all potential environment-dependent paths. Do not claim100% precision/recall.

Only the two requested audit reports were written. Paper, experiments, configs and runtime code are unchanged.

