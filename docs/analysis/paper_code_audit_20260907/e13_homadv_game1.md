# E13: Game 1 homogeneous-adversary experiment audit

## Scope and observed result
Current appendix lines 580–613 contains three associated figures. Each Game1 model/n mean pools five competition levels, two positions and two seeds: 20 runs. All 500 selected configs/results were independently read in this audit, with 3 no-agreement outcomes retained.

Independent raw calculations give slopes per 100 Elo for n=2,4,6,8,10 of 6.3553, 8.6185, 8.6455, 7.7086, 10.8441. Mean adversary-minus-baseline gap changes by +3.030299377913334 from n=2 to n=10, matching the +3.03 result. The 25 per-competition Game1 fits belong to the shared 65-fit figure; this audit did not independently recount their signs.

## Provenance chain
Saved batch manifest identifies April28 production root, five adversaries, n=2/4/6/8/10, c=0/.25/.5/.75/1. Config selection is game1 AND homogeneous_adversary (500). Concrete config_0511 sets six agents, 15 items, c=.5, random seed 1678107020, gamma=.9, max10 rounds and discussion2.

The original config_0511.log line3 records project Python calling run_strong_models_experiment.py with these values and --parallel-phases. Its status records success and attempt/job identity. Current batch launcher builds the same CLI and stores submission/status metadata. Slurm enables OpenRouter file proxy routing; external /home/jz4391/openrouter_proxy and credentials are protected operational dependencies, not deletion candidates. Secret contents were not opened.

Current CLI imports StrongModelsExperiment, agent factories, phase handlers, prompts, game factory and provider classes. Game1 utility generation depends on negotiation preferences and RandomVectorGenerator. Other game modules are eager imports in game_environments/__init__.py and cannot be removed just because this task is Game1.

Raw utility/role maps feed plot_full_games123_clean_subset and analyze_n2_plus_multiagent_comparison. Mean adversary utility gives payoff scaling; adversary utility minus within-run mean baseline utility gives gap. Final competition renderer consumes reproduction_audit/fig24_hom_adversary_competition/aggregated_plot_data.csv; final top-only gap renderer consumes reproduction_audit/fig30_multiagent_dilution/output/pooled_plot_points.csv. These old-looking reproduction_audit paths are active default inputs, not automatically disposable.

## History
Used codex-search skill and called bundled script twice (no visible results returned). Manually inspected /home/jz4391/.codex/sessions/2026/08/02/rollout-2026-08-02T01-57-18-019fc10c-5259-76a0-8efa-0e41696b8ab5.jsonl:1694-1697, confirming August2 top-only figure generation and integration. Earlier figure reconstruction identifies source patch at1632 and installation1650. Requested Sept5 session exists locally; its reconstruction artifacts were treated as leads, not independent original launch proof. April raw log provides direct launch evidence.

## Needed paths
Machine-readable companion enumerates current import dependencies and exact analysis inputs. Raw directories have explicit selection rules, NOT blanket required classification. Failed attempt logs/status must remain to explain retries; earlier recovery wording should not be erased merely because final status is SUCCESS.

## Limits and deletion candidates
- Original April experiment-code commit not recorded in sampled saved config/status. Current import chain cannot establish byte-identical historical runtime.
- Bundled Codex search invoked twice but returned no visible results this turn; manually verified original locally available August session via rg. Full pre-June conversation unavailable in figure audit.
- No full transitive static/dynamic import closure, external package versions or every retry archive reviewed. Current runtime has evolved and optional dependencies cannot be deleted based on this report.
- No candidate file has repository-wide negative dependency proof. No safe deletion recommendation.
- Group size changes item count as floor(2.5*n), so endpoint gap is descriptive, not isolated causal effect of n.
No edits, experiment runs, commits or deletions were performed.

