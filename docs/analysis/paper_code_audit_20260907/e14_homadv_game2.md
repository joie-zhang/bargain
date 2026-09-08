# E14 homogeneous-adversary Game 2 audit

## Scope and result
Current appendix lines 582–604 references three figures: pooled payoff/Elo by n, competition-separated payoff/Elo, and adversary-minus-baseline gap by n. This audit covers Game 2 panels. Current prose says gap grows; +7.67 is the earlier numerical endpoint description.

Read all 400 selected configs and raw results independently, without executing experiments. They form 5 models × 5 group sizes × 4 rho/theta settings × 2 orders × 2 seeds. Mean gap by n is:
- n=2: -0.8984672362500007
- n=4: 3.0580162500000005
- n=6: 2.0136875000000005
- n=8: 4.5385292857142865
- n=10: 6.774704694444445

Endpoint difference is +7.673171930694446; this is not monotonic growth across every adjacent n.

## Verified chain
1. scripts/full_games123_multiagent_batch.py:582 generates n_issues=10, rho at n-specific negative bound or0.9, theta0.2/0.8. Actual selected config1109 confirms rho=-1, theta0.8, seed1474959263.
2. Same launcher:1535 builds run_strong_models_experiment.py CLI with parallel independent phases, 10round cap,2discussion turns,gamma0.9. Actual logs/config_1109.log:3 records this exact command executed April28.
3. run_strong_models_experiment.py imports StrongModelsExperiment and provider-key access; experiment.py calls factory, game factory, phases, prompts, metrics and output writers. Runtime closure files are individually listed in JSON. Other game modules are eagerly imported by game_environments/__init__.py, even for Game2.
4. Game2 engine game_environments/diplomatic_treaty.py generates correlated uniform ideals and normalized importance weights then calculates discounted utility. Raw final_utilities plus agent_role_map supply gaps; each gap subtracts mean baseline payoff from adversary payoff.
5. scripts/plot_full_games123_clean_subset.py:355 reads raw outputs; scripts/analyze_n2_plus_multiagent_comparison.py:550–590 produces pooled analyses. Current by-n renderer is in lh_review_20260829/02_agent_n/candidates/fig24/reproduce.py, so that apparently disposable review directory contains a retained producer.
6. Competition renderer scripts/paper_figures/plot_figure24_hom_adversary_competition_clean.py reads reproduction_audit/fig24_hom_adversary_competition/aggregated_plot_data.csv. Reproduction producer remains needed.
7. Dilution renderer scripts/paper_figures/plot_figure30_dilution_top_only_clean.py reads reproduction_audit/fig30_multiagent_dilution/output/pooled_plot_points.csv; its reproduce.py reconstructs gaps from raw files. Some old reproduction routines use controls for competition-band extrema, so controls cannot be deleted merely because this top-only plot averages adversary runs.

## Runtime and external dependencies
The launch wrapper exports OpenRouter transport=proxy and shared queue path /home/jz4391/openrouter_proxy. It sources a credential env file whose values were not read. Current provider factory and configs determine routing. NumPy, SciPy, matplotlib, pandas, provider SDKs and project Python environment are required operational dependencies. Shared credential/proxy files are outside cleanup scope.
Current runtime has import dependencies on preference helpers, context compaction, JSON repair, analysis serialization and provider rotation, not just Game2 engine. Retain tests.

## History
Used codex-search skill and launched bundled repo search. Manually inspected original August2 session line1632 which adds exact retained dilution renderer and records its input CSV; cited original path in JSON. Also inspected August29 original asset/source inspection. Figure-recreation reports24/25/26 provide additional leads but are not substituted for raw verification. Full original May launch conversation and immutable original execution commit were not established.

## Limits and cleanup recommendation
Exact selected data membership is 400 configs with game_label=game2 and experiment_family=homogeneous_adversary in the production batch. IDs span1053..1870 but are not contiguous. Retain matching raw final results, interaction transcripts, statuses and all referenced attempt logs, plus selection/submission metadata. The JSON directory record explicitly limits membership and does not certify the entire batch tree.
All final observations were read, but every historical retry/provider switch and every launch selection was not resolved. Current code is not necessarily the historical execution snapshot. No files are certified removable; absence from this dependency list is not deletion evidence.

