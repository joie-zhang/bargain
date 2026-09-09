# Reproduction scripts

Keep these 14 scripts with the paper's analysis code. They prepare or validate
annotations, calculate reported comparisons, or supply coalition figure inputs.
The game runtime, experiment launchers, and newer TTC tools remain in their
existing directories.

Run these scripts from the checkout root. This move preserves their contents
and output locations. Several scripts still use the original cluster path;
this directory split does not make those scripts portable to another checkout.

## Annotation preparation and validation

| Script | Role |
| --- | --- |
| `prepare_llm_strategic_tag_adjudication.py` | Prepare multi-agent annotation tasks from the saved rollout inventory and codebook. |
| `prepare_llm_strategic_tag_adjudication_n2_gpt5.py` | Prepare bilateral annotation tasks. |
| `prepare_llm_strategic_tag_adjudication_random_monoculture.py` | Prepare random-monoculture annotation tasks. |
| `prepare_ttc_llm_strategic_tag_adjudication.py` | Prepare the original TTC cohort used in the combined qualitative analysis. |
| `validate_llm_strategic_tag_adjudication.py` | Validate and aggregate the multi-agent annotation outputs. |
| `validate_ttc_llm_strategic_tag_adjudication.py` | Validate and aggregate the original TTC annotation outputs. |
| `explore_llm_strategic_tag_elo_trends.py` | Calculate tag summaries used by the bilateral qualitative export. |

## Coalition calculations and figures

| Script | Role |
| --- | --- |
| `build_gemini_coalition_qualitative.py` | Calculate coalition prevalence and voting comparisons, including the Gemini cases. |
| `build_minimum_winning_coalition_report_20260817.py` | Build outcome tables from the recorded coalition audit. |
| `audit_strict_coalition_proposals_20260817.py` | Classify the recorded candidate proposals by harm to excluded agents. |
| `build_game1_coalition_proposer_elo_20260823.py` | Build the Game 1 proposer table. |
| `build_game1_coalition_proposal_rate_vs_elo_20260823.py` | Calculate coalition proposal rates by model Elo. |
| `plot_coalition_proposed_accepted_by_game_20260823.py` | Plot proposed and accepted coalitions by game. |
| `plot_game1_coalition_conversion_by_family_20260823.py` | Plot proposed and accepted coalitions by experiment family. |

These scripts need the saved raw runs, annotation outputs, codebooks, rollout
inventories, and coalition review records. Some coalition classifications are
recorded human judgments in the source, rather than automatic transcript labels.
Keep those records with the reproduction inputs.

Older codebook construction and batch repair scripts are listed in
[historical support](../historical_support/README.md). No Python import from
this directory requires that directory. The saved inputs those scripts created
can still be required.
