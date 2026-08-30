# ICML Count and Statistic Update Proposals

Date: 2026-07-20

## Status

This file is a review document.
It does not change the paper source.
Do not edit an ICML `.tex` file from this document without author approval.

The current paper source is
`overleaf/icml_aiwild_template/icml_aiwild_2026.tex`.

## Confirmed Inventory

The current paper scope excludes all Phi-3 Mini runs.
It includes the valid rerun of Game 3 `config_0412`.

| Batch | Accepted runs |
| --- | ---: |
| GPT-5-nano bilateral, Game 1 | 840 |
| GPT-5-nano bilateral, Game 2 | 540 |
| GPT-5-nano bilateral, Game 3 | 540 |
| GPT-5-nano bilateral subtotal | 1,920 |
| Llama 3.3 70B bilateral | 500 |
| Multi-agent homogeneous | 1,430 |
| Multi-agent heterogeneous | 1,300 |
| Random monoculture | 325 |
| Test-time compute | 216 |
| **Paper total** | **5,691** |

The arithmetic is:

```text
1,920 + 500 + 1,430 + 1,300 + 325 + 216 = 5,691
```

The earlier 5,712-run inventory contained 22 valid Phi runs.
It did not contain the malformed Game 3 `config_0412` result.
The new scope removes 22 runs and adds the valid `config_0412` rerun.
Thus, the net change is minus 21 runs.

## Confirmed Paper Replacements

The table gives facts and numeric replacements only.
It does not propose new paper prose.

| Source | Current value | Correct value | Reason |
| --- | --- | --- | --- |
| `abstract.tex:1` | `5712` | `5691` | New total after the Phi exclusion and the valid `config_0412` rerun. |
| `5_conclusions.tex:1` | `5,712` | `5,691` | Same total. |
| `4_analysis.tex:26` | `1,941` bilateral runs | `1,920` bilateral runs | Game 1 has 840 accepted non-Phi runs. Games 2 and 3 have 540 each. |
| `appendix.tex:39` | `1,941` bilateral runs | `1,920` bilateral runs | Same subtotal. |
| `appendix.tex:39` | `862` Game 1 runs | `840` Game 1 runs | Remove 22 historical Phi results. |
| `appendix.tex:39` | `31` Game 1 adversary models | `30` Game 1 adversary models | Remove Phi-3 Mini. |
| `appendix.tex:39` | `539` Game 3 runs | `540` Game 3 runs | Include the valid `config_0412` rerun. |

## Protocol Facts

The Game 1 inventory has two protocol arms:

| Protocol | Accepted runs | Use in paper |
| --- | ---: | --- |
| One discussion turn | 420 | Appendix ablation |
| Two discussion turns | 420 | Primary analysis |

Games 2 and 3 each have 540 runs.
They use two discussion turns.

The following locations currently say or imply that all bilateral runs use two
discussion turns:

- `3_approach.tex:112`
- `appendix.tex:39`

The author must revise these statements to distinguish the Game 1 ablation from
the primary two-turn protocol.
No replacement prose is supplied here.

The primary GPT-5-nano bilateral analysis has 1,500 runs:

```text
420 Game 1 two-turn runs + 540 Game 2 runs + 540 Game 3 runs = 1,500
```

## Confirmed Derived Statistics

These values come from model-level means in
`experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv`.
That file contains zero Phi rows.

| Game | Current adversary slope | Correct slope | Current 95% CI | Correct 95% CI |
| --- | ---: | ---: | --- | --- |
| Game 1 | +5.28 | +6.75 | [3.54, 7.01] | [4.46, 9.05] |
| Game 2 | +6.76 | +6.76 | [4.53, 8.99] | [4.53, 8.99] |
| Game 3 | +7.37 | +7.40 | [4.74, 9.99] | [4.80, 10.00] |

The unit is utility per 100 Elo.
The affected locations are:

- `4_analysis.tex:26`
- `appendix.tex:314-316`, in the GPT adversary-slope column
- `appendix.tex:517`

The GPT-5-nano baseline-payoff slopes also change:

| Game | Current value | Correct value |
| --- | ---: | ---: |
| Game 1 | +1.14 | +0.24 |
| Game 2 | +1.66 | +1.66 |
| Game 3 | +2.42 | +2.44 |

The affected location is `appendix.tex:314-316`.
The Llama-baseline slopes do not change.

The Game 1 rounds-to-consensus slope changes from `-0.15` to `-0.23` rounds per
100 Elo. The affected location is `appendix.tex:532`.
The Game 2 value remains `-0.09`.
The Game 3 slope remains approximately zero.

## Confirmed Qualitative Replacements

The qualitative source now uses the 1,920-row Phi-free adjudication manifest.
Of these rows, 1,891 have usable adversary payoff data.
The category counts are deduplicated by rollout, speaker, round, discussion
turn, phase, and category.

The source and provenance files are:

- `overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_intensity_dedup.csv`
- `overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_payoff_corr_dedup.csv`
- `overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_qualitative_dedup_provenance.json`

The exporter is:

`scripts/paper_figures/export_n2_qualitative_dedup.py`

Use these replacements in the paper:

| Source | Current value | Correct value |
| --- | --- | --- |
| `4_analysis.tex:47` | `1,941` transcripts | `1,920` transcripts |
| `4_analysis.tex:49` | pressure `-0.05` | pressure `-0.05` (rounded; exact `-0.0508`) |
| `4_analysis.tex:49` | trade/compromise `+0.17` | trade/compromise `+0.17` (rounded; exact `+0.1746`) |
| `4_analysis.tex:49` | logical persuasion `+0.12` | logical persuasion `+0.11` (exact `+0.1091`) |
| `appendix.tex:725` | `2,730 N=2 rollouts` | `1,920 N=2 rollouts` |
| `appendix.tex:732` | `1,912` speaker-rollouts | `1,891` speaker-rollouts |
| `appendix.tex:749` | `1,912` speaker-rollouts | `1,891` speaker-rollouts |

The category-level appendix table at `appendix.tex:741-746` must use these
values:

| Category | Spearman r | p | Nonzero rollouts |
| --- | ---: | ---: | ---: |
| trade/compromise | +0.175 | 2.1e-14 | 1,357 |
| emotional persuasion | +0.135 | 4.3e-09 | 703 |
| logical persuasion | +0.109 | 2.0e-06 | 1,000 |
| pressure | -0.051 | 0.027 | 941 |
| self-interest/exploitation | -0.273 | 8.7e-34 | 609 |
| formalization | -0.341 | 9.5e-53 | 181 |

The table header says `n rollouts used`, but these values count rollouts with a
nonzero event count. The correlation uses all 1,891 payoff-valid rollouts,
including zero-count rows. The author should rename that column to prevent a
sample-size misinterpretation.

### Confirmed Per-Tag Replacements

The per-tag mechanism rows at `appendix.tex:765-792` must be replaced from:

`overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_tag_mechanism_dedup.csv`

Each correlation uses all 30 non-Phi models. The exporter computes model mean
utility from the available payoff rows. It does not drop a model when one of its
64 rollout payoffs is missing.

| Category | Tag | rho(Elo) | slope/100 Elo | rho(payoff) |
| --- | --- | ---: | ---: | ---: |
| pressure | Adversarial callout | +0.85 | +0.12 | +0.65 |
| pressure | Conditional veto threat | +0.72 | +0.16 | +0.59 |
| pressure | Fairness accusation pressure | +0.66 | +0.07 | +0.43 |
| pressure | Frustration or disappointment display | +0.59 | +0.02 | +0.45 |
| pressure | Ultimatum language | +0.61 | +0.06 | +0.47 |
| emotional persuasion | Rapport before pressure | +0.24 | +0.04 | +0.30 |
| emotional persuasion | Empathy-then-pivot | -0.62 | -0.04 | -0.48 |
| logical persuasion | Threshold-gap calculation | +0.63 | +0.02 | +0.46 |
| logical persuasion | Agent-specific payoff accounting | +0.70 | +0.07 | +0.68 |
| logical persuasion | Fairness ledger argument | +0.47 | +0.03 | +0.47 |
| logical persuasion | Utility arithmetic receipts | +0.54 | +0.10 | +0.58 |
| trade/compromise | Low-weight concession leverage | +0.52 | +0.04 | +0.51 |
| trade/compromise | Conditional quid pro quo | +0.48 | +0.06 | +0.40 |
| trade/compromise | Vote-history diagnostics | +0.41 | +0.04 | +0.37 |
| trade/compromise | Conditional support ledger | +0.31 | +0.01 | +0.30 |
| trade/compromise | Concession laddering | +0.01 | -0.00 | -0.11 |
| self-interest/exploitation | Silent free beneficiary | +0.36 | +0.01 | +0.45 |
| self-interest/exploitation | Zero-value subsidy | -0.10 | -0.00 | -0.11 |
| self-interest/exploitation | Leverage preservation | +0.16 | +0.00 | +0.04 |
| self-interest/exploitation | Self-advocacy/value maximization | +0.09 | +0.01 | +0.12 |
| self-interest/exploitation | Accepted-loss capitulation | -0.43 | -0.01 | -0.48 |
| formalization | Counter-anchor cost policing | +0.84 | +0.06 | +0.83 |
| formalization | Budget carryover hallucination | -0.67 | -0.03 | -0.68 |

The old exploration code joined model payoff summaries on the number of usable
payoff rows and the total rollout denominator. This could silently remove a
model from a correlation when one payoff was missing. The retained exporter
does not use that join.

### NBS Decomposition

The NBS decomposition table is at `appendix.tex:331-334`.
Its current values were not validated against the Phi-free primary table in this
audit. Regenerate the table before a paper edit.

## Figure Status

The Phi-free main bilateral render exists at:

`analysis/recreated_figures/figure2_bilateral_overview_combined_large_fonts.png`

Its SHA-256 hash is:

```text
58ad787139a56f7df99a37b6805e9fbcbc98474f9b3aaf949b0d52774bb4f926
```

The ICML paper now uses the same Phi-free render at:

`overleaf/icml_aiwild_template/graphics/n2_gpt5_nano/bilateral_overview_combined.png`

Its current SHA-256 hash is:

```text
58ad787139a56f7df99a37b6805e9fbcbc98474f9b3aaf949b0d52774bb4f926
```

The hashes match.

The ICML qualitative figure is also Phi-free. Its SHA-256 hash is:

```text
857a69ac968efcbd8f245bdc31f4a71d895366c03ea657f45360801d28297d0c
```

The Game 1 discussion-turn ablation render contains 420 runs in each arm and no
Phi rows. It is at:

`overleaf/icml_aiwild_template/graphics/n2_gpt5_nano/game1_discussion_turn_ablation.png`

The rollback removed this plot from `appendix.tex`.
The file remains a proposed appendix asset, but it is not listed as an active
ICML figure.

### Appendix Asset Audit

The following active ICML appendix assets were regenerated or promoted from
the current Phi-free primary N=2 analysis bundle:

- Adversary payoff overall and by competition.
- Baseline payoff by competition.
- Total welfare by competition.
- Fairness distance overall, by competition, and across the three games.
- Fairness excess by role.
- Rounds to consensus overall and by competition.
- Bilateral order diagnostics.

The fair-share residual and endpoint fairness assets no longer use the old
fixed bitmap or Elo snapshot. Their current producers load canonical data and
enforce the Phi-free, two-turn Game 1 primary protocol.

### Fair-share Caption Proposal

Do not apply this change without author approval.

- Open `overleaf/icml_aiwild_template/4_analysis.tex`.
- Go to line 34.
- Find this exact text:

```latex
Every series rises with Elo and crosses zero near Elo $\sim 1450$; positive values mean the focal agent takes more than its fair share.
```

- Replace it with this text:

```latex
The heterogeneous-agent and inserted-adversary series rise with Elo; the homogeneous GPT-5-nano control provides a fixed-Elo reference. Positive values mean the focal agent takes more than its fair share.
```

Reason: the homogeneous control is a fixed reference point, not a rising
series. The homogeneous-baseline trend also does not cross zero in the shown
range.

## Repository Values

The paper experiment manifest currently has 5,691 rows and zero Phi rows:

`docs/reproducibility/paper_experiment_data_manifest.csv`

The manifest builder validates the same counts:

```bash
python scripts/build_paper_experiment_data_manifest.py --check
```

The random-monoculture manifest is release-complete.
The July 20 recovery of `config_0262` produced a result and rollout with the same
experiment ID.
The monoculture count remains 325 runs.

## Numeric Strings That Must Not Change

Some search matches are identifiers, not paper counts.
Do not change these values mechanically:

- A CSV row number such as `5712`.
- A config ID such as `config_1941`.
- A random seed or file size that contains the same digits.
- The historical statement that the earlier inventory had 5,712 runs.

## Validation Basis

The confirmed counts come from these Phi-free files:

- `experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv`
- `experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv`
- `docs/reproducibility/paper_experiment_data_manifest.csv`

The all-protocol table has 2,420 rows and zero Phi rows.
It contains 1,920 GPT-5-nano bilateral rows and 500 Llama-baseline rows.
The primary table has 2,000 rows and zero Phi rows.
It contains 1,500 GPT-5-nano bilateral rows and 500 Llama-baseline rows.
