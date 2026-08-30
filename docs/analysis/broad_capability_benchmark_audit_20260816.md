# Capability measures with more than 20 matched models

## Question

Which capability measures cover more than 20 of the 30 bilateral adversary models, and does the positive payoff trend remain across them?

## Short answer

- Eleven measures pass the strict `n > 20` filter.
- All 33 benchmark-by-game slopes are positive.
- Thirty of the 33 slopes pass a 5% false-discovery-rate threshold after correction across the filtered tests.
- CritPt has positive point estimates, but none of its three adjusted tests pass 5%.
- MMLU-Pro is absent because it covers 17 models.
- Only the Artificial Analysis composite contains estimated x-axis values.
  - Twenty-two of its 29 index values are estimates.
  - The individual Artificial Analysis benchmark fields are not marked as estimated by the source.

## Filtered cell plot

- Inclusion rule: more than 20 matched roster models.
- Each point is one model's mean adversary payoff in one game.
- Error bars show the payoff standard error over that model's runs.
- Dashed lines are unweighted OLS fits over model-level means.

![Capability measures with more than 20 matched models](../../analysis/capability_metric_robustness_20260816/plots/bilateral_payoff_capability_benchmarks_over20.png)

- PNG: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/plots/bilateral_payoff_capability_benchmarks_over20.png`
- Vector PDF: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/plots/bilateral_payoff_capability_benchmarks_over20.pdf`

## Filtered coverage

| Measure | Matched models | Nearby configurations | Estimated x values | Adjusted significant games |
|---|---:|---:|---:|---:|
| LM Arena Elo | 30 | 0 | 0 | 3/3 |
| Artificial Analysis Intelligence Index v4.1.1 | 29 | 5 | 22 | 3/3 |
| GPQA Diamond, Artificial Analysis evaluation | 29 | 5 | 0 | 3/3 |
| Humanity's Last Exam, Artificial Analysis evaluation | 29 | 5 | 0 | 3/3 |
| SciCode | 29 | 5 | 0 | 3/3 |
| IFBench, retired from the AA Index | 27 | 4 | 0 | 3/3 |
| tau2-bench, superseded | 26 | 5 | 0 | 3/3 |
| AA-LCR | 26 | 4 | 0 | 3/3 |
| Terminal-Bench Hard, superseded | 25 | 5 | 0 | 3/3 |
| AA-Omniscience | 24 | 4 | 0 | 3/3 |
| CritPt | 24 | 4 | 0 | 0/3 |

## Individual benchmark plots

### LM Arena Elo

![Bilateral payoff versus LM Arena Elo](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_arena_elo.png)

### Artificial Analysis Intelligence Index v4.1.1

![Bilateral payoff versus Artificial Analysis Intelligence Index v4.1.1](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_index.png)

### GPQA Diamond

![Bilateral payoff versus GPQA Diamond](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_gpqa.png)

### Humanity's Last Exam

![Bilateral payoff versus Humanity's Last Exam](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_hle.png)

### SciCode

![Bilateral payoff versus SciCode](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_scicode.png)

### IFBench

![Bilateral payoff versus IFBench](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_ifbench.png)

### tau2-bench

![Bilateral payoff versus tau2-bench](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_tau2.png)

### AA-LCR

![Bilateral payoff versus AA-LCR](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_lcr.png)

### Terminal-Bench Hard

![Bilateral payoff versus Terminal-Bench Hard](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_terminalbench_hard.png)

### AA-Omniscience

![Bilateral payoff versus AA-Omniscience](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_omniscience.png)

### CritPt

![Bilateral payoff versus CritPt](../../analysis/capability_metric_robustness_20260816/plots/broad_individual/bilateral_payoff_vs_aa_critpt.png)

## GPQA result

| Game | Models | Payoff per score standard deviation | 95% CI | R-squared | Adjusted p |
|---|---:|---:|---:|---:|---:|
| Item allocation | 29 | +7.73 | [5.71, 9.76] | 0.694 | 0.0000007 |
| Diplomatic Treaty | 29 | +6.53 | [3.96, 9.10] | 0.502 | 0.000044 |
| Co-funding | 29 | +8.31 | [5.87, 10.74] | 0.645 | 0.0000013 |

- Removing the five nearby configuration matches leaves 24 models.
- The standardized slopes remain +7.95, +6.49, and +8.23.

## Humanity's Last Exam result

| Game | Models | Payoff per score standard deviation | 95% CI | R-squared | Adjusted p |
|---|---:|---:|---:|---:|---:|
| Item allocation | 29 | +5.11 | [2.05, 8.17] | 0.304 | 0.0027 |
| Diplomatic Treaty | 29 | +3.58 | [0.23, 6.94] | 0.151 | 0.0411 |
| Co-funding | 29 | +5.04 | [1.47, 8.60] | 0.237 | 0.0087 |

## Interpretation

- The direction is stable across every retained measure and game.
- GPQA, SciCode, and AA-LCR have some of the strongest model-level associations.
- HLE is positive but weaker than GPQA.
- CritPt is narrow and its confidence intervals include zero in all three games.
- Nine of the 11 retained measures come from Artificial Analysis.
- Shared evaluation source and high correlations between capability scores mean these are related sensitivity checks, not independent replications.
- IFBench, tau2-bench, and Terminal-Bench Hard are historical measures.
  - IFBench was removed from the current AA Index.
  - tau2-bench was replaced by tau3-bench Banking.
  - Terminal-Bench Hard was replaced by Terminal-Bench 2.1.

## Source notes

- Artificial Analysis model results: https://artificialanalysis.ai/leaderboards/models
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- GPQA paper: https://arxiv.org/abs/2311.12022
- Humanity's Last Exam paper: https://arxiv.org/abs/2501.14249
- SciCode paper: https://arxiv.org/abs/2407.13168
- tau-bench paper: https://arxiv.org/abs/2406.12045

## Reproducibility files

- Filtered coverage: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/benchmark_coverage_over20.csv`
- Filtered trends: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/benchmark_trends_over20.csv`
- Full matched-score table: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/broad_benchmark_scores_matched.csv`
- Source records: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/provenance.json`
- Analysis script: `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_capability_metric_robustness.py`
