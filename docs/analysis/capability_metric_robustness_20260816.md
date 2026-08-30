# Bilateral payoff trends with alternative capability metrics

## Question

Does the positive relation between adversary capability and adversary payoff in Figure 2 remain when capability is measured with public benchmarks instead of LM Arena Elo?

## Short answer

- The direction is stable across the four academic metrics with at least 10 matched models.
  - All 12 benchmark-by-game OLS slopes are positive.
  - Five of the 12 slopes remain significant at a 5% false-discovery-rate threshold across the 12 academic tests.
- MMLU-Pro gives the clearest result.
  - It covers 17 of the 30 models.
  - Its slope is positive in all three games.
  - All three confidence intervals exclude zero after the multiple-test correction.
- LiveBench, HELM Lite, and original MMLU give less consistent statistical support.
  - Their samples contain only 10 or 11 models.
  - Their same-subset Arena Elo fits are also weak in several cells.
  - Small and selected samples explain part of this instability.
- The paper can state that the positive direction reproduces with several capability measures.
- The paper should not state that every benchmark gives a statistically reliable trend in every game.

## Figure 2 estimand

- The source is the 1,500-run GPT-5-nano bilateral cohort used by Figure 2.
- Each point is one adversary model's mean round-discounted payoff within one game.
- Error bars show the standard error over that model's runs.
- Dashed lines are unweighted OLS fits over model-level means.
- Each alternative plot includes only models that have a matched score for its x-axis metric.

## Candidate benchmark audit

| Candidate | Published matches | Direct scores | Configuration proxies | Plot? | Assessment |
|---|---:|---:|---:|---|---|
| LM Arena Elo | 30 | 30 | 0 | Existing Figure 2 | Full roster and behavior-based, but it is the metric being tested. |
| Artificial Analysis Intelligence Index v4.1.1 | 29 | 7 | 5 | Diagnostic only | Wide nominal coverage, but 22 scores are marked as estimates rather than direct v4.1.1 evaluations. |
| MMLU-Pro | 17 | 17 | 2 | Yes | Best balance of recognition, difficulty, breadth, and roster coverage. |
| LiveBench overall | 11 | 11 | 0 | Yes | Strong current design with objective grading and fixed releases, but this release has a small and older model subset. |
| HELM Lite mean win rate | 10 | 10 | 0 | Yes | Standardized multi-scenario suite, but mean win rate depends on the evaluated model pool. |
| MMLU through HELM Lite | 10 | 10 | 0 | Yes | Very recognizable and standardized here, but MMLU is older and near saturation for strong models. |
| Humanity's Last Exam | 7 | 7 | 3 | No | High current relevance, but the matched sample is too small and uses several nearby reasoning configurations. |
| BIG-Bench Hard through Open LLM Leaderboard 2 | 6 | 6 | 0 | No | Well-known reasoning set, but public exact-config coverage is too small. |
| GPQA through Open LLM Leaderboard 2 | 6 | 6 | 0 | No | Well-known hard science benchmark, but public exact-config coverage is too small. |
| Open LLM Leaderboard 2 composite | 6 | 6 | 0 | No | Standardized composite, but it covers only open-weight models in this roster. |

- A direct score means that the source presents a measured benchmark score rather than an estimated score.
- A configuration proxy uses the same model family but a nearby reasoning or inference setting.
- Name and provider aliases that preserve the named model weights are counted as matches.
- The exact match table is in `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/benchmark_scores_matched.csv`.

## Trend results

The slope unit below is payoff change for one sample standard deviation of the benchmark score.

| Benchmark | Game | n | Slope | 95% CI | R² | BH-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| MMLU-Pro | Item allocation | 17 | +7.24 | [3.46, 11.03] | 0.526 | 0.0039 |
| MMLU-Pro | Diplomatic Treaty | 17 | +9.41 | [5.96, 12.85] | 0.693 | 0.00020 |
| MMLU-Pro | Co-funding | 17 | +7.60 | [5.14, 10.06] | 0.743 | 0.00010 |
| LiveBench | Item allocation | 11 | +3.79 | [-0.45, 8.03] | 0.312 | 0.111 |
| LiveBench | Diplomatic Treaty | 11 | +0.48 | [-2.57, 3.54] | 0.014 | 0.729 |
| LiveBench | Co-funding | 11 | +7.58 | [1.79, 13.37] | 0.493 | 0.038 |
| HELM Lite | Item allocation | 10 | +1.18 | [-3.33, 5.68] | 0.043 | 0.614 |
| HELM Lite | Diplomatic Treaty | 10 | +2.56 | [0.68, 4.45] | 0.551 | 0.038 |
| HELM Lite | Co-funding | 10 | +6.31 | [-0.29, 12.91] | 0.378 | 0.100 |
| MMLU through HELM | Item allocation | 10 | +2.36 | [-1.83, 6.54] | 0.174 | 0.277 |
| MMLU through HELM | Diplomatic Treaty | 10 | +1.61 | [-0.87, 4.10] | 0.218 | 0.231 |
| MMLU through HELM | Co-funding | 10 | +6.68 | [0.34, 13.03] | 0.424 | 0.083 |

- Benjamini-Hochberg adjustment controls the false discovery rate across the 12 academic benchmark-by-game tests.
- The alternative scores remain close to Arena Elo within their matched samples.
  - MMLU-Pro has Spearman rho 0.949 with Arena Elo.
  - LiveBench has rho 0.818.
  - HELM Lite has rho 0.842.
  - MMLU through HELM has rho 0.782.
- These plots test measurement sensitivity.
- They do not show that benchmark skill and Arena preference are separate causes of negotiation payoff.

## MMLU-Pro sensitivity checks

- The public MMLU-Pro table mixes TIGER-Lab evaluations and self-reported results.
- Two matches use nearby thinking configurations.
- Removing the two configuration proxies leaves 15 models.
  - The standardized slopes remain positive in all games at +6.77, +9.09, and +7.51.
  - Their unadjusted p-values are 0.0049, 0.00026, and 0.000076.
- Keeping only the nine TIGER-Lab-scored rows also leaves positive slopes.
  - The slopes are +8.20, +10.75, and +7.28.
  - Their unadjusted p-values are 0.0049, 0.0086, and 0.0040.
- The small nine-model check is a sensitivity analysis, not a replacement for a single-protocol evaluation.

## Pros and cons

### MMLU-Pro

- Pros
  - It has the highest academic coverage at 17 models.
  - It is multi-domain and harder than original MMLU.
  - It reproduces the positive trend in all three games.
- Cons
  - The public table mixes evaluator and prompting provenance.
  - Two frontier matches use nearby thinking configurations.

### LiveBench

- Pros
  - It uses objective ground-truth grading.
  - Its fixed releases make the score snapshot reproducible.
  - Its update process aims to reduce contamination.
- Cons
  - The 2024-11-25 release matches only 11 models.
  - It does not cover most 2025 and 2026 frontier configurations in the paper roster.

### HELM Lite

- Pros
  - It evaluates several scenarios under one framework.
  - It provides both a suite score and original MMLU under the same run protocol.
- Cons
  - It matches only 10 models.
  - Mean win rate is relative to the comparison pool and release.

### Artificial Analysis Intelligence Index v4.1.1

- Pros
  - It has a published value for 29 of 30 models.
  - It covers recent closed models that academic leaderboards often omit.
- Cons
  - Only seven of the matched v4.1 scores are direct evaluations.
  - The remaining 22 scores are estimates.
  - The composite definition changes by index version.

## Recommendation for the paper

- Use MMLU-Pro as the primary alternative x-axis.
- Put LiveBench overall and HELM Lite in the appendix as directional checks.
- Include original MMLU only as a recognition check, not as the strongest current measure.
- Do not use the Artificial Analysis composite as primary evidence with the current data.
- Do not plot HLE, GPQA, BBH, or the Open LLM composite until at least 10 exact or clearly matched configurations are available.
- Use this claim in the paper.
  - "Across four public academic capability measures, all 12 game-specific slope estimates are positive. MMLU-Pro, which covers 17 of 30 adversaries, yields positive adjusted associations in all three games. The smaller 10 to 11 model subsets give mixed statistical support."
- Avoid a claim that all benchmarks independently confirm a universal scaling law.

## Public sources

- MMLU introduced a 57-task multi-domain test: https://arxiv.org/abs/2009.03300
- MMLU-Pro expanded difficulty and answer options: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ad236edc564f3e3156e1f7b8b8284c08-Abstract-Datasets_and_Benchmarks_Track.html
- LiveBench describes its objective grading, six categories, fixed releases, and update policy: https://github.com/LiveBench/livebench
- HELM Lite publishes standardized scenario results: https://crfm.stanford.edu/helm/lite/latest/
- Humanity's Last Exam is a broad expert-level benchmark: https://arxiv.org/abs/2501.14249
- GPQA is a hard graduate-level science benchmark: https://arxiv.org/abs/2311.12022
- BIG-Bench Hard is a selected set of difficult BIG-Bench tasks: https://arxiv.org/abs/2210.09261
- Open LLM Leaderboard 2 publishes a six-task composite for open models: https://huggingface.co/open-llm-leaderboard
- Artificial Analysis documents its versioned intelligence index: https://artificialanalysis.ai/data-api/docs

## Reproducibility files

- Combined plot: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/plots/bilateral_payoff_by_alternative_capability_metrics.png`
- Individual plots: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/plots/`
- Candidate coverage: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/candidate_benchmark_coverage.csv`
- Matched model scores: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/benchmark_scores_matched.csv`
- Trend statistics: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/trend_summary.csv`
- MMLU-Pro sensitivity checks: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/mmlu_pro_sensitivity.csv`
- Source snapshots and access records: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/capability_metric_robustness_20260816/provenance.json`
- Analysis script: `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_capability_metric_robustness.py`
