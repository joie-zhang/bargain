# E23. Baseline-agent Gini

## Scope and result

- Current Figure 13, `fig:multiagent_homo_adversary`, is baseline-only despite its legacy all-agent filename.
- It reuses 1,300 homogeneous-adversary runs, with 260 per model, 500 Game 1, 400 Game 2, and 400 Game 3.
- The caption states an overall decline, not a strictly monotonic decline.

## Independent verification

I read all 1,300 selected raw result JSONs and standalone configs, checked matching config IDs and baseline counts, and independently recomputed every run's Gini without invoking any producer.

| Elo | Mean corrected baseline Gini | SEM |
| --- | --- | --- |
| 1240 | 0.185663 | 0.014303 |
| 1317 | 0.186891 | 0.014696 |
| 1389 | 0.171558 | 0.013908 |
| 1448 | 0.144479 | 0.012934 |
| 1484 | 0.141851 | 0.013049 |

- Maximum mean difference from the current input summary is 5.55e-17; maximum SEM difference is 9.89e-17.
- Select baseline agents within each run before applying any shift.
- If their minimum payoff is negative, subtract that minimum from every baseline payoff; 94 runs require this shift.
- Compute ordinary Gini as mean pairwise absolute difference divided by twice the mean payoff, then multiply by k/(k-1), where k is the baseline-agent count.
- A singleton or numerically equal vector gives zero; the corrected value is capped at one.
- All 260 n=2 runs therefore give zero baseline Gini.
- All 45 no-agreement runs remain included; there are 377 zero-Gini runs overall.
- Each bar averages 260 run-level values without game reweighting; SEM is sample standard deviation divided by sqrt(260).
- The original metric drops nonfinite inputs, but the independently recomputed raw values were finite.

## Verified chain and history

The production batch launcher constructs the runner CLI. The current shared runtime writes final utilities and embedded configs. The comparison analysis builds the agent table through `build_tables`. The original Gini producer computed both all-agent and baseline-only corrected Gini in four buckets. The dated raw recreation computes five model means. The September 7 label renderer reads that dated summary, executes the current five-bar renderer, and copies the PNG into the current ICLR paper.

- Original June 27 request, implementation, and execution were manually read at /home/jz4391/.codex/sessions/2026/06/26/rollout-2026-06-26T01-43-15-019f0274-2995-7fe0-97a1-80a4f211dc49.jsonl:4548,4561,4567.
- August 23 removal of the right panel and label/style edits were manually read at /home/jz4391/.codex/sessions/2026/08/23/rollout-2026-08-23T07-32-24-01a02e64-a937-7b32-a07d-3d364b555e4b.jsonl:9,144,190.
- The bundled codex-search was run with the exact Gini asset name; original evidence was also located through the prior audit's search script and then opened directly.
- The old four-bucket generator and current five-bar renderer are not interchangeable.
- The current label producer also executes the dated Figure 12 renderer and reads its summary before rendering Gini.

## Needed files

The companion JSON lists concrete script/input paths and precise data selection rules. Its production-directory entry protects only the manifest-selected files and related status/design records; it is not a per-file dependency claim for the whole directory.

## Cleanup decision

No deletion candidates are supported by this result audit. Preserve dated analysis scripts that are current renderer dependencies. Keep original scripts and histories as provenance even when their defaults point to an older paper. The exact historical runtime, provider routes, and recovery records are shared with E13-E15 and require their family-level findings. This audit does not assert that all other files are unused.

