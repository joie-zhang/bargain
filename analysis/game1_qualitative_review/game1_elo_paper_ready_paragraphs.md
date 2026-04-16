# Game 1 Elo Paper-Ready Paragraphs

Across the Elo-conditioned labeled subset (115 runs with recorded Elo; 22 labeled runs had missing Elo and are excluded from the band summaries), the lowest-Elo region looks qualitatively different from the top of the ladder. In the low band (<1250, n=29), openings are dominated by `targeted_anchor` (11/29 (38%)), with another 7/29 (24%) falling into `maximalist_anchor` or `parser_or_degenerate`. Adaptation is most often `rigid_repetition` (12/29 (41%)), and the dominant failure labels are `parser_failure` (12/29 (41%)) and `repetitive_deadlock` (7/29 (24%)). In this same band, the adversary is labeled more stubborn than the baseline in 16/29 (55%) of runs.

The middle band (1250-1399, n=55) already looks more functional. `balanced_tradeoff` and `targeted_anchor` appear at nearly identical rates (21/55 (38%) vs 19/55 (35%)), while `responsive_tradeoff` becomes the modal adaptation style (29/55 (53%)). Most mid-Elo runs receive no failure label (40/55 (73%)), although `repetitive_deadlock` remains visible (12/55 (22%)). Resolution is usually driven by `hybrid_compromise` (32/55 (58%)), and the dominant relative-stubbornness label shifts to `neither` (32/55 (58%)).

At the top of the Elo range (>=1400, n=31), the main change is not simply that models become softer; it is that they become more strategically coherent. `cooperative_exploration` is the single most common opening label (11/31 (35%)), `responsive_tradeoff` remains dominant (18/31 (58%)), and 28/31 (90%) of runs receive no failure label at all. `hybrid_compromise` remains the modal resolution driver (22/31 (71%)), while `neither` dominates relative stubbornness (26/31 (84%)). Taken together, the Elo plots support a capability story about parser reliability and strategic coherence more than a simple monotone stubbornness story: the bottom of the ladder fails disproportionately because negotiations become malformed or cyclic, whereas stronger models more often converge through package-level tradeoffs.

## Figure References

- [opening_style_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/opening_style_by_elo.png)
- [adaptation_style_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/adaptation_style_by_elo.png)
- [failure_mode_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/failure_mode_by_elo.png)
- [resolution_driver_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/resolution_driver_by_elo.png)
- [relative_stubbornness_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/relative_stubbornness_by_elo.png)
