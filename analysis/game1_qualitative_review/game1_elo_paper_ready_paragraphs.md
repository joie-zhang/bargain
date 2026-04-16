# Game 1 Elo Paper-Ready Paragraphs

Across the Elo-conditioned labeled subset (127 runs with recorded Elo; 0 labeled runs had missing Elo and are excluded from the band summaries), the lowest-Elo region looks qualitatively different from the top of the ladder. In the low band (<1250, n=36), openings are dominated by `targeted_anchor` (14/36 (39%)), with another 9/36 (25%) falling into `maximalist_anchor` or `parser_or_degenerate`. Adaptation is most often `rigid_repetition` (13/36 (36%)), and the dominant failure labels are `parser_failure` (20/36 (56%)) and `repetitive_deadlock` (5/36 (14%)). In this same band, the adversary is labeled more stubborn than the baseline in 23/36 (64%) of runs.

The middle band (1250-1399, n=59) already looks more functional. `balanced_tradeoff` and `targeted_anchor` appear at nearly identical rates (24/59 (41%) vs 20/59 (34%)), while `responsive_tradeoff` becomes the modal adaptation style (29/59 (49%)). Most mid-Elo runs receive no failure label (39/59 (66%)), although `repetitive_deadlock` remains visible (14/59 (24%)). Resolution is usually driven by `hybrid_compromise` (30/59 (51%)), and the dominant relative-stubbornness label shifts to `neither` (29/59 (49%)).

At the top of the Elo range (>=1400, n=32), the main change is not simply that models become softer; it is that they become more strategically coherent. `cooperative_exploration` is the single most common opening label (9/32 (28%)), `responsive_tradeoff` remains dominant (21/32 (66%)), and 29/32 (91%) of runs receive no failure label at all. `hybrid_compromise` remains the modal resolution driver (27/32 (84%)), while `neither` dominates relative stubbornness (26/32 (81%)). Taken together, the Elo plots support a capability story about parser reliability and strategic coherence more than a simple monotone stubbornness story: the bottom of the ladder fails disproportionately because negotiations become malformed or cyclic, whereas stronger models more often converge through package-level tradeoffs.

## Figure References

- [opening_style_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/opening_style_by_elo.png)
- [adaptation_style_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/adaptation_style_by_elo.png)
- [failure_mode_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/failure_mode_by_elo.png)
- [resolution_driver_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/resolution_driver_by_elo.png)
- [relative_stubbornness_by_elo.png](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/figures/game1_qualitative_review/relative_stubbornness_by_elo.png)
