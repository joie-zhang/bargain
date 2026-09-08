# E42 shared metric and solver dependency audit

## Scope and decision

- Current paper methods at appendix.tex:242-337 define variance, Gini, NBS, Lindahl and social welfare optima.
- E19, E20 and E21 trace the fairness result cohorts and figure chains; this report audits their shared solver implementations.
- Keep all 26 concrete files in the companion JSON.
- No deletion candidate was established.
- Similar function names do not establish interchangeable implementations.

## Verified execution and analysis chain

- Agreement execution in strong_models_experiment/phases/phase_handlers.py:4134 calls the selected game's calculate_utility.
- Item allocation sums allocated-item values and discounts by gamma^(round-1).
- Treaty utility sums weight times one minus absolute distance, scales to100, then discounts.
- Co-funding subtracts payments only for funded projects and optionally discounts according to enable_time_discount.
- scripts/analyze_n2_baseline_comparison.py imports treaty welfare and co-funding benchmark functions from game_environments.
- scripts/analyze_nash_lindahl_fairness.py independently rebuilds raw utilities and benchmark utilities.
- The shared analyzer records saved discounted utilities separately from raw utility, raw fairness residual and raw efficiency.
- scripts/paper_figures/plot_fairshare_residual_combined.py imports both analyzer families and explicitly selects lindahl_residual for Game3 versus nbs_residual otherwise.
- scripts/analyze_n2_plus_multiagent_comparison.py imports the shared fairness analyzer and the multiagent raw table builder.
- Result-chain agents must preserve the exact selected saved data and benchmark caches, because current recomputation can use corrected methods.

## Important implementation distinctions

| Quantity | Implementation distinction | Cleanup consequence |
|---|---|---|
| Game1 welfare | Itemwise maximum is exact and distinct from allocation NBS. | Keep both. |
| Game1 bilateral NBS | Baseline analyzer enumerates32 allocations and maximizes the product; shared analyzer enumerates assignments and uses log(max(u,0)+EPS). | Tie and zero handling can differ. |
| Game1 multiagent NBS | Nine deterministic/seeded starts and single-item local search, at most100passes per start. | Approximation is not an exact global solver. |
| Game2 welfare | Both runtime metrics and shared analyzer use weighted medians. | Both remain required through separate callers. |
| Game2 NBS | Runtime metrics uses20 RandomState(42) L-BFGS-B starts; baseline fast helper uses5 random starts; shared bilateral helper uses3 structured and2 random starts. | Keep all current dependencies; outputs need not be bit-identical. |
| Game2 multiagent NBS | Five-start SLSQP, coordinate refinement and numerical subgradient check. | Keep correction and regression tests. |
| Game3 optimum | Runtime helper brute-forces up to20projects and uses cents-DP above20; shared analyzer uses integer-rounded costs and integer-budget DP throughout. | Different assumptions, not duplicate code. |
| Game3 fair share | Shared analyzer computes both realized-funded-set Lindahl utility and a separate feasible-set reference. | Preserve distinction between cost sharing and global benchmark. |
| Game3 bilateral reference | Enumerates funded sets, checks group budget and each agent's proportional-payment budget, maximizes smoothed log product. | Not equivalent to welfare optimum. |
| Game3 multiagent reference | Uses proportional sharing on welfare-optimal funded set without an individual-budget feasibility check. | Its own method label calls it a proxy. |
| Gini | game_environments/multiagent_metrics.py shifts negative utilities but does not finite-sample-correct. | Existing visualization imports it. |
| Gini | Shared fairness analyzer shifts negatives and applies n/(n-1), bounded to[0,1]. | Current corrected-Gini tests cover this variant. |
| Gini | Multiagent comparison shifted_gini does not apply that correction; raw table builder's gini does not shift negatives and applies a separate count threshold correction. | Do not replace or remove by name matching. |
| Variance | Raw table builder uses np.var on saved final utilities, population variance. | Discounting is already in saved utilities. |
| Efficiency | Runtime co-funding helper clips ratio to[0,1]; shared analyzer does not clip positive-optimum ratios. | Negative-payoff behavior differs. |

- scripts/analyze_nash_lindahl_fairness.py:686 excludes refunded pledges from executed-payment distance.
- Its no-consensus empty funded set has zero Lindahl payment distance by construction.
- Its Game3 realized-set residual differs from its enumerated/optimal-set NBS residual.
- None of these definitions should be silently interchanged during cleanup.
- The paper's finite-sample formula has uppercase N and lowercase n; no paper changes were made.

## History and version evidence

- Used the bundled codex-search script with query "Nash Lindahl optimal welfare solver gini".
- Manually verified session019f1d1f-86bb-7711-a818-ad5adaaa9211 at line8221.
- The user requested the cooperative/competitive joint-role extension there, which explains why overall and endpoint benchmark consumers coexist.
- Full source is /home/jz4391/.codex/sessions/2026/07/01/rollout-2026-07-01T06-00-33-019f1d1f-86bb-7711-a818-ad5adaaa9211.jsonl.
- This session is a fork; it is evidence of the conversation, not proof that it is the original authoring session.
- /home/jz4391/.codex/history.jsonl:4151 records the later request to revise metric prose.
- Git history identifies4ef9d4f, dated2026-05-07, as adding fairness report builders, and444fcf9, dated2026-08-30, as a later change affecting these files.
- Historical report reproduction_audit/fig32_fairness_inequality_efficiency/report.md:102 describes the former weighted-compromise Game2 proxy.
- That report must not be treated as a specification of the current corrected numerical solver.

## Validation

- Ran PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider against test_metrics, test_cofunding_metrics, test_corrected_gini, test_issue30_game2_nbs and test_issue36_refunded_pledges.
- All65tests passed in6.69seconds.
- These are regression/analytic tests, not generated experimental evidence.
- The check did not rerun paid experiments, write paper outputs or alter code.

## Remaining limits

- Original source versions for every historical benchmark cache are not established by this shared audit alone.
- Broader caches and data selections belong to E19/E20/E21 and the welfare audit.
- The26-file list is a concrete verified dependency/support list, not a claim that everything else is unused.
- No code, paper, configuration or raw data was changed.

