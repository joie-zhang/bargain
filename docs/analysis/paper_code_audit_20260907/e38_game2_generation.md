# E38 Game 2 preference generation

- Scope: current appendix A.3, ideal positions, importance weights, feasibility limits, and five-versus-ten issue settings.
- Current implementation: `game_environments/diplomatic_treaty.py` generates one correlated normal draw per issue, with off-diagonal correlation `2*sin(pi*rho/6)`, then applies the normal CDF.
- Both the config validator and generator reject infeasible negative equicorrelation; the multiagent launcher chooses the feasible endpoint, not clipping arbitrary samples.
- Weights use joint SLSQP on nonnegative simplex vectors; theta=1 copies a Dirichlet vector across agents.
- Two-agent weights allow 15 attempts and summed squared error up to .0001; larger groups allow 30 attempts and error up to .001.
- Actual game-state construction then rounds positions to 1% increments, rounds weights to sum to100, and improves integer weights by up to200 one-point local moves.
- These conversion helpers are needed runtime code, not presentation-only code.
- A3 currently omits integerization; its continuous uniform marginal and exact population correlation describe the pre-rounding construction.
- Rho is not a guaranteed empirical correlation for a realized five- or ten-issue vector.
- Tests cover bounds, PSD feasibility, uniformity, correlation monotonicity, seed reproducibility, integer percentages, pairwise cosine and utility.

## Verified launch and raw records

- The primary config0000 specifies five issues,rho=-1,theta=0,seed42 and is launched through the retained diplomacy batch generator/runner.
- Its stored Agent1 positions are [.31,.26,.59,.06,.68], with complementary Agent2 positions [.69,.74,.41,.94,.32].
- Multiagent-family config1051 specifies n=2 but ten issues,rho=-1,theta=.2,seed184164847.
- Its stored ten-position vectors also have 1% increments and complementary positions.
- The multiagent launcher explicitly fixes n_issues=10, rho in {rho_min(n),.9}, and validates this grid.
- These are checked raw samples, not an all-cohort verification; cohort agents cover all raw configs/results.

## History and preservation

- Bundled codex-search returned June30 session019f1b6c-475f-7a71-9181-ae6c17387dc2; manually read line6547 confirms endpoint selection rather than clipping.
- Local history.jsonl line325 records the April integer-percentage request; September session01a0753c-872c-7e00-88ea-311b7c1c43ec lines20-40 shows direct inspection of these methods for the current conceptual rewrite.
- History is local and incomplete; an original full April implementation session was not found by filename.
- Current source has later commits, so it is not established as the exact executed checkout for every historical run.
- The companion JSON lists24 concrete needed files and evidence, including import-time dependencies that are not themselves Game2 preference algorithms.
- No removal candidates are established by this result.
- No experiments, synthetic data generation, test execution or source changes were performed.

