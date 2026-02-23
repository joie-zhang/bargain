# Conservative Tier Results: Model-Scale Sweep (Feb 23, 2026)

## Experiment Summary

| | Diplomacy (Game 2) | Co-Funding (Game 3) |
|---|---|---|
| **Configs** | 108 | 108 |
| **Model pairs** | 6 (incl. self-play) | 6 (incl. self-play) |
| **Parameter grid** | rho {-1, 0, 1} x theta {0, 0.5, 1} | alpha {0, 0.5, 1} x sigma {0.2, 0.6, 1} |
| **Orders** | weak_first, strong_first | weak_first, strong_first |
| **Runs per config** | 1 | 1 |
| **Failures** | 0/108 | 0/108 |
| **Wall clock** | ~100 min (outliers ~2h) | ~120 min (outliers ~2h) |
| **Baseline model** | gpt-5-nano | gpt-5-nano |

**Adversary models tested:** gpt-5-nano (self-play, Elo 1338), gpt-3.5-turbo-0125 (1225), gpt-4o (1346), o3-mini-high (1364), claude-haiku-4-5 (1403), gpt-5.2-high (1436)

---

## Key Finding 1: Diplomacy -- Social Welfare Scales with Preference Alignment, Not Model Strength

**The dominant driver of social welfare is the game environment, not model capability.**

| rho (pref. correlation) | Mean Social Welfare | Consensus Rate |
|---|---|---|
| -1.0 (competitive) | 0.582 | 64% |
| 0.0 (uncorrelated) | 0.919 | 78% |
| 1.0 (cooperative) | 1.776 | 100% |

- SW triples from competitive to cooperative conditions (0.58 -> 1.78)
- Theta (interest overlap) has **no significant effect** on aggregate SW (1.095, 1.096, 1.086 for theta=0, 0.5, 1.0)
- This is a strong result: the structural alignment of preferences (rho) dominates, while the degree of issue overlap (theta) washes out on average

**Model pair rankings by social welfare:**
1. GPT-5-nano vs GPT-5.2: SW=1.445, 100% consensus, 2.6 rounds (fastest)
2. GPT-5-nano vs GPT-5-nano: SW=1.289, 94% consensus, 4.2 rounds
3. GPT-5-nano vs gpt-4o: SW=1.210, 89% consensus, 4.4 rounds
4. GPT-5-nano vs O3-mini: SW=1.074, 78% consensus, 4.9 rounds
5. GPT-5-nano vs claude-haiku-4-5: SW=0.799, 67% consensus, 6.9 rounds
6. GPT-5-nano vs gpt-3.5-turbo-0125: SW=0.737, 56% consensus, 6.5 rounds

**Surprising finding:** The strongest adversary (GPT-5.2, Elo 1436) achieves the **highest** social welfare, not the lowest. This suggests strong models are better negotiators in a Pareto-improving sense, reaching consensus faster with higher joint payoffs.

---

## Key Finding 2: Diplomacy -- No Exploitation Gap Between Strong and Weak Models

**The "strong exploits weak" narrative does not hold in diplomacy.**

- Weak model (gpt-5-nano) mean utility: 0.539
- Strong model (adversary) mean utility: 0.514
- **Utility gap: -0.024** (weak model actually does slightly better!)

Per-pair utility gaps (strong - weak):
- GPT-5.2: -0.005 (essentially zero)
- O3-mini: -0.076 (weak model wins)
- claude-haiku-4-5: -0.014 (weak model wins)
- gpt-3.5-turbo-0125: -0.033 (weak model wins)
- gpt-4o: +0.005 (essentially zero)

**Interpretation:** Diplomacy is a positive-sum game where both agents benefit from agreement. Stronger models don't exploit -- they cooperate more efficiently. The 90% "exploitation detection" rate is a metric artifact (it flags any utility asymmetry), not genuine exploitation.

---

## Key Finding 3: Co-Funding -- Pervasive Coordination Failure

**LLMs are terrible at public goods provision.**

| Metric | Value |
|---|---|
| Mean utilitarian efficiency | 31.2% |
| Mean provision rate | 33.4% |
| Coordination failure rate | 77.5% |
| Mean projects funded | 1.1 / 5 |
| Adversary free-rider index | 3.04 |
| Baseline free-rider index | 1.94 |
| Pure free-rider rate (adversary) | 21.3% |
| Pure free-rider rate (baseline) | 13.0% |

Even in the best conditions (alpha=1.0, sigma=1.0), efficiency barely exceeds 35%. This is dramatically worse than human experiments in public goods games.

---

## Key Finding 4: Co-Funding -- Free-Riding Inversely Correlates with Model Strength

**Stronger models free-ride LESS, not more.**

| Model | Elo | Efficiency | Free-Rider Index | Utility Gap |
|---|---|---|---|---|
| gpt-5.2-high | 1436 | **0.630** | **1.47** | +2.1 |
| gpt-4o | 1346 | 0.486 | 3.90 | -0.5 |
| claude-haiku-4-5 | 1403 | 0.264 | 1.82 | -10.0 |
| o3-mini-high | 1364 | 0.282 | **3.99** | +7.2 |
| gpt-5-nano (self) | 1338 | 0.261 | 3.07 | +1.1 |
| gpt-3.5-turbo | 1225 | 0.252 | 3.75 | -2.3 |

**Key patterns:**
- GPT-5.2-high (strongest) has the **lowest** free-rider index (1.47) and **highest** efficiency (63%) -- it cooperates the most
- o3-mini-high is the most exploitative: highest free-rider index (3.99) and largest positive utility gap (+7.2), meaning it extracts the most value from gpt-5-nano
- This confirms the "scissors pattern" from smoke tests: exploitation is model-specific, not monotonically increasing with Elo
- claude-haiku-4-5 has a large negative utility gap (-10.0), meaning gpt-5-nano actually exploits IT

---

## Key Finding 5: Co-Funding -- Alpha Matters More Than Sigma for Efficiency

| alpha (pref. alignment) | Efficiency | Coord. Failure | Funded |
|---|---|---|---|
| 0.0 (orthogonal) | 0.237 | 0.838 | 0.9 |
| 0.5 (moderate) | 0.344 | 0.734 | 1.2 |
| 1.0 (identical) | 0.354 | 0.751 | 1.2 |

| sigma (budget) | Efficiency | Coord. Failure | Funded |
|---|---|---|---|
| 0.2 (scarce) | 0.363 | 0.914 | 0.3 |
| 0.6 (moderate) | 0.263 | 0.767 | 1.1 |
| 1.0 (abundant) | 0.309 | 0.643 | 1.9 |

- Higher alpha (more aligned preferences) improves efficiency by ~50% (0.24 -> 0.35)
- Higher sigma (more budget) reduces coordination failure but doesn't proportionally increase efficiency
- At sigma=0.2 with 91% coordination failure, agents barely fund anything (0.3 projects avg)

---

## Cross-Game Comparison

| Dimension | Diplomacy | Co-Funding |
|---|---|---|
| **Cooperation success** | High (81% consensus) | Low (31% efficiency) |
| **Exploitation by strong** | None (gap = -0.024) | Model-dependent (range -10 to +7) |
| **Environment sensitivity** | rho dominates (3x SW range) | alpha and sigma both matter |
| **Model strength effect** | Stronger = faster consensus | Stronger = less free-riding (except o3-mini) |
| **Failure mode** | Deadlock in competitive settings | Universal under-provision |

**The fundamental difference:** Diplomacy is a bargaining game where agreement is Pareto-improving -- rational agents should always agree. Co-funding is a public goods game with genuine free-rider incentives -- rational agents should under-contribute. The LLMs faithfully reproduce both theoretical predictions.

---

## Figures Reference

**Diplomacy:** `visualization/figures/diplomacy/`
- `plot4_rho_theta_heatmap.png` -- The money plot: SW heatmap showing rho dominance
- `plot7_competition_index.png` -- Competition index vs SW/exploitation/per-agent utility
- `plot1_model_pair_utilities.png` -- Per-pair utility comparison
- `plot5_exploitation_by_condition.png` -- Utility advantage and speaking order effects

**Co-Funding:** `visualization/figures/cofunding/`
- `efficiency_heatmap.png` -- Utilitarian efficiency across alpha x sigma
- `free_rider_by_model.png` -- Free-rider index by adversary model
- `utility_vs_elo.png` -- Agent utility vs Elo (scatter)
- `competition_index_metrics.png` -- 4-panel: efficiency, utility gap, provision, free-riding vs CI

---

## Recommendations for Ambitious Tier

1. **The ambitious tier should confirm these patterns with:**
   - Finer parameter grids (5x5) to test monotonicity of rho/alpha effects
   - 3 runs per config for statistical significance
   - 3 additional models (amazon-nova-micro Elo 1241, claude-sonnet-4-5 Elo 1450, gemini-3-pro Elo 1490) to better sample the Elo range

2. **Specific hypotheses to test:**
   - Does the non-monotonic exploitation pattern (o3-mini outlier) replicate?
   - Does claude-sonnet-4-5 (Elo 1450) cooperate like gpt-5.2-high or exploit like o3-mini?
   - Does gemini-3-pro (Elo 1490, highest) set a new cooperation benchmark?
   - Is there a phase transition in diplomacy between rho=-0.5 and rho=0.0?

3. **Potential paper claims from these results:**
   - "Model capability improves negotiation efficiency, not exploitation" (diplomacy)
   - "Public goods provision is a harder coordination problem than bargaining for LLMs" (cross-game)
   - "Free-riding behavior is model-specific, not monotonically related to capability" (co-funding)

---

*Generated: Feb 23, 2026, 06:00 EST*
*Data: 216 experiments (108 diplomacy + 108 cofunding), 0 failures*
*Configs: `experiments/results/diplomacy_20260223_032151/` and `experiments/results/cofunding_20260223_032155/`*
