# Formalism vs. Implementation Discrepancy Audit
**Date:** 2026-03-27
**Source document:** `approach_mar_27.tex`

---

## Fixed Discrepancies

### C1 — Budget formula (`co_funding.py`)
**Paper:** `b^i = σ × C / n` where `C = Σ c_j`
**Old code:** `budget_ratio = 0.5 + 0.5 * sigma` → range [0.5, 1.0] of total cost
**Fixed code:** `budget_ratio = sigma` → range (0, 1.0] of total cost
**Impact:** σ=0.2 now correctly means agents can collectively fund 20% of total cost (not 60%). All existing experiment results with σ < 1.0 are affected.

### C2 — Time discount default (`base.py`)
**Paper:** No discount factor specified for Game 3 (co-funding).
**Old code:** `enable_time_discount: bool = True`
**Fixed code:** `enable_time_discount: bool = False`
**Impact:** Round-5 utility now equals round-1 utility for the same funded set. Games 1 and 2 are unaffected (they override this default in their own configs if needed).

### C3 — NBS utility scale (`metrics.py`)
**Paper:** Utility `U_i ∈ [0, 100]` (matches `compute_utility` formula with `* 100.0`)
**Old code:** `nash_bargaining_solution()` returned values in `[0, 1]`
**Fixed code:** Added `* 100.0` to NBS utility computation.
**Impact:** Exploitation index (`actual_utility / nbs_utility`) was 100× too large. Any existing exploitation metrics computed against NBS are invalid and must be recomputed.

---

## Open Items (not fixing now)

| ID | Location | Issue | Status |
|----|----------|-------|--------|
| S1 | `phase_handlers.py` | Commit vote phase not described in paper protocol | Keep code; update paper later |
| S2 | `cofunding_metrics.py` | Underfunding rate does not check budget feasibility before labeling coordination failure | Not fixing now |
| S3 | `item_allocation.py` | Game 1 valuations: paper implies sum=100 per agent, not verified in code | Revisit later |
| M1 | `metrics.py` | Variable names `positions`/`weights` vs. paper's `p_i`/`w_i` notation | Minor, no fix needed |
| M2 | `metrics.py` | NBS convergence tolerance 1e-12 vs. paper's unspecified tolerance | Minor |
| M3 | `base.py` | `alpha` described as "preference alignment" but implemented as cosine-similarity target | Minor naming issue |
| M4 | `diplomatic_treaty.py` | Gaussian copula correlation vs. paper's `alpha` definition | Revisit when writing Game 2 section |

---

## Verification Checklist (post-fix)

- [ ] Game 3 with σ=0.2: confirm `total_budget / total_cost ≈ 0.20`
- [ ] Game 3 multi-round: confirm round-5 utility equals round-1 utility for same funded set (no discounting)
- [ ] Game 2: confirm `nash_bargaining_solution()` returns values in [0, 100]
