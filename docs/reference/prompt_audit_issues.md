# Prompt Audit — Formalism & Clarity Issues
**Date:** 2026-03-27
**Scope:** `game_environments/item_allocation.py`, `diplomatic_treaty.py`, `co_funding.py`
**Goal:** Reduce coordination failure, remove ambiguity, fix formalisms

---

## Summary Table

| ID | Game | Priority | Title | Status |
|----|------|----------|-------|--------|
| G3-01 | 3 | 🔴 HIGH | Rules: emphasize ALL-OR-NOTHING threshold | proposed |
| G3-02 | 3 | 🔴 HIGH | Rules: add budget scarcity constraint | proposed |
| G3-03 | 3 | 🔴 HIGH | Rules: warn about negative utility from over-contributing | proposed |
| G3-04 | 3 | 🔴 HIGH | Preference assignment: show collective coverage % | proposed |
| G3-05 | 3 | 🔴 HIGH | Preference assignment: add utility formula | proposed |
| G3-06 | 3 | 🔴 HIGH | Pledge prompt: warn that aggregate status is last-round only | proposed |
| G3-07 | 3 | 🟡 MEDIUM | Commit vote: explain unanimous consequence | proposed |
| G3-08 | 3 | 🟡 MEDIUM | Feedback prompt: expand with actionable guidance | proposed |
| G1-01 | 1 | 🟡 MEDIUM | Voting prompt: add utility calculation reminder | proposed |
| G1-02 | 1 | 🟡 MEDIUM | Preference: soften misleading "all items" max utility | proposed |
| G2-01 | 2 | 🟡 MEDIUM | Rules: state no-deal = zero utility for all | proposed |
| G2-02 | 2 | 🟡 MEDIUM | Voting: fix "score/opposed/supportive" terminology | proposed |

---

## Game 3 Issues (HIGH PRIORITY — coordination failure rate is high)

### G3-01: Rules — ALL-OR-NOTHING threshold not emphasized
**File:** `co_funding.py` — `get_game_rules_prompt()`
**Problem:** The current wording "A project is FUNDED if and only if the TOTAL contributions meet or exceed its cost" is technically correct but fails to convey the binary nature. Agents may believe that spreading thin contributions across many projects has proportional value — causing nobody's projects to reach their threshold.
**Fix:** Add an explicit callout that funding is all-or-nothing and that partial contributions to unfunded projects have zero value.

---

### G3-02: Rules — Budget scarcity constraint never communicated
**File:** `co_funding.py` — `get_game_rules_prompt()`
**Problem:** With σ < 1.0, the collective budgets cover strictly less than 100% of total project costs. Agents are never told this. A naive agent trying to fund all projects will fail (mathematically impossible with σ=0.5), yet the rules make it sound like full funding is achievable if everyone cooperates. This single omission is likely the primary driver of coordination failure.
**Fix:** Add a "BUDGET CONSTRAINT" section to the rules stating that the collective budget may not cover all projects and agents must prioritize a subset.

---

### G3-03: Rules — Negative utility risk from over-contributing
**File:** `co_funding.py` — `get_game_rules_prompt()`
**Problem:** Utility = valuation − contribution for each funded project. If an agent contributes more than their valuation to a funded project, their net utility from that project is negative. Nowhere in the rules or preference assignment is this risk communicated. Agents may naively maximize contributions to their top project without realizing diminishing/negative returns.
**Fix:** Add a warning in the YOUR UTILITY section: contributing more than your valuation to any project yields negative utility from that project.

---

### G3-04: Preference assignment — Collective coverage % not shown
**File:** `co_funding.py` — `get_preference_assignment_prompt()`
**Problem:** The preference prompt shows TOTAL BUDGET and TOTAL PROJECT COSTS, but doesn't compute the ratio. An agent who sees `total_budget=73.11` and `total_cost=146.20` must do their own division to realize only 50% of costs are covered collectively. Agents with weaker arithmetic reasoning will miss this and attempt to fund all projects.
**Fix:** Add `**COLLECTIVE COVERAGE:** X% of total project costs` computed inline.

---

### G3-05: Preference assignment — Utility formula never restated
**File:** `co_funding.py` — `get_preference_assignment_prompt()`
**Problem:** The rules explain the utility formula but the preference assignment (which is the prompt where agents first see their actual valuations) doesn't restate it. An agent's first concrete encounter with their valuations has no connection to the utility formula they were told earlier.
**Fix:** Add a "HOW YOUR UTILITY IS COMPUTED" block before the STRATEGIC INSIGHT section.

---

### G3-06: Pledge prompt — Aggregate/funded status is last-round data, not current
**File:** `co_funding.py` — `get_proposal_prompt()`
**Problem:** The pledge submission prompt displays `aggregate=30.00, FUNDED` for projects, but this is last round's state. This round starts with zero contributions. An agent might see "FUNDED" and reduce their contribution to that project, causing it to be unfunded this round. The discussion prompt (in own-mode) has a partial warning about this, but the pledge prompt — where the actual decision is made — has no such warning.
**Fix:** Add a prominent note below the "Currently funded projects" line clarifying this is last-round status.

---

### G3-07: Commit vote — No explanation of what unanimous yay means
**File:** `co_funding.py` — `get_commit_vote_prompt()`
**Problem:** The commit vote prompt says "vote yay if you are willing to finalize this exact profile now / nay if you want another revision round" but does not explain:
- How many yay votes are needed (unanimous = all N agents)
- What happens after the game ends (final funded set locked in, contributions to funded projects paid)
- The strategic cost of voting nay (wastes a revision round)
**Fix:** Add a sentence explaining that ALL participants must vote yay for early termination, otherwise one more revision round occurs.

---

### G3-08: Feedback prompt — Too sparse to be actionable
**File:** `co_funding.py` — `get_feedback_prompt()`
**Problem:** "Consider adjusting your contributions based on these aggregate results." provides no guidance on HOW to adjust. After seeing the round results, an agent needs to know: should I increase to push a near-funded project over? Decrease to avoid over-contributing to an already-funded one? Abandon a far-from-threshold project?
**Fix:** Add project-category guidance (close-to-threshold vs. far-from-threshold vs. already-funded).

---

## Game 1 Issues (MEDIUM PRIORITY)

### G1-01: Voting prompt — No utility calculation reminder
**File:** `item_allocation.py` — `get_voting_prompt()`
**Problem:** Game 2's voting prompt has a full "REMINDER — HOW YOUR UTILITY IS CALCULATED" section. Game 1's voting prompt has nothing — agents vote on an allocation without being reminded how to calculate their utility from it, or what their disagreement utility is (zero).
**Fix:** Add a reminder that utility = sum of preference values for received items, discounted by γ^(round−1), and that no-deal = 0.

---

### G1-02: Preference assignment — "Maximum possible utility" framing encourages greed
**File:** `item_allocation.py` — `get_preference_assignment_prompt()`
**Problem:** "Your maximum possible utility: 28.71 points (if you get ALL items)" is technically accurate but implies that getting all items is the reference point for success. In a multi-agent negotiation, this primes agents toward greedy proposals that will never be unanimously accepted.
**Fix:** Relabel as "Theoretical maximum utility (solo) to remind agents this is a bound, not a realistic target in negotiation.

---

## Game 2 Issues (MEDIUM PRIORITY)

### G2-01: Rules — No-deal outcome not stated
**File:** `diplomatic_treaty.py` — `get_game_rules_prompt()`
**Problem:** Game 1's rules say "No deal means everyone gets zero utility." Game 2's rules do not contain an equivalent statement. Agents might not realize that failing to reach agreement by the final round yields zero utility for all parties — a key incentive for compromise.
**Fix:** Add "If no agreement is reached by the final round, all parties receive zero utility — failing to agree is strictly worse than any positive-utility compromise."

---

### G2-02: Voting prompt — "score/opposed/supportive" terminology is wrong
**File:** `diplomatic_treaty.py` — `get_voting_prompt()`
**Problem:** The voting reminder says "A score of 0.0 means fully opposed; 1.0 means fully supportive on each proposition." But throughout the game, values are consistently described as "rates" (0%–100%), not scores, and 0.0 is the *minimum policy level*, not "opposition." An AI chip export quota of 0.0 means a complete ban — it's an extreme position, not "opposition to the issue." This framing is inconsistent with the actual semantics.
**Fix:** Replace with "A rate of 0.0 means 0% (minimum policy level); 1.0 means 100% (maximum policy level) on each issue."

---

## Open Items (not proposing fixes now)

| ID | Issue | Reason for deferring |
|----|-------|----------------------|
| G3-D1 | Discussion prompt doesn't restate budget | Lower leverage; budget is in preference prompt |
| G3-D2 | "own" transparency mode computed fields are confusing | Hard to fix without redesigning transparency |
| G1-D1 | Fallback assigns all items to proposer (not disclosed) | Intentionally undisclosed; strategic |
| G1-D2 | Proposal prompt doesn't show estimated utility for voter | Requires agent_id → preferences lookup in get_voting_prompt |
| G2-D1 | Position vs. weight conflation in discussion | Agents are sophisticated enough to distinguish |
| G2-D2 | "propositions" vs. "issues" inconsistency in Game 2 rules | Minor; no semantic confusion observed |
