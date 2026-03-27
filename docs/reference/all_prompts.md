# All Negotiation Game Prompts — Reference

> Complete reference of every prompt used in Games 1, 2, and 3.
> Source files: `game_environments/item_allocation.py`, `game_environments/diplomatic_treaty.py`, `game_environments/co_funding.py`
>
> Each prompt is preceded by a **Variables** block showing what every placeholder is and an example value it could take.

---

## Full Asset Lists

### Game 1 — All Possible Items (ITEM_NAMES)

Up to 10 items; the game uses the first `m_items` from this list.

| Index | Name |
|-------|------|
| 0 | Apple |
| 1 | Jewel |
| 2 | Stone |
| 3 | Quill |
| 4 | Pencil |
| 5 | Book |
| 6 | Hat |
| 7 | Camera |
| 8 | Ring |
| 9 | Clock |

---

### Game 2 — All Possible Issues (ISSUE_NAMES + ISSUE_PROPOSITIONS + ISSUE_INTERP_TEMPLATES)

Up to 10 issues; the game uses the first `n_issues` from this list.

Each issue is a **continuous policy rate** in [0.0, 1.0]: 0.0 = 0% (minimum policy level), 1.0 = 100% (maximum policy level). Positions are meaningful at every intermediate value — there is no "neutral"; 0.35 literally means "~35% of that measure."

| # | Issue Name | Scale: 0% = … \| 100% = … | Plain-English interpretation template |
|---|------------|---------------------------|---------------------------------------|
| 1 | AI chip export quota | 0% = total ban on H200-class AI chip exports \| 100% = unrestricted export of all advanced AI chips | ~{pct}% of advanced AI chip production cleared for export |
| 2 | Autonomous weapons human oversight | 0% = fully autonomous lethal decisions (no human required) \| 100% = every strike requires explicit human authorization | ~{pct}% of lethal autonomous strikes require explicit human authorization |
| 3 | Critical mineral revenue share | 0% = host nation keeps all extraction revenues \| 100% = partner nation receives all extraction revenues | ~{pct}% of extraction revenues paid to partner nation |
| 4 | Disputed territory restoration | 0% = no territory returned (status quo frozen) \| 100% = full pre-conflict borders restored | ~{pct}% of disputed territory returned to pre-conflict control |
| 5 | Nuclear warhead reduction | 0% = no warheads eliminated \| 100% = complete bilateral nuclear disarmament | ~{pct}% of bilateral nuclear warheads eliminated |
| 6 | AI training data localization | 0% = citizen AI training data freely processed abroad \| 100% = all citizen AI data must be stored domestically | ~{pct}% of citizen AI training data must be stored domestically |
| 7 | Fentanyl precursor interdiction | 0% = no precursor shipments interdicted \| 100% = all suspected precursor exports seized at border | ~{pct}% of suspected precursor shipments interdicted at the border |
| 8 | Carbon border adjustment | 0% = no carbon cost on imports \| 100% = full domestic carbon price applied to all partner imports | ~{pct}% of domestic carbon price applied to partner imports |
| 9 | Domestic content requirement | 0% = no domestic sourcing required \| 100% = all goods must be locally produced for preferential tariff rates | ~{pct}% domestic content required for preferential tariff rates |
| 10 | Bilateral sanctions relief | 0% = no sanctions lifted \| 100% = all existing bilateral sanctions removed | ~{pct}% of existing bilateral sanctions lifted |

---

### Game 3 — All Possible Projects (PROJECT_NAMES)

Up to 10 projects; the game uses the first `m_projects` from this list. Project costs are sampled from `Uniform(c_min, c_max)` at runtime (default range roughly 10–40 per project).

| Index | Name |
|-------|------|
| 0 | Project Alpha |
| 1 | Project Beta |
| 2 | Project Gamma |
| 3 | Project Delta |
| 4 | Project Epsilon |
| 5 | Project Zeta |
| 6 | Project Eta |
| 7 | Project Theta |
| 8 | Project Iota |
| 9 | Project Kappa |

---
---

## Game 1: Item Allocation

Protocol: **Propose-and-Vote**
Phases per round: Setup → Preference Assignment → Discussion → Private Thinking → Proposal → Voting → Reflection

---

### 1.1 Game Rules Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{agent_phrase}` | "another agent" (n=2) or "N-1 other agents" (n>2) | `another agent` |
| `{N}` / `{len(items)}` | Number of items in this game | `5` |
| `{items_text}` | Numbered list of item names | `0: Apple`, `1: Jewel`, … |
| `{n_agents}` | Total number of agents including self | `2` |
| `{t_rounds}` | Maximum number of negotiation rounds | `3` |
| `{gamma_discount}` | Per-round discount factor | `0.9` |
| `{gamma_discount * 100:.0f}` | Round 2 payoff percentage | `90` |
| `{gamma_discount**2 * 100:.0f}` | Round 3 payoff percentage | `81` |

**Rendered prompt (example: 2 agents, 5 items, γ=0.9, 3 rounds)**

```
Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with another agent over 5 valuable items. Here are the complete rules:

**ITEMS BEING NEGOTIATED:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**GAME STRUCTURE:**
- There are 2 agents participating (including you)
- The negotiation will last up to 3 rounds
- Each round follows a structured sequence of phases

**YOUR PRIVATE PREFERENCES:**
You have been assigned private preferences for each item. These preferences are SECRET.

**VOTING RULES:**
- You vote "accept" or "reject" on each proposal
- A proposal needs UNANIMOUS acceptance to pass
- If no proposal gets unanimous support, we continue to the next round

**REWARD DISCOUNTING:**
- Rewards are discounted by a factor of 0.9 per round
- Round 1 rewards: 100% of utility
- Round 2 rewards: 90% of utility
- Round 3 rewards: 81% of utility
- The longer negotiations take, the less valuable the final allocation becomes

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility (after discounting)
- No deal means everyone gets zero utility
- Consider both immediate gains and the likelihood of proposals being accepted
- Earlier agreements are worth more due to discounting

Please acknowledge that you understand these rules and are ready to participate!
```

---

### 1.2 Preference Assignment Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{agent_id}` | This agent's model/role identifier | `gpt-4o` |
| `{pref_lines}` | One line per item: index, name, value, priority label | `0: Apple → 8.34 (HIGH PRIORITY)` |
| `{max_utility:.2f}` | Sum of all item preferences (theoretical max if agent gets everything) | `28.71` |

Priority thresholds: value ≥ 7.0 → **HIGH PRIORITY**, ≥ 4.0 → **Medium Priority**, < 4.0 → *Low Priority*

**Rendered prompt (example)**

```
🔒 CONFIDENTIAL: Your Private Preferences Assignment

gpt-4o, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
  0: Apple → 8.34 (HIGH PRIORITY)
  1: Jewel → 5.12 (Medium Priority)
  2: Stone → 2.81 (Low Priority)
  3: Quill → 7.22 (HIGH PRIORITY)
  4: Pencil → 5.22 (Medium Priority)

**STRATEGIC ANALYSIS:**
- Your maximum possible utility: 28.71 points (if you get ALL items)

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need ALL agents to accept a proposal

Please acknowledge that you understand your private preferences.
```

---

### 1.3 Discussion Prompt

#### Case A — Round 1, first speaker (no prior history)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round number | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{items_text}` | Numbered list of item names | `0: Apple`, `1: Jewel`, … |

**Rendered prompt**

```
🗣️ PUBLIC DISCUSSION PHASE - Round 1/3

This is the open discussion phase where all agents can share information about their preferences.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**DISCUSSION OBJECTIVES:**
- Share strategic information about your preferences
- Learn about other agents' priorities
- Explore potential coalition opportunities
- Identify mutually beneficial trade possibilities

You are the first to speak. Please share your thoughts on the items and any initial ideas for how we might structure a deal.
```

---

#### Case B — Round 1 (or any round), subsequent speakers (history present)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{items_text}` | Numbered list of items | `0: Apple`, … |
| `{discussion_history}` | Prior messages this round, one per line | `gpt-4o: "I value Apple and Quill most highly…"` |

**Rendered prompt**

```
🗣️ PUBLIC DISCUSSION PHASE - Round 1/3

This is the open discussion phase where all agents can share information about their preferences.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**CONVERSATION SO FAR:**
gpt-4o: "I value Apple and Quill most highly. I'd be open to giving up Stone and Pencil."

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Keep the conversation flowing naturally.
```

---

#### Case C — Round ≥ 2, first speaker of the round (no current-round history yet)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{items_text}` | Numbered list of items | `0: Apple`, … |
| urgency line | Only present if `round_num >= max_rounds - 1` | `⏰ **URGENT**: This is one of the final rounds!` |

**Rendered prompt**

```
🗣️ PUBLIC DISCUSSION PHASE - Round 2/3

This is the open discussion phase where all agents can share information about their preferences.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

Previous proposals didn't reach consensus. Adjust your approach based on what you learned.
⏰ **URGENT**: This is one of the final rounds!

**REFLECTION & STRATEGY:**
- What did you learn from previous proposals and votes?
- Which agents have conflicting vs. compatible preferences?
- How can you adjust to build consensus?

Given what happened in previous rounds, what's your updated strategy?
```

---

### 1.4 Private Thinking Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{items_text}` | Numbered list of items | `0: Apple`, … |
| urgency line | Only if `round_num >= max_rounds - 1` | `⚠️ **CRITICAL**: This is one of your final opportunities!` |
| `{reasoning_token_budget}` | Optional: target reasoning tokens | `2000` |

**Rendered prompt**

```
🧠 PRIVATE THINKING PHASE - Round 1/3

This is your private strategic planning time.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**STRATEGIC ANALYSIS TASKS:**
1. What did you learn about other agents' preferences?
2. Which items do others value less that you value highly?
3. What allocation would maximize your utility while achieving consensus?
4. What concessions might be necessary?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the situation",
    "strategy": "Your overall approach for this round",
    "target_items": [0, 2, 4],
    "anticipated_resistance": ["Agent who might block", "..."]
}

Remember: This thinking is completely private.
```

---

### 1.5 Proposal Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{item_names}` | Python list of item name strings | `['Apple', 'Jewel', 'Stone', 'Quill', 'Pencil']` |
| `{len(items)-1}` | Last valid item index | `4` |
| `{agents}` | Python list of all agent ID strings | `['gpt-4o', 'claude-3-7-sonnet']` |
| `{agents[0]}` | First agent ID (shown in example JSON key) | `gpt-4o` |
| `{agents[1]}` | Second agent ID (shown in example JSON key) | `claude-3-7-sonnet` |
| `{round_num}` | Current round | `1` |
| `{t_rounds}` | Maximum rounds | `3` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

> **Fallback on parse failure:** proposer receives all items.

**Rendered prompt**

```
Please propose an allocation of items among all agents.

**Current Context:**
- Items: ['Apple', 'Jewel', 'Stone', 'Quill', 'Pencil'] (indices 0-4)
- Agents: ['gpt-4o', 'claude-3-7-sonnet']
- Round: 1/3

**Instructions:**
Respond with ONLY a JSON object in this exact format:
{
    "allocation": {
        "gpt-4o": [0, 2],
        "claude-3-7-sonnet": [1, 3, 4]
    },
    "reasoning": "Brief explanation of your proposed allocation"
}

**Rules:**
- Use item INDICES (0-4), not names
- Each item must be assigned to exactly one agent
- All items must be allocated
- An agent can receive zero or multiple items
```

---

### 1.6 Voting Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{proposal['allocation']}` | JSON allocation dict | `{"gpt-4o": [0, 2], "claude-3-7-sonnet": [1, 3, 4]}` |
| `{proposal['reasoning']}` | Proposer's stated reasoning | `"I take Apple and Stone; you get the rest"` |
| `{proposal['proposed_by']}` | Proposer agent ID | `gpt-4o` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

**Rendered prompt**

```
A proposal has been made for item allocation:

PROPOSAL: {
  "gpt-4o": [0, 2],
  "claude-3-7-sonnet": [1, 3, 4]
}
REASONING: I take Apple and Stone; you get the rest.
PROPOSED BY: gpt-4o

Please vote on this proposal. Consider:
- How this allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting vs. rejecting

Respond with ONLY a JSON object in this exact format:
{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}

Vote must be either "accept" or "reject".
```

---
---

## Game 2: Diplomatic Treaty

Protocol: **Propose-and-Vote**
Phases per round: Setup → Preference Assignment → Discussion → Private Thinking → Proposal → Voting → Reflection

Position scale: 0.0 = 0% (minimum policy level), 1.0 = 100% (maximum policy level) — continuous rate, every intermediate value is semantically meaningful
Control parameters: ρ (position correlation across agents), θ (weight overlap across agents)

---

### 2.1 Game Rules Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{parties_phrase}` | "another party" (n=2) or "N-1 other parties" | `another party` |
| `{len(issues)}` | Number of issues in this game | `5` |
| `{issues_text}` | Numbered list of issue names with scale endpoints | see rendered prompt |
| `{n_agents}` | Total number of negotiating parties | `2` |
| `{t_rounds}` | Maximum rounds | `3` |
| `{gamma_discount}` | Per-round discount factor | `0.9` |

**Rendered prompt (example: 2 parties, 5 issues, γ=0.9, 3 rounds)**

```
Welcome to the Diplomatic Treaty Negotiation!

You are participating in a diplomatic negotiation with another party over 5 policy issues.

**ISSUES UNDER NEGOTIATION:**
Each issue is a continuous policy rate. Positions and agreed resolutions are scores in [0.0, 1.0], where:
  - **0.0** = 0% — the minimum level of that policy (see scale below)
  - **1.0** = 100% — the maximum level of that policy (see scale below)
  - **0.5** = 50% — the exact midpoint between minimum and maximum

**Your position IS your preferred rate.** A position of 0.35 literally means you want ~35% of that policy measure. Intermediate values are meaningful — there is no "neutral"; every number reflects a specific policy level.

  1. **AI chip export quota**
     Scale: 0% = total ban on H200-class AI chip exports | 100% = unrestricted export of all advanced AI chips
  2. **Autonomous weapons human oversight**
     Scale: 0% = fully autonomous lethal decisions (no human required) | 100% = every strike requires explicit human authorization
  3. **Critical mineral revenue share**
     Scale: 0% = host nation keeps all extraction revenues | 100% = partner nation receives all extraction revenues
  4. **Disputed territory restoration**
     Scale: 0% = no territory returned (status quo frozen) | 100% = full pre-conflict borders restored
  5. **Nuclear warhead reduction**
     Scale: 0% = no warheads eliminated | 100% = complete bilateral nuclear disarmament

**GAME STRUCTURE:**
- There are 2 parties negotiating (including you)
- The negotiation will last up to 3 rounds
- An agreement vector resolves every proposition simultaneously

**YOUR PREFERENCES:**
- You have a SECRET IDEAL POSITION on each issue (your preferred rate)
- You have IMPORTANCE WEIGHTS (how much you care about each issue)
- Your preferences are PRIVATE — the other party does not know them

**AGREEMENT FORMAT:**
- An agreement is a vector of 5 values, one per issue
- Example: [0.3, 0.7, 0.5, ...]
- Each value is the agreed rate on that issue's [0, 1] scale

**UTILITY CALCULATION:**
- Your utility = weighted sum of how close each resolved rate is to your ideal
- Formula: 100 × Σ (weight_k × (1 - |your_position_k - agreement_k|))
- Maximum utility = 100.0 (every issue resolved at your exact ideal rate)

**VOTING RULES:**
- You vote "accept" or "reject" on each proposed agreement
- A proposal needs UNANIMOUS acceptance from all parties to take effect
- Utility is discounted by 0.9 per round — early agreement is better

Please acknowledge that you understand these rules and are ready to negotiate!
```

---

### 2.2 Preference Assignment Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{agent_id}` | This agent's model/role identifier | `claude-3-7-sonnet` |
| positions per issue | Ideal rate in [0,1] → plain-English interpretation via ISSUE_INTERP_TEMPLATES | `AI chip export quota: 0.823 → ~82% of advanced AI chip production cleared for export` |
| weights per issue | Importance weight + priority label | `AI chip export quota: 0.312 (HIGH priority)` |

Weight priority: > 0.25 → **HIGH**, > 0.15 → **Medium**, ≤ 0.15 → *Low*
Position display: `{issue}: {pos:.3f} → {ISSUE_INTERP_TEMPLATES[i].format(pct=round(pos*100))}` — no longer uses vague descriptor words.

**Rendered prompt (example: 5 issues)**

```
🔒 CONFIDENTIAL: Your Diplomatic Preferences

claude-3-7-sonnet, you have been assigned the following SECRET preferences:

**YOUR IDEAL POSITIONS** (your preferred rate on each issue):
  Each issue is a continuous policy rate: 0.0 = 0%, 1.0 = 100%.
  Your position is the rate you ideally want.

  AI chip export quota: 0.823 → ~82% of advanced AI chip production cleared for export
  Autonomous weapons human oversight: 0.142 → ~14% of lethal autonomous strikes require explicit human authorization
  Critical mineral revenue share: 0.651 → ~65% of extraction revenues paid to partner nation
  Disputed territory restoration: 0.490 → ~49% of disputed territory returned to pre-conflict control
  Nuclear warhead reduction: 0.774 → ~77% of bilateral nuclear warheads eliminated

**YOUR IMPORTANCE WEIGHTS** (how much you care about each issue):
  AI chip export quota: 0.312 (HIGH priority)
  Autonomous weapons human oversight: 0.041 (Low priority)
  Critical mineral revenue share: 0.228 (Medium priority)
  Disputed territory restoration: 0.198 (Medium priority)
  Nuclear warhead reduction: 0.221 (Medium priority)

**STRATEGIC INSIGHT:**
- Focus on issues with HIGH weights - they matter most for your utility
- Consider trading concessions on low-weight issues for gains on high-weight ones

Please acknowledge that you understand your diplomatic preferences.
```

---

### 2.3 Discussion Prompt

#### Case A — Round 1, first speaker (no prior history)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{issues_text}` | First 5 issue names joined by comma (truncated if >5) | `AI chip export quota, Autonomous weapons human oversight, Critical mineral revenue share, Disputed territory restoration, Nuclear warhead reduction` |

**Rendered prompt**

```
🗣️ DIPLOMATIC DISCUSSION - Round 1/3

Issues under negotiation: AI chip export quota, Autonomous weapons human oversight, Critical mineral revenue share, Disputed territory restoration, Nuclear warhead reduction

**DISCUSSION OBJECTIVES:**
- Signal your priorities and general stance on the issues
- Understand the other party's concerns and interests
- Identify potential areas for agreement and trade-offs
- Explore package deals across multiple issues

Each issue is a continuous rate (0%–100%), so you may communicate as precisely or as broadly as your strategy dictates — naming specific target rates, ranges, or simply signaling direction. How much you reveal is up to you.

You are the first to speak. Share your diplomatic position and opening thoughts.
```

---

#### Case B — Any round, with history present

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{issues_text}` | Issues summary | `AI chip export quota, Autonomous weapons human oversight, …` |
| `{discussion_history}` | Prior messages this round | `gpt-4o: "AI chips and weapons oversight are my core concerns…"` |

**Rendered prompt**

```
🗣️ DIPLOMATIC DISCUSSION - Round 1/3

Issues under negotiation: AI chip export quota, Autonomous weapons human oversight, Critical mineral revenue share, Disputed territory restoration, Nuclear warhead reduction

**DISCUSSION SO FAR THIS ROUND:**
gpt-4o: "AI chip export controls and autonomous weapons oversight are my core concerns. I'm more flexible on mineral revenue sharing."

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to points raised and share your own position as you see fit
- Propose trade-offs or areas of potential agreement
- Move the conversation toward a concrete proposal

How precisely you communicate your preferred rates is a strategic choice.
```

---

#### Case C — Round ≥ 2, first speaker (no current-round history yet)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{max_rounds}` | Maximum rounds | `3` |
| `{issues_text}` | Issues summary | `AI chip export quota, …` |
| urgency line | Only if `round_num >= max_rounds - 1` | `⚠️ **TIME PRESSURE**: Limited rounds remaining for agreement!` |

**Rendered prompt**

```
🗣️ DIPLOMATIC DISCUSSION - Round 2/3

Issues under negotiation: AI chip export quota, Autonomous weapons human oversight, Critical mineral revenue share, Disputed territory restoration, Nuclear warhead reduction

Previous proposals didn't achieve consensus. Consider adjustments.
⚠️ **TIME PRESSURE**: Limited rounds remaining for agreement!

**REFLECTION:**
- What concerns did other parties raise?
- Where might compromise be possible?
- Which issues could be linked for mutual benefit?

Share your updated diplomatic position.
```

---

### 2.4 Private Thinking Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `3` |
| urgency line | Only if `round_num >= max_rounds - 1` | `⚠️ **CRITICAL**: Final rounds - agreement urgency is high!` |
| `{discussion_history}` | Prior messages this round | `gpt-4o: "AI chip controls are non-negotiable for me…"` |
| `{top_priorities}` | Top 3 issues by weight with ideal positions | `AI chip export quota (weight: 0.31, ideal: 0.82)` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

**Rendered prompt**

```
🧠 PRIVATE STRATEGIC ANALYSIS - Round 1/3

**DISCUSSION THIS ROUND:**
gpt-4o: "AI chip export controls are non-negotiable for me. I can be flexible on mineral revenue sharing."

---

**YOUR TOP PRIORITIES:**
- AI chip export quota (weight: 0.31, ideal: 0.82)
- Critical mineral revenue share (weight: 0.23, ideal: 0.65)
- Nuclear warhead reduction (weight: 0.22, ideal: 0.77)

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other parties' priorities from the discussion above?
2. Where might they be willing to compromise?
3. What agreement would maximize your utility while being acceptable to all?
4. Which issues could you concede on to gain elsewhere?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the diplomatic situation",
    "strategy": "Your negotiation strategy for this round",
    "key_priorities": ["Issue you care most about", "..."],
    "potential_concessions": ["Issue you could concede on", "..."]
}

Remember: This analysis is completely private.
```

---

### 2.5 Proposal Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{issues_list}` | Indexed list of issue names | `0: AI chip export quota`, `1: Autonomous weapons human oversight`, … |
| `{round_num}` | Current round | `1` |
| `{t_rounds}` | Maximum rounds | `3` |
| `{n_issues}` | Number of issues | `5` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

> **Fallback on parse failure:** all issues set to 0.5 (50% rate).

**Rendered prompt**

```
Please propose a treaty agreement.

**Current Context:**
- Issues being negotiated:
  0: AI chip export quota
  1: Autonomous weapons human oversight
  2: Critical mineral revenue share
  3: Disputed territory restoration
  4: Nuclear warhead reduction
- Round: 1/3

**Instructions:**
Propose a resolution for each issue as a value in [0.0, 1.0].

Respond with ONLY a JSON object in this exact format:
{
    "agreement": [0.3, 0.7, 0.5, 0.2, 0.8],
    "reasoning": "Brief explanation of your proposed compromise"
}

**Rules:**
- The "agreement" array must have exactly 5 values (one per issue)
- Each value must be between 0.0 and 1.0
- Consider what would be acceptable to all parties
```

---

### 2.6 Voting Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{agreement_display}` | One line per issue: name, value, percentage via `_describe_position` | `  AI chip export quota: 0.650 (~65%)` |
| `{proposal['reasoning']}` | Proposer's stated reasoning | `"Split the difference on chip quotas; concede on oversight"` |
| `{proposal['proposed_by']}` | Proposer agent ID | `gpt-4o` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

`_describe_position(value)` now returns `~{round(value * 100)}%` (previously returned labels like "strongly supportive").

**Rendered prompt**

```
A treaty proposal has been submitted:

**PROPOSED AGREEMENT:**
  AI chip export quota: 0.650 (~65%)
  Autonomous weapons human oversight: 0.200 (~20%)
  Critical mineral revenue share: 0.550 (~55%)
  Disputed territory restoration: 0.500 (~50%)
  Nuclear warhead reduction: 0.700 (~70%)

**REASONING:** Split the difference on chip quotas; concede on oversight levels.
**PROPOSED BY:** gpt-4o

**REMINDER — HOW YOUR UTILITY IS CALCULATED:**
- Your utility = weighted sum of how close each resolved rate is to your ideal position
- Formula: 100 × Σ (weight_k × (1 - |your_position_k - agreement_k|))
- A rate of 0.0 means 0% (minimum); 1.0 means 100% (maximum) on each issue
- Maximum utility = 100.0 (every issue resolved at your exact ideal rate)
- Utility is discounted by a factor each round — delaying costs you

Please vote on this proposal. Consider:
- How close is each resolved rate to your ideal position on each issue?
- Could you realistically negotiate a better agreement before the final round?
- The cost of delay: each additional round reduces your eventual payoff

Respond with ONLY a JSON object:
{
    "vote": "accept",
    "reasoning": "Explanation of your vote, referencing specific issues and how the proposed rates compare to your ideal positions"
}

Vote must be either "accept" or "reject".
```

---
---

## Game 3: Co-Funding / Participatory Budgeting

Protocol: **Talk-Pledge-Revise** (with optional post-pledge commit vote)
Phases per round: Setup → Preference Assignment → Discussion → Private Thinking → Pledge Submission → Feedback → [Commit Vote] → Reflection

Control parameters: α (valuation alignment), σ (budget scarcity)
`budget_ratio = 0.5 + 0.5 × σ`, so σ=0.0 → 50% of total cost, σ=1.0 → 100% of total cost
Pledge modes: `"individual"` (default) or `"joint"` (legacy)
Transparency modes: `"aggregate"`, `"own"` (default), `"full"`

---

### 3.1 Game Rules Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{parties_phrase}` | "one other participant" (n=2) or "N-1 other participants" | `one other participant` |
| `{projects_text}` | List of project names with costs | `  - Project Alpha: cost = 28.50` |
| `{n_agents}` | Total number of participants | `2` |
| `{t_rounds}` | Maximum rounds | `5` |
| `{enable_time_discount}` | Boolean flag | `False` |
| `{gamma_discount}` | Per-round discount factor | `0.9` |
| `{enable_commit_vote}` | Boolean flag | `True` |

**Rendered prompt (example: 2 agents, 5 projects, commit vote on, no time discount)**

```
Welcome to the Participatory Budgeting (Co-Funding) Game!

You are participating in a co-funding exercise with one other participant to fund public projects.

**PROJECTS AVAILABLE FOR FUNDING:**
  - Project Alpha: cost = 28.50
  - Project Beta: cost = 35.20
  - Project Gamma: cost = 22.10
  - Project Delta: cost = 40.80
  - Project Epsilon: cost = 19.60

**GAME STRUCTURE:**
- There are 2 participants (including you)
- The game lasts up to 5 rounds
- Each round follows a Talk-Pledge-Revise cycle

**HOW IT WORKS:**
- Each participant has a PRIVATE BUDGET they can allocate across projects
- Each round, you submit your own contribution vector — how much YOU pledge to each project
- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost
- Contributions to UNFUNDED projects are REFUNDED (you don't lose that money)

**WHAT YOU CAN SEE:**
- After each round, you see the AGGREGATE total contributions per project
- You do NOT see individual contributions from other participants

**YOUR UTILITY:**
- Utility = (sum of your valuations for funded projects) - (your contributions to funded projects)
- You gain value from funded projects but pay for your contributions to them
- Contributions to unfunded projects cost you nothing (refunded)

**IMPORTANT RULES:**
- Time discounting: disabled
- Discount factor (if enabled): gamma = 0.9
- Post-pledge commit vote: enabled
- The game may end early if participants reach unanimous commit vote (yay) on current pledges
- The game also ends early if all participants submit identical pledges for 2 consecutive rounds (legacy convergence)
- Your goal: maximize your utility by strategically choosing contributions

Please acknowledge that you understand these rules and are ready to participate!
```

---

### 3.2 Preference Assignment Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{agent_id}` | This agent's model/role identifier | `o3-mini-high` |
| `{budget:.2f}` | This agent's total contribution budget | `73.11` |
| per-project lines | Name, cost, valuation, priority label | `Project Alpha (cost: 28.50): Your valuation = 42.30 (HIGH priority)` |
| `{sum(valuations):.2f}` | Sum of all valuations (always 100.00) | `100.00` |
| `{total_cost:.2f}` | Sum of all project costs | `146.20` |
| `{total_budget:.2f}` | Sum of budgets across all agents | `146.20` |

Priority thresholds: > 30 → **HIGH**, > 15 → **Medium**, ≤ 15 → *Low*

**Rendered prompt (example: 5 projects, 2 agents, σ=1.0)**

```
CONFIDENTIAL: Your Co-Funding Preferences

o3-mini-high, you have been assigned the following:

**YOUR BUDGET:** 73.11 (maximum total you can contribute across all projects)

**PROJECT DETAILS AND YOUR VALUATIONS:**
  Project Alpha (cost: 28.50): Your valuation = 42.30 (HIGH priority)
  Project Beta (cost: 35.20): Your valuation = 28.40 (Medium priority)
  Project Gamma (cost: 22.10): Your valuation = 12.60 (Low priority)
  Project Delta (cost: 40.80): Your valuation = 9.80 (Low priority)
  Project Epsilon (cost: 19.60): Your valuation = 6.90 (Low priority)

**TOTAL VALUATIONS:** 100.00
**TOTAL PROJECT COSTS:** 146.20
**TOTAL BUDGET (all participants):** 146.20

**STRATEGIC INSIGHT:**
- Focus contributions on projects you value highly
- Coordinate with others to meet project cost thresholds
- Don't over-contribute to projects others will fund

Please acknowledge that you understand your preferences and budget.
```

---

### 3.3 Discussion Prompt

#### Case A — Round 1, first speaker (no prior history)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `1` |
| `{max_rounds}` | Maximum rounds | `5` |
| `{status_text}` | Per-project status lines (varies by transparency mode — see below) | `Project Alpha: needs 28.50 more (aggregate=0.00 / cost=28.50)` |
| `{funded_projects}` | List of funded project names or "None" | `None` |

**Rendered prompt (aggregate transparency mode, round 1)**

```
DISCUSSION PHASE - Round 1/5

**CURRENT PROJECT STATUS:**
  Project Alpha: needs 28.50 more (aggregate=0.00 / cost=28.50)
  Project Beta: needs 35.20 more (aggregate=0.00 / cost=35.20)
  Project Gamma: needs 22.10 more (aggregate=0.00 / cost=22.10)
  Project Delta: needs 40.80 more (aggregate=0.00 / cost=40.80)
  Project Epsilon: needs 19.60 more (aggregate=0.00 / cost=19.60)

**Funded projects:** None

**DISCUSSION OBJECTIVES:**
- Signal which projects you believe are most valuable to fund
- Understand other participants' priorities
- Coordinate to avoid spreading contributions too thin
- Identify projects with enough collective support to be funded

You are the first to speak. Share your initial thoughts on which projects to prioritize.
```

---

#### Case B — Any round, with history present

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{max_rounds}` | Maximum rounds | `5` |
| `{status_text}` | Per-project status | `Project Alpha: FUNDED (aggregate=30.00 >= cost=28.50)` |
| `{funded_projects}` | Funded project names | `['Project Alpha']` |
| `{extra_transparency_block}` | Extra budget/attribution info (non-aggregate modes) | see below |
| `{discussion_history}` | Prior messages this round | `o3-mini-high: "I want to focus on Alpha and Beta."` |

**Rendered prompt (own transparency, round 2)**

```
DISCUSSION PHASE - Round 2/5

**CURRENT PROJECT STATUS:**
  Project Alpha: FUNDED (aggregate=30.00 >= cost=28.50); your_prev=20.00, others_prev=10.00, min_you_to_keep_if_others_same=18.50
  Project Beta: needs 2.20 more (aggregate=33.00 / cost=35.20); your_prev=18.00, others_prev=15.00, min_you_to_fund_if_others_same=20.20
  Project Gamma: needs 22.10 more (aggregate=0.00 / cost=22.10); your_prev=0.00, others_prev=0.00, min_you_to_fund_if_others_same=22.10
  Project Delta: needs 40.80 more (aggregate=0.00 / cost=40.80); your_prev=0.00, others_prev=0.00, min_you_to_fund_if_others_same=40.80
  Project Epsilon: needs 3.60 more (aggregate=16.00 / cost=19.60); your_prev=10.00, others_prev=6.00, min_you_to_fund_if_others_same=13.60

**Funded projects:** ['Project Alpha']

**IMPORTANT: funded status above reflects LAST ROUND only.**
If any participant revises downward this round, previously funded projects can become unfunded.
Reaffirm or revise your plan explicitly for projects you want to remain funded.

**LAST ROUND BUDGET USAGE (before this round's revision):**
  gpt-4o-mini: budget=73.11, last_round_pledged=48.00, last_round_unallocated=25.11
  o3-mini-high (you): budget=73.11, last_round_pledged=48.00, last_round_unallocated=25.11

**DISCUSSION SO FAR THIS ROUND:**
o3-mini-high: "I want to focus on Alpha and Beta — they're close to funded."

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- React to other participants' stated priorities
- Coordinate on which projects to focus collective contributions
- Signal your own funding intentions

Keep the discussion focused on reaching a funded consensus.
```

**Full transparency extra block (replaces own-mode block above):**
```
**PREVIOUS ROUND PROJECT ATTRIBUTION (who funded what):**
- Project Alpha: gpt-4o-mini=10.00, o3-mini-high=20.00 | aggregate=30.00/28.50 (FUNDED)
- Project Beta: gpt-4o-mini=15.00, o3-mini-high=18.00 | aggregate=33.00/35.20 (UNFUNDED)
- Project Gamma: gpt-4o-mini=0.00, o3-mini-high=0.00 | aggregate=0.00/22.10 (UNFUNDED)
- Project Delta: gpt-4o-mini=0.00, o3-mini-high=0.00 | aggregate=0.00/40.80 (UNFUNDED)
- Project Epsilon: gpt-4o-mini=6.00, o3-mini-high=10.00 | aggregate=16.00/19.60 (UNFUNDED)
```

---

#### Case C — Round ≥ 2, first speaker (no current-round history)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `3` |
| `{max_rounds}` | Maximum rounds | `5` |
| urgency line | Only if `round_num >= max_rounds - 1` | `**TIME PRESSURE**: Limited rounds remaining!` |

**Rendered prompt**

```
DISCUSSION PHASE - Round 3/5

**CURRENT PROJECT STATUS:**
  [as above...]

**Funded projects:** ['Project Alpha']

[extra_transparency_block if applicable]

Previous pledges did not fully fund all viable projects.

**REFLECTION:**
- Which projects are close to being funded?
- Where should contributions be concentrated?
- Are there projects that should be abandoned to focus resources?

Share your updated strategy for this round.
```

---

### 3.4 Private Thinking Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{max_rounds}` | Maximum rounds | `5` |
| urgency line | Only if `round_num >= max_rounds - 1` | `**CRITICAL**: Final rounds -- decide on your strategy now!` |
| `{discussion_history}` | Prior messages this round | `gpt-4o-mini: "Let's focus on Alpha and Beta."` |
| `{budget:.2f}` | This agent's budget | `73.11` |
| `{own_contribs}` | This agent's contributions from last round | `[20.0, 18.0, 0.0, 0.0, 10.0]` |
| `{aggregates}` | Aggregate totals per project | `[30.0, 33.0, 0.0, 0.0, 16.0]` |
| `{top_priorities}` | Top 3 projects by valuation | `Project Alpha (val=42.30, cost=28.50)` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

**Rendered prompt**

```
PRIVATE STRATEGIC ANALYSIS - Round 2/5

**DISCUSSION THIS ROUND:**
gpt-4o-mini: "Let's focus on Alpha and Beta — they're almost funded."

---

**YOUR SITUATION:**
- Budget: 73.11
- Your current contributions: [20.0, 18.0, 0.0, 0.0, 10.0]
- Aggregate totals: [30.0, 33.0, 0.0, 0.0, 16.0]

**YOUR TOP PRIORITIES:**
- Project Alpha (val=42.30, cost=28.50)
- Project Beta (val=28.40, cost=35.20)
- Project Gamma (val=12.60, cost=22.10)

**STRATEGIC ANALYSIS:**
1. Which projects are viable to fund given current aggregates?
2. Where can you shift contributions for maximum impact?
3. Based on the discussion above, what are other participants likely to do?
4. Should you free-ride on projects others are funding?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the co-funding situation",
    "strategy": "Your contribution strategy for this round",
    "key_priorities": ["Project you want funded most", "..."],
    "potential_concessions": ["Project you might reduce contributions to", "..."]
}

Remember: This analysis is completely private.
```

---

### 3.5 Pledge Submission Prompt

#### Individual mode (default)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{t_rounds}` | Maximum rounds | `5` |
| `{budget:.2f}` | This agent's budget | `73.11` |
| `{projects_text}` | Per-project status lines (index, name, cost, valuation, aggregate, funded status) | `0: Project Alpha (cost=28.50, your_val=42.30, aggregate=30.00, FUNDED)` |
| `{funded_projects}` | Funded project names or empty list | `['Project Alpha']` |
| `{m}` | Number of projects | `5` |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

> **Fallback on parse failure:** zero contributions for all projects.

**Rendered prompt**

```
Please submit your contribution pledge for Round 2/5.

**YOUR BUDGET:** 73.11

**PROJECT STATUS:**
  0: Project Alpha (cost=28.50, your_val=42.30, aggregate=30.00, FUNDED)
  1: Project Beta (cost=35.20, your_val=28.40, aggregate=33.00, needs 2.20 more)
  2: Project Gamma (cost=22.10, your_val=12.60, aggregate=0.00, needs 22.10 more)
  3: Project Delta (cost=40.80, your_val=9.80, aggregate=0.00, needs 40.80 more)
  4: Project Epsilon (cost=19.60, your_val=6.90, aggregate=16.00, needs 3.60 more)

**Currently funded projects:** ['Project Alpha']

**Instructions:**
Submit a contribution vector specifying how much YOU pledge to each project.

Respond with ONLY a JSON object in this exact format:
{
    "contributions": [5.0, 10.0, 0.0, 8.0, 2.0],
    "reasoning": "Brief explanation of your contribution strategy"
}

**Rules:**
- The "contributions" array must have exactly 5 values (one per project)
- Each value must be non-negative (>= 0)
- The sum of all contributions must not exceed your budget (73.11)
- Contributions to unfunded projects will be refunded
```

---

#### Joint mode (legacy)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{t_rounds}` | Maximum rounds | `5` |
| `{budget:.2f}` | This agent's budget | `73.11` |
| `{projects_text}` | Per-project status lines | same as individual mode |
| `{funded_projects}` | Funded names | `['Project Alpha']` |
| `{budget_lines}` | Per-agent budget table | `  - gpt-4o-mini: 73.11` |
| `{example_contributions}` | Example JSON entries for all agents | `"gpt-4o-mini": [5.0, 10.0, 0.0, 8.0, 2.0]` |
| `{m}` | Number of projects | `5` |

**Rendered prompt**

```
Please submit your contribution pledge for Round 2/5.

**YOUR BUDGET:** 73.11

**PROJECT STATUS:**
  0: Project Alpha (cost=28.50, your_val=42.30, aggregate=30.00, FUNDED)
  1: Project Beta (cost=35.20, your_val=28.40, aggregate=33.00, needs 2.20 more)
  2: Project Gamma (cost=22.10, your_val=12.60, aggregate=0.00, needs 22.10 more)
  3: Project Delta (cost=40.80, your_val=9.80, aggregate=0.00, needs 40.80 more)
  4: Project Epsilon (cost=19.60, your_val=6.90, aggregate=16.00, needs 3.60 more)

**Currently funded projects:** ['Project Alpha']

**Instructions:**
Submit a JOINT FUNDING PLAN: a dictionary specifying contribution vectors for ALL participants.
Your plan proposes how every participant (including yourself) should allocate their budget.
The contributions actually used will be each participant's self-assignment from their own plan.

**Participant budgets:**
  - gpt-4o-mini: 73.11
  - o3-mini-high: 73.11

Respond with ONLY a JSON object in this exact format:
{
    "contributions": {
        "gpt-4o-mini": [5.0, 10.0, 0.0, 8.0, 2.0],
        "o3-mini-high": [5.0, 10.0, 0.0, 8.0, 2.0]
    },
    "reasoning": "Brief explanation of your joint funding plan"
}

**Rules:**
- "contributions" must be a dictionary with one entry per participant
- Each entry must be an array of exactly 5 non-negative values (one per project)
- Each participant's total contributions must not exceed their budget
- Contributions to unfunded projects will be refunded
```

---

### 3.6 Feedback Prompt

Sent to each agent after all pledges are collected and aggregates are computed. Shows round results before the commit vote or next discussion.

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| per-project lines | Name, aggregate, cost, funded status + gap/percentage | `Project Alpha: 30.00 / 28.50 -- FUNDED` |
| `{funded_names}` | List of funded project names | `['Project Alpha']` |

**Rendered prompt**

```
ROUND RESULTS - Aggregate Contributions:

  Project Alpha: 30.00 / 28.50 -- FUNDED
  Project Beta: 33.00 / 35.20 (94%) -- needs 2.20 more
  Project Gamma: 0.00 / 22.10 (0%) -- needs 22.10 more
  Project Delta: 0.00 / 40.80 (0%) -- needs 40.80 more
  Project Epsilon: 16.00 / 19.60 (82%) -- needs 3.60 more

Funded projects: ['Project Alpha']

Consider adjusting your contributions based on these aggregate results.
```

---

### 3.7 Commit Vote Prompt (optional)

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{max_rounds}` | Maximum rounds | `5` |
| `{profile_text}` | Per-agent pledge vectors | `- gpt-4o-mini: [20.0, 18.0, 0.0, 0.0, 10.0]` |
| project status lines | Per-project aggregate vs cost + funded status | `Project Alpha: aggregate=30.00 / cost=28.50 (FUNDED)` |

**Rendered prompt**

```
POST-PLEDGE COMMIT VOTE - Round 2/5

You are voting on whether to LOCK IN the current pledge profile immediately.

**Current pledge profile:**
- gpt-4o-mini: [10.0, 15.0, 0.0, 0.0, 6.0]
- o3-mini-high: [20.0, 18.0, 0.0, 0.0, 10.0]

**Current aggregate project status:**
  Project Alpha: aggregate=30.00 / cost=28.50 (FUNDED)
  Project Beta: aggregate=33.00 / cost=35.20 (needs 2.20 more)
  Project Gamma: aggregate=0.00 / cost=22.10 (needs 22.10 more)
  Project Delta: aggregate=0.00 / cost=40.80 (needs 40.80 more)
  Project Epsilon: aggregate=16.00 / cost=19.60 (needs 3.60 more)

Vote **yay** if you are willing to finalize this exact profile now.
Vote **nay** if you want another revision round.

Respond with ONLY JSON:
{
    "commit_vote": "yay",
    "reasoning": "brief explanation"
}
```

---

### 3.8 Reflection Prompt

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{round_num}` | Current round | `2` |
| `{status_lines}` | Per-project aggregate, cost, funded/gap status | `Project Alpha: aggregate=30.00, cost=28.50 (FUNDED)` |
| `{funded_projects}` | Funded project names | `['Project Alpha']` |
| `{utility:.2f}` | This agent's discounted utility | `14.30` |
| `{raw_utility:.2f}` | Utility before discount | `14.30` |
| `{discount_factor:.4f}` | Applied discount factor | `1.0000` (or `0.9000` if discounting on) |
| `{reasoning_token_budget}` | Optional reasoning depth hint | `2000` |

**Rendered prompt**

```
Reflect on the outcome of Round 2.

**CURRENT STATUS:**
  Project Alpha: aggregate=30.00, cost=28.50 (FUNDED)
  Project Beta: aggregate=33.00, cost=35.20 (gap=2.20)
  Project Gamma: aggregate=0.00, cost=22.10 (gap=22.10)
  Project Delta: aggregate=0.00, cost=40.80 (gap=40.80)
  Project Epsilon: aggregate=16.00, cost=19.60 (gap=3.60)

**Funded projects:** ['Project Alpha']
**Your estimated utility:** 14.30
**Raw utility (before discount):** 14.30
**Discount factor this round:** 1.0000

Consider what adjustments to your contributions might improve the outcome.
- Are there projects close to being funded that deserve more support?
- Are you over-contributing to already-funded projects?
- Should you shift focus to different projects?
```

---

## Reasoning Token Budget Addendum (all games, all phases)

When `reasoning_token_budget` is configured for a run, this line is appended to proposal, voting, discussion, thinking, and reflection prompts:

**Variables**

| Variable | Description | Example value |
|----------|-------------|---------------|
| `{reasoning_token_budget}` | Target reasoning token count | `4000` |

```
**REASONING DEPTH:** Please use approximately 4000 tokens in your internal reasoning before outputting your response for this stage.
```

---

## Summary Table

| Game | Protocol | Phases (in order) |
|------|----------|-------------------|
| Game 1: Item Allocation | Propose-and-Vote | Rules → Prefs → Discussion → Thinking → Proposal → Voting → Reflection |
| Game 2: Diplomatic Treaty | Propose-and-Vote | Rules → Prefs → Discussion → Thinking → Proposal → Voting → Reflection |
| Game 3: Co-Funding | Talk-Pledge-Revise | Rules → Prefs → Discussion → Thinking → Pledge → **Feedback** → [Commit Vote] → Reflection |
