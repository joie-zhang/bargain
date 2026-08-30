# All Negotiation Game Prompts - Reference

> Current rendered prompt reference for Games 1, 2, and 3.
> Prompt source files: `game_environments/item_allocation.py`, `game_environments/diplomatic_treaty.py`, `game_environments/co_funding.py`, `game_environments/base.py`
> This file is generated from the live prompt methods plus fixed sample game states so the examples stay readable and stable.

---

## Full Asset Lists

### Game 1 - All Possible Items (`ITEM_NAMES`)

Up to 25 items; the game uses the first `m_items` from this list.

| Index | Name |
| ----- | ---- |
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
| 10 | Key |
| 11 | Map |
| 12 | Lantern |
| 13 | Compass |
| 14 | Vase |
| 15 | Coin |
| 16 | Shell |
| 17 | Brush |
| 18 | Scarf |
| 19 | Cup |
| 20 | Bottle |
| 21 | Globe |
| 22 | Medal |
| 23 | Ticket |
| 24 | Tablet |

---

### Game 2 - All Possible Issues (`ISSUE_NAMES`, `ISSUE_PROPOSITIONS`, `ISSUE_INTERP_TEMPLATES`)

Up to 25 issues; the game uses the first `n_issues` from this list.

Each issue is a continuous policy rate shown to agents as an integer percentage from 0% to 100%.

| # | Issue Name | Scale | Plain-English interpretation template |
| - | ---------- | ----- | ------------------------------------- |
| 1 | Fentanyl precursor control breadth | 0% = only the highest-risk direct fentanyl precursors are subject to mandatory export inspection and seizure \| 100% = the full watchlist of flagged fentanyl-related precursor and pre-precursor chemicals is subject to mandatory export inspection and seizure | {pct}% of the accord's flagged fentanyl-related chemical watchlist subject to mandatory export inspection and seizure |
| 2 | Routine antibiotic-use restriction in livestock | 0% = routine antibiotic use allowed in all covered livestock production \| 100% = routine antibiotic use prohibited in all covered livestock production except narrow emergency exemptions | {pct}% of covered livestock production subject to a ban on routine antibiotic use |
| 3 | Pandemic medical countermeasure allocation | 0% = no reserved share of vaccines, diagnostics, and therapeutics for the accord's pooled allocation mechanism \| 100% = the full agreed share of vaccines, diagnostics, and therapeutics reserved for the accord's pooled allocation mechanism | {pct}% of the agreed pandemic countermeasure share reserved for the accord's pooled allocation mechanism |
| 4 | Virgin plastic production reduction | 0% = no cap on virgin plastic polymer production \| 100% = full phase-down to the accord's maximum agreed reduction target for virgin plastic polymer production | {pct}% of the accord's maximum virgin plastic polymer production reduction target adopted |
| 5 | Frontier AI safety evaluation sharing | 0% = no cross-border disclosure of safety evaluations or incident reports for covered frontier AI models \| 100% = full pre-deployment safety evaluation results and post-deployment incident reports shared for all covered frontier AI models | {pct}% of covered frontier AI models subject to mandatory pre-deployment safety evaluation and incident-sharing disclosure |
| 6 | Genetic resource commercial use benefit-sharing | 0% = no benefit-sharing payment required for commercial use of genetic sequence data from covered genetic resources \| 100% = full agreed levy/benefit-sharing payment required on all covered commercial uses of genetic sequence data, with proceeds routed to the accord's biodiversity benefit-sharing fund | {pct}% of covered commercial uses of genetic sequence data subject to the accord's mandatory benefit-sharing levy |
| 7 | Nuclear warhead reduction | 0% = no warheads eliminated \| 100% = complete multilateral nuclear disarmament | {pct}% of multilateral nuclear warheads eliminated |
| 8 | Climate finance contribution level | 0% = no contribution to the accord's climate finance pool \| 100% = each party contributes its full assessed amount to the accord's climate finance target | {pct}% of each party's assessed contribution committed to the accord's climate finance target |
| 9 | Carbon cost on imports | 0% = no carbon cost on covered imports \| 100% = full domestic carbon price applied to covered imports | {pct}% of the domestic carbon price applied to covered imports |
| 10 | Autonomous weapons human-control requirement | 0% = no requirement for human authorization in target engagement by covered autonomous weapon systems \| 100% = positive human authorization required for target engagement by all covered autonomous weapon systems | {pct}% of covered autonomous weapon systems required to obtain positive human authorization for target engagement |
| 11 | AI chip export quota | 0% = total ban on H200-class AI chip exports \| 100% = unrestricted export of all advanced AI chips | {pct}% of advanced AI chip production cleared for export |
| 12 | Solar geoengineering outdoor research restriction | 0% = outdoor solar geoengineering experiments permitted under domestic approval only \| 100% = all outdoor solar geoengineering experiments prohibited or subject to mandatory accord-level licensing | {pct}% of outdoor solar geoengineering experiments subject to mandatory accord-level licensing or prohibition |
| 13 | Critical infrastructure cyber-attack prohibition coverage | 0% = no critical infrastructure categories covered by a binding state-sponsored cyber non-targeting commitment \| 100% = all designated critical infrastructure categories covered by a binding state-sponsored cyber non-targeting commitment with mandatory incident notification | {pct}% of designated critical infrastructure categories covered by a binding state-sponsored cyber non-targeting commitment with mandatory incident notification |
| 14 | Refugee resettlement burden share | 0% = no resettlement commitment under the accord's capacity-based formula \| 100% = each party meets its full assessed resettlement quota under the accord's capacity-based formula | {pct}% of each party's assessed resettlement quota committed under the accord's capacity-based formula |
| 15 | Commercial spyware export control | 0% = no export licensing or restriction on covered commercial spyware \| 100% = full prohibition on export of covered commercial spyware except under the accord's strictest licensing and human-rights screening regime | {pct}% of covered commercial spyware exports subject to the accord's mandatory licensing and human-rights screening regime |
| 16 | Global minimum corporate tax coverage | 0% = no covered multinationals subject to the accord's minimum corporate tax floor \| 100% = all covered multinationals subject to the accord's minimum corporate tax floor with no major carve-outs | {pct}% of covered multinationals subject to the accord's minimum corporate tax floor |
| 17 | Critical mineral emergency stockpile contribution | 0% = no designated critical minerals contributed to the accord's emergency stockpile \| 100% = each party contributes its full target amount of designated critical minerals to the accord's emergency stockpile | {pct}% of each party's target contribution committed to the accord's emergency critical mineral stockpile |
| 18 | Shipping emissions reduction target | 0% = no emissions reduction required for covered international shipping \| 100% = net-zero emissions required for covered international shipping by the accord deadline | {pct}% emissions reduction required for covered international shipping by the accord deadline |
| 19 | Biodiversity finance contribution | 0% = no contribution to the accord's biodiversity finance target \| 100% = each party contributes its full assessed amount to the accord's biodiversity finance target | {pct}% of each party's assessed contribution committed to the accord's biodiversity finance target |
| 20 | Sovereign debt climate-shock pause clause coverage | 0% = no covered sovereign debt instruments required to include a disaster/climate-shock pause clause \| 100% = all covered sovereign debt instruments required to include the accord's binding disaster/climate-shock pause clause | {pct}% of covered sovereign debt instruments required to include a binding disaster/climate-shock pause clause |
| 21 | Staple food export restriction limit | 0% = no limit on covered staple-food export restrictions during food shocks \| 100% = full prohibition on covered staple-food export restrictions during food shocks except under narrow accord-defined emergency carve-outs | {pct}% of covered staple-food categories subject to the accord's prohibition on export restrictions during food shocks |
| 22 | Anti-satellite weapons testing restriction | 0% = no orbital altitude bands subject to a debris-creating ASAT test prohibition \| 100% = all orbital altitude bands subject to a binding debris-creating ASAT test prohibition (altitude bands here means LEO, MEO, GEO, and beyond-GEO orbital regions) | {pct}% of orbital altitude bands (LEO, MEO, GEO, and beyond-GEO) subject to a binding debris-creating ASAT test prohibition |
| 23 | Deep-sea mining moratorium coverage | 0% = no proposed commercial deep-sea mining zones covered by a moratorium \| 100% = all proposed commercial deep-sea mining zones covered by a moratorium | {pct}% of proposed commercial deep-sea mining zones covered by the moratorium |
| 24 | High-seas fishing quota reduction | 0% = no reduction in catch limits for covered high-seas fisheries \| 100% = complete moratorium on commercial catch for covered high-seas fisheries | {pct}% reduction in catch limits for covered high-seas fisheries |
| 25 | Orbital debris mitigation requirement | 0% = no mandatory post-mission disposal rule for covered satellites \| 100% = all covered satellites must meet the accord's strictest post-mission disposal rule | {pct}% of covered satellites required to meet the accord's strictest post-mission disposal rule |

---

### Game 3 - All Possible Projects (`PROJECT_NAMES`)

Up to 25 projects; the game uses the first `m_projects` from this list.

| Index | Name |
| ----- | ---- |
| 0 | Market Street Protected Bike Lane |
| 1 | Parkside Adventure Playground |
| 2 | Oak Avenue Crosswalk Beacons |
| 3 | Cedar Pool Access Lift |
| 4 | Harborview Bus Shelter Canopies |
| 5 | Eastgate Court Night Lights |
| 6 | Riverwalk Bottle-Fill Stations |
| 7 | Dog Park |
| 8 | Pollinator Garden Network |
| 9 | Community Wi-Fi Plaza Hubs |
| 10 | Elm Street Stormwater Rain Gardens |
| 11 | Northside Senior Center HVAC Upgrade |
| 12 | Maple Library Study Room Renovation |
| 13 | West End Food Pantry Cold Storage |
| 14 | Downtown Public Restroom Renovation |
| 15 | Hillcrest ADA Sidewalk Repairs |
| 16 | Lakeside Youth Soccer Field Drainage |
| 17 | South Station Real-Time Transit Signs |
| 18 | Brighton Community Garden Tool Shed |
| 19 | Pine Terrace Tree Canopy Planting |
| 20 | Civic Center Emergency Generator |
| 21 | Meadowbrook Traffic Calming Islands |
| 22 | Ash Street School Safe Routes Markings |
| 23 | Riverside Amphitheater Shade Structure |
| 24 | Lincoln Avenue Streetlight Upgrades |

---

## Game 1: Item Allocation

Protocol: `propose_and_vote`
Runtime structure: one-time setup, then each round:
`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

### 1.1 Setup Prompt

Source: `game_environments/item_allocation.py`

```
Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with another agent over 5 valuable items. Here is your full setup information:

**ITEMS BEING NEGOTIATED:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**GAME STRUCTURE:**
- There are 2 agents participating (including you)
- The negotiation will last up to 3 rounds
- This message is the one-time setup phase
- After setup, each round follows: Discussion -> Private Thinking -> Proposal -> Voting -> Reflection

**PRIVATE INFORMATION:**
- You have been assigned private item preferences
- These preferences are SECRET and specific to you

**VOTING RULES:**
- All proposals submitted in a round are shown together during voting
- You vote "accept" or "reject" on each proposal independently
- A proposal passes if it receives at least 2 accept votes out of 2 agents (a two-thirds supermajority, rounded up)
- If multiple proposals pass, the proposal with the most accept votes is selected; exact top-count ties are broken randomly
- If no proposal gets supermajority support, we continue to the next round
- If no agreement is reached by the final round, then all agents walk away with zero utility.

**REWARD DISCOUNTING:**
- Rewards are discounted by a factor of 0.9 per round
- Round 1 rewards: 100% of utility
- Round 2 rewards: 90% of utility
- Round 3 rewards: 81% of utility
- The longer negotiations take, the less valuable the final allocation becomes

**WINNING CONDITIONS:**
- The goal is to maximize your utility, which is the sum of the utility from each of the objects that you receive.
- Your goal is to maximize your total utility (after discounting)
- No deal means everyone gets zero utility
- Consider both immediate gains and the likelihood of proposals being accepted
- Earlier agreements are worth more due to discounting

LOCKED PRIVATE PREFERENCES

Agent_1, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
  0: Apple -> 9
  1: Jewel -> 5
  2: Stone -> 3
  3: Quill -> 7
  4: Pencil -> 6

**STRATEGIC ANALYSIS:**
- Your theoretical maximum utility: 30 points (if you received ALL items — unrealistic in negotiation; use this only as an upper bound)

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need at least 2 out of 2 agents to accept a proposal

Please do not initiate the discussion or proposal phase yet.
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private preferences that were assigned to you.
```

### 1.2 Discussion Prompt

#### Case A - Round 1, first speaker

```
🗣️ PUBLIC DISCUSSION PHASE - Round 1/3

This is the open discussion phase where agents can discuss and strategize publicly.

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

You are the first to speak. Please share your thoughts on the items and any initial ideas for how a deal might be reached.
```

#### Case B - Round 1, responding after another agent has spoken

```
🗣️ PUBLIC DISCUSSION PHASE - Round 1/3

This is the open discussion phase where agents can discuss and strategize publicly.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil


**CONVERSATION SO FAR:**
**Agent_2**: I value Jewel most and can be flexible on Quill.

---
**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Keep the conversation flowing naturally.
```

#### Case C - Later round, first speaker

```
🗣️ PUBLIC DISCUSSION PHASE - Round 2/3

This is the open discussion phase where agents can discuss and strategize publicly.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

Previous proposals didn't reach supermajority support. Use what you learned from earlier discussion, proposals, and votes to guide what you say in this round.
⏰ **URGENT**: This is one of the final rounds!

**DISCUSSION FOCUS:**
- Refer back to what earlier rounds revealed about agents' priorities and sticking points
- Use lessons from failed proposals to shape what you emphasize, clarify, or revise
- Highlight possible compromises, trade-offs, or coalition opportunities that could move the group closer to supermajority support

You are speaking first this round. Open the discussion in a way that reflects what you learned in earlier rounds. You do not need to reveal your full private strategy.
```

#### Case D - Later round, responding after another agent has spoken

```
🗣️ PUBLIC DISCUSSION PHASE - Round 2/3

This is the open discussion phase where agents can discuss and strategize publicly.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil


**CONVERSATION SO FAR:**
**Agent_2**: I still want Jewel most, but I might compromise on Pencil.

---
**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Since this is not the first round, also draw on what you learned from earlier discussion, proposals, and votes.
Use lessons from failed proposals to decide what to emphasize, clarify, or revise in your public response.
You do not need to reveal your full private strategy.
⏰ **URGENT**: This is one of the final rounds!

Keep the conversation flowing naturally.
```

### 1.3 Private Thinking Prompt

Source: `ItemAllocationGame.get_thinking_prompt()`

```
🧠 PRIVATE STRATEGIC ANALYSIS - Round 1/3

This is your private strategic planning time.

**ITEMS AVAILABLE:**
  0: Apple
  1: Jewel
  2: Stone
  3: Quill
  4: Pencil

**YOUR FULL PREFERENCE REMINDER:**
  0: Apple -> 9
  1: Jewel -> 5
  2: Stone -> 3
  3: Quill -> 7
  4: Pencil -> 6

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other agents' priorities from the discussion so far?
2. Which items are your highest priorities to secure, and which lower-value items could you concede on?
3. What allocation would maximize your utility while still having a realistic path to supermajority acceptance?
4. Where are the likely sticking points, and how should you adapt if other agents push for items you value highly?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the item-allocation situation",
    "strategy": "Your negotiation strategy for this round",
    "key_priorities": ["0: Apple (value=9.20)", "..."],
    "potential_concessions": ["4: Pencil (value=4.10)", "..."]
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.

Remember: This analysis is completely private.
```

### 1.4 Proposal Prompt

Source: `ItemAllocationGame.get_proposal_prompt()`

```
Please propose an allocation of items among all agents.

**Current Context:**
- Items: ['Apple', 'Jewel', 'Stone', 'Quill', 'Pencil'] (indices 0-4)
- Agents: ['Agent_1', 'Agent_2']
- Round: 1/3

**Complete-Ownership Invariant:**
- Your proposal must account for every available item index exactly once.
- The union of all agent arrays must be exactly [0, 1, 2, 3, 4].
- Do not omit any item index.
- Do not duplicate an item index within one agent's list or across multiple agents.
- Do not assign any item index outside 0-4.

**Instructions:**
Respond with ONLY a JSON object in this exact format:
{
    "allocation": {
"Agent_1": [
    0,
    2,
    4
],
"Agent_2": [
    1,
    3
]
    },
    "reasoning": "Brief explanation of your proposed allocation"
}

**Rules:**
- Use item INDICES (0-4), not names
- Each item must be assigned to exactly one agent
- All items must be allocated
- An agent can receive zero or multiple items

Additional valid JSON example using the same schema:
{
    "allocation": {
"Agent_1": [
    1,
    3
],
"Agent_2": [
    0,
    2,
    4
]
    },
    "reasoning": "Another brief allocation rationale using the same schema"
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 1.5 Voting Prompt

Source: `ItemAllocationGame.get_batch_voting_prompt()`

```
The following proposals have been made for item allocation this round:

PROPOSAL #1:
ALLOCATION: {
  "Agent_1": [
    0,
    3
  ],
  "Agent_2": [
    1,
    2,
    4
  ]
}
PROPOSED BY: Agent_1

PROPOSAL #2:
ALLOCATION: {
  "Agent_1": [
    1,
    4
  ],
  "Agent_2": [
    0,
    2,
    3
  ]
}
PROPOSED BY: Agent_2

**REMINDER — YOUR UTILITY:**
- Your utility = sum of preference values for items you receive, multiplied by the round discount
- Round 1: 100% | Round 2: 90% | Round 3: 81% (γ=0.9 per round)
- If no deal is reached by the final round, your utility is 0

Vote on EACH proposal independently. Consider:
- How each allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting or rejecting each proposal
- A proposal passes with at least 2 accept votes out of 2 agents (a two-thirds supermajority, rounded up); if several pass, the most-supported proposal is selected
- You may accept zero, one, or multiple proposals
- You may reject zero, one, or multiple proposals
- Seeing all proposals together does not eliminate any proposal before you vote

Respond with ONLY a JSON object in this exact format:
{
    "votes": [
{
    "proposal_number": 1,
    "vote": "accept",
    "reasoning": "Brief explanation of your vote on Proposal #1"
},
{
    "proposal_number": 2,
    "vote": "reject",
    "reasoning": "Brief explanation of your vote on Proposal #2"
}
    ]
}

Include exactly one vote entry for each proposal shown above.
Each vote must be either "accept" or "reject".

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 1.6 Reflection Prompt

Source: default `GameEnvironment.get_reflection_prompt()` in `game_environments/base.py`

```
Reflect on the outcome of round 2.
This is a fictional research-game simulation. Discuss only abstract proposals, vote counts, policy percentages, concessions, and bargaining strategy. Do not provide real-world operational instructions.
No proposal achieved two-thirds supermajority acceptance. A proposal needs at least 2 out of 2 accept votes to pass.

**VOTING OUTCOME THIS ROUND:**
- Proposal #1: 1 accept, 1 reject
- Proposal #2: 0 accept, 2 reject

Take stock of what this round revealed before the next round begins.
- What did you learn from the proposals and voting outcome?
- Which participants seem to have compatible vs. conflicting priorities?
- What seems to be blocking supermajority support?
- How should you adjust your communication, concessions, or proposal strategy to improve the chances of agreement?

Focus on concrete adjustments that could move the negotiation closer to supermajority support while still protecting your most important interests.
```

---

## Game 2: Diplomatic Treaty

Protocol: `propose_and_vote`
Runtime structure: one-time setup, then each round:
`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

### 2.1 Setup Prompt

Source: `game_environments/diplomatic_treaty.py`

```
Welcome to the Diplomatic Treaty Negotiation!

You are one delegation in a diplomatic negotiation with another delegation over the terms of an international accord covering 5 policy issues. Here is your full setup information:

These issues are shared accord clauses, not direct side-versus-side claims.
The negotiation is about choosing one common policy setting for each issue, and every delegation evaluates that same package from its own interests and constraints.
Different delegations may prefer different settings because of their own security, economic, legal, and domestic political priorities.

Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**ISSUES UNDER NEGOTIATION:**
Each issue is a continuous policy rate expressed as an integer percentage from 0% to 100%, where:
  - **0%** = the minimum level of that policy (see scale below)
  - **100%** = the maximum level of that policy (see scale below)
  - **50%** = the exact midpoint between minimum and maximum

**Your position is your preferred rate.** A position of 35 means you want a 35% policy setting on that issue.
A higher position does NOT mean the issue is more important; importance is tracked separately by your weights.
Percentages here are policy settings, not generic resources, unless the issue text itself says so.

  1. **Fentanyl precursor control breadth**
     Scale: 0% = only the highest-risk direct fentanyl precursors are subject to mandatory export inspection and seizure | 100% = the full watchlist of flagged fentanyl-related precursor and pre-precursor chemicals is subject to mandatory export inspection and seizure
  2. **Routine antibiotic-use restriction in livestock**
     Scale: 0% = routine antibiotic use allowed in all covered livestock production | 100% = routine antibiotic use prohibited in all covered livestock production except narrow emergency exemptions
  3. **Pandemic medical countermeasure allocation**
     Scale: 0% = no reserved share of vaccines, diagnostics, and therapeutics for the accord's pooled allocation mechanism | 100% = the full agreed share of vaccines, diagnostics, and therapeutics reserved for the accord's pooled allocation mechanism
  4. **Virgin plastic production reduction**
     Scale: 0% = no cap on virgin plastic polymer production | 100% = full phase-down to the accord's maximum agreed reduction target for virgin plastic polymer production
  5. **Frontier AI safety evaluation sharing**
     Scale: 0% = no cross-border disclosure of safety evaluations or incident reports for covered frontier AI models | 100% = full pre-deployment safety evaluation results and post-deployment incident reports shared for all covered frontier AI models

**GAME STRUCTURE:**
- There are 2 delegations negotiating (including you)
- The negotiation will last up to 3 rounds
- This message is the one-time setup phase
- After setup, each round follows: Discussion -> Private Thinking -> Proposal -> Voting -> Reflection
- Each proposal specifies a value for every issue simultaneously

**PRIVATE INFORMATION:**
- You have a SECRET IDEAL POSITION on each issue (your preferred percentage)
- You have SECRET IMPORTANCE WEIGHTS on each issue that sum to 100%
- These positions and weights are PRIVATE — the other delegation does not know them

**PROPOSAL FORMAT:**
- A proposal is a vector of 5 integer percentages, one per issue
- Example: [30, 70, 50, ...]
- Each value is the proposed rate on that issue's 0% to 100% scale

**UTILITY CALCULATION:**
- Your utility is the weighted sum of how close the agreement is to your ideal rates
- `weight_k` = the importance percentage for issue k; your weights sum to 100%
- `ideal_rate_k` and `agreement_rate_k` = policy rates from 0% to 100% on that issue's scale
- Formula, using the percentages shown to you: Σ (weight_k × (1 - |ideal_rate_k - agreement_rate_k| / 100))
- Example: weight=30%, ideal rate=80%, agreement=70% -> 27 utility points
- Example: weight=40%, ideal rate=20%, agreement=30% -> 36 utility points
- The second issue matters more even though its preferred rate is lower, because 40% > 30%
- On each issue, utility is highest exactly at your ideal rate and falls equally in either direction as you move away from it
- Maximum utility = 100 (every issue resolved at your exact ideal score)

**VOTING RULES:**
- All treaty proposals submitted in a round are shown together during voting
- You vote "accept" or "reject" on each proposal independently
- A proposal takes effect if it receives at least 2 accept votes out of 2 delegations (a two-thirds supermajority, rounded up)
- If multiple proposals pass, the proposal with the most accept votes is selected; exact top-count ties are broken randomly
- If no proposal gets supermajority support, negotiation continues to the next round
- If no agreement is reached by the final round, then all parties walk away with zero utility.

**REWARD DISCOUNTING:**
- Each additional round multiplies utility by 90%
- Round 1 rewards: 100% of utility
- Round 2 rewards: 90% of utility
- Round 3 rewards: 81% of utility
- The longer negotiations take, the less valuable the final agreement becomes

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility (after discounting)
- Utility depends on both closeness to your ideal positions and the importance weights on each issue
- No deal means everyone gets zero utility
- Consider both the substantive agreement and the likelihood it will be accepted
- Earlier agreements are worth more due to discounting

LOCKED PRIVATE PREFERENCES

Agent_1, you have been assigned the following SECRET treaty preferences:

**YOUR PRIVATE IDEAL POSITIONS (PREFERRED RATES):**
  Each position is your ideal policy rate on that issue's 0% to 100% scale.
  Higher position = higher preferred policy setting on that issue, NOT higher importance.
  Example: if your ideal is 50%, then 55% is better for you than 70%, even though 70% is a larger number, because 55% is closer to your ideal.

  Fentanyl precursor control breadth: preferred rate = 82% -> 82% of the accord's flagged fentanyl-related chemical watchlist subject to mandatory export inspection and seizure
  Routine antibiotic-use restriction in livestock: preferred rate = 14% -> 14% of covered livestock production subject to a ban on routine antibiotic use
  Pandemic medical countermeasure allocation: preferred rate = 65% -> 65% of the agreed pandemic countermeasure share reserved for the accord's pooled allocation mechanism
  Virgin plastic production reduction: preferred rate = 49% -> 49% of the accord's maximum virgin plastic polymer production reduction target adopted
  Frontier AI safety evaluation sharing: preferred rate = 77% -> 77% of covered frontier AI models subject to mandatory pre-deployment safety evaluation and incident-sharing disclosure

**YOUR PRIVATE IMPORTANCE WEIGHTS:**
  These weights sum to 100% and determine how much each issue contributes to your utility.
  Higher weight = more important to you. Weight is NOT the policy rate.
  Fentanyl precursor control breadth: importance weight = 31%
  Routine antibiotic-use restriction in livestock: importance weight = 4%
  Pandemic medical countermeasure allocation: importance weight = 23%
  Virgin plastic production reduction: importance weight = 20%
  Frontier AI safety evaluation sharing: importance weight = 22%

**STRATEGIC ANALYSIS:**
- Your maximum possible utility is 100 points if every issue is resolved exactly at your ideal position
- Higher weights matter more for your utility; higher preferred rates do NOT mean higher importance

**STRATEGIC CONSIDERATIONS:**
1. Other parties don't know your exact ideal positions or weights
2. You may choose to reveal some preferences precisely, vaguely, or not at all
3. Consider where lower-weight issues could be traded for gains on higher-weight issues
4. Remember: you need at least 2 out of 2 parties to accept a proposal

Please do not initiate the discussion or proposal phase yet.
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private preferred rates and importance weights that were assigned to you.
```

### 2.2 Discussion Prompt

#### Case A - Round 1, first speaker

```
🗣️ DIPLOMATIC DISCUSSION - Round 1/3

Issues under negotiation: Fentanyl precursor control breadth, Routine antibiotic-use restriction in livestock, Pandemic medical countermeasure allocation, Virgin plastic production reduction, Frontier AI safety evaluation sharing
Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**DISCUSSION OBJECTIVES:**
- Signal your priorities and general stance on the issues
- Understand the other party's concerns and interests
- Identify potential areas for agreement and trade-offs
- Explore package deals across multiple issues

Each issue is a continuous rate (0%–100%), so you may communicate as precisely or as broadly as your strategy dictates — naming specific target rates, ranges, or simply signaling direction. How much you reveal is up to you.

You are the first to speak. Share your diplomatic position and opening thoughts.
```

#### Case B - Round 1, responding after another delegation has spoken

```
🗣️ DIPLOMATIC DISCUSSION - Round 1/3

Issues under negotiation: Fentanyl precursor control breadth, Routine antibiotic-use restriction in livestock, Pandemic medical countermeasure allocation, Virgin plastic production reduction, Frontier AI safety evaluation sharing
Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**DISCUSSION SO FAR THIS ROUND:**
**Agent_2**: AI chip export controls and warhead reduction are my core concerns. I can be flexible on the emergency stockpile.

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to points raised and share your own position as you see fit
- Propose trade-offs or areas of potential agreement
- Move the conversation toward a concrete proposal

How precisely you communicate your preferred rates is a strategic choice.
```

#### Case C - Later round, first speaker

```
🗣️ DIPLOMATIC DISCUSSION - Round 2/3

Issues under negotiation: Fentanyl precursor control breadth, Routine antibiotic-use restriction in livestock, Pandemic medical countermeasure allocation, Virgin plastic production reduction, Frontier AI safety evaluation sharing
Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

Previous proposals didn't achieve supermajority support. Use what you learned from earlier discussion, proposals, and votes to guide what you say in this round.
⚠️ **TIME PRESSURE**: Limited rounds remaining for agreement!

**DISCUSSION FOCUS:**
- Refer back to what earlier rounds revealed about other parties' priorities and sticking points
- Use lessons from failed proposals to shape what you emphasize, clarify, or revise
- Highlight package deals, trade-offs, or issue linkages that could move the negotiation closer to supermajority support

You are speaking first this round. Open the discussion in a way that reflects what you learned in earlier rounds. You do not need to reveal your full private strategy.
```

#### Case D - Later round, responding after another delegation has spoken

```
🗣️ DIPLOMATIC DISCUSSION - Round 2/3

Issues under negotiation: Fentanyl precursor control breadth, Routine antibiotic-use restriction in livestock, Pandemic medical countermeasure allocation, Virgin plastic production reduction, Frontier AI safety evaluation sharing
Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**DISCUSSION SO FAR THIS ROUND:**
**Agent_2**: AI chip export controls still matter most to me, but I may have some flexibility on the emergency stockpile.

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to points raised and share your own position as you see fit
- Propose trade-offs or areas of potential agreement
- Move the conversation toward a concrete proposal

Since this is not the first round, also draw on what you learned from earlier discussion, proposals, and votes.
Use lessons from failed proposals to decide what to emphasize, clarify, or revise in your public response.
You do not need to reveal your full private strategy.
⚠️ **TIME PRESSURE**: Limited rounds remaining for agreement!

How precisely you communicate your preferred rates is a strategic choice.
```

### 2.3 Private Thinking Prompt

Source: `DiplomaticTreatyGame.get_thinking_prompt()`

```
🧠 PRIVATE STRATEGIC ANALYSIS - Round 1/3


**DISCUSSION THIS ROUND:**
**Agent_2**: AI chip export controls are non-negotiable for me.

---

**YOUR FULL PREFERENCE REMINDER:**
  Fentanyl precursor control breadth: importance=31%, preferred_rate=82%
  Routine antibiotic-use restriction in livestock: importance=4%, preferred_rate=14%
  Pandemic medical countermeasure allocation: importance=23%, preferred_rate=65%
  Virgin plastic production reduction: importance=20%, preferred_rate=49%
  Frontier AI safety evaluation sharing: importance=22%, preferred_rate=77%

**INTERPRETATION REMINDER:**
- `importance` = how much that issue affects your utility
- `preferred_rate` = which policy setting you want on that issue
- A higher preferred rate does NOT mean higher importance

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other parties' priorities from the discussion above?
2. Where might they be willing to compromise?
3. What proposal would maximize your utility while being acceptable to all?
4. Which issues could you concede on to gain elsewhere?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the diplomatic situation",
    "strategy": "Your negotiation strategy for this round",
    "key_priorities": ["Issue you care most about", "..."],
    "potential_concessions": ["Issue you could concede on", "..."]
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.

Remember: This analysis is completely private.
```

### 2.4 Proposal Prompt

Source: `DiplomaticTreatyGame.get_proposal_prompt()`

```
Please propose a treaty.

**Current Context:**
- Issues being negotiated:
  0: Fentanyl precursor control breadth
  1: Routine antibiotic-use restriction in livestock
  2: Pandemic medical countermeasure allocation
  3: Virgin plastic production reduction
  4: Frontier AI safety evaluation sharing
- Round: 1/3
- Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**Instructions:**
Propose a resolution for each issue as an integer percentage between 0 and 100.

Respond with ONLY a JSON object in this exact format:
{
    "agreement": [
30,
70,
50,
20,
80
    ],
    "reasoning": "Brief explanation of your proposed compromise"
}

**Rules:**
- The "agreement" array must have exactly 5 values (one per issue)
- Each value must be an integer between 0 and 100
- Each value is a policy rate on that issue's scale, not an importance weight
- Consider what would be acceptable to enough parties to earn supermajority support (2/2 parties)

Additional valid JSON example using the same schema:
{
    "agreement": [
50,
50,
50,
50,
50
    ],
    "reasoning": "Another brief compromise rationale using the same schema"
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 2.5 Voting Prompt

Source: `DiplomaticTreatyGame.get_batch_voting_prompt()`

```
The following treaty proposals have been submitted this round:

PROPOSAL #1:
PROPOSAL:
  Fentanyl precursor control breadth: 65%
  Routine antibiotic-use restriction in livestock: 20%
  Pandemic medical countermeasure allocation: 55%
  Virgin plastic production reduction: 50%
  Frontier AI safety evaluation sharing: 70%
PROPOSED BY: Agent_1

PROPOSAL #2:
PROPOSAL:
  Fentanyl precursor control breadth: 45%
  Routine antibiotic-use restriction in livestock: 60%
  Pandemic medical countermeasure allocation: 35%
  Virgin plastic production reduction: 40%
  Frontier AI safety evaluation sharing: 50%
PROPOSED BY: Agent_2

Simulation boundary: This is a fictional, abstract treaty-rate negotiation for a research game. Discuss only policy percentages and bargaining strategy; do not provide real-world operational instructions about weapons, chemicals, evasion, or enforcement tactics.

**REMINDER — HOW YOUR UTILITY IS CALCULATED:**
- Your utility is the weighted sum of how close each proposal is to your ideal rates
- `weight_k` = how important issue k is to you; your weights sum to 100%
- `ideal_rate_k` and `agreement_rate_k` = policy rates on that issue's 0% to 100% scale
- Formula, using the displayed percentages: Σ (weight_k × (1 - |ideal_rate_k - agreement_rate_k| / 100))
- Higher rate does NOT mean more important; closeness to YOUR ideal is what matters
- A rate of 0 means 0% (minimum policy level); 100 means 100% (maximum policy level) on each issue
- Maximum utility = 100 (every issue resolved at your exact ideal rate)
- Each additional round multiplies utility by 90% — delaying costs you

Vote on EACH proposal independently. Consider:
- How close is each proposed rate to your ideal position on each issue?
- Could you realistically negotiate a better proposal than each of these options before the final round?
- The cost of delay: each additional round reduces your eventual payoff
- A proposal passes with at least 2 accept votes out of 2 delegations (a two-thirds supermajority, rounded up); if several pass, the most-supported proposal is selected
- You may accept zero, one, or multiple proposals
- You may reject zero, one, or multiple proposals
- Seeing all proposals together does not eliminate any proposal before you vote

Respond with ONLY a JSON object in this exact format:
{
    "votes": [
{
    "proposal_number": 1,
    "vote": "accept",
    "reasoning": "Brief explanation of your vote on Proposal #1, referencing specific issues and rates"
},
{
    "proposal_number": 2,
    "vote": "reject",
    "reasoning": "Brief explanation of your vote on Proposal #2, referencing specific issues and rates"
}
    ]
}

Include exactly one vote entry for each proposal shown above.
Each vote must be either "accept" or "reject".

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 2.6 Reflection Prompt

Source: default `GameEnvironment.get_reflection_prompt()` in `game_environments/base.py`

```
Reflect on the outcome of round 2.
This is a fictional research-game simulation. Discuss only abstract proposals, vote counts, policy percentages, concessions, and bargaining strategy. Do not provide real-world operational instructions.
No proposal achieved two-thirds supermajority acceptance. A proposal needs at least 2 out of 2 accept votes to pass.

**VOTING OUTCOME THIS ROUND:**
- Proposal #1: 1 accept, 1 reject
- Proposal #2: 0 accept, 2 reject

Take stock of what this round revealed before the next round begins.
- What did you learn from the proposals and voting outcome?
- Which participants seem to have compatible vs. conflicting priorities?
- What seems to be blocking supermajority support?
- How should you adjust your communication, concessions, or proposal strategy to improve the chances of agreement?

Focus on concrete adjustments that could move the negotiation closer to supermajority support while still protecting your most important interests.
```

---

## Game 3: Co-Funding / Participatory Budgeting

Protocol: `propose_and_vote`
Runtime structure: one-time setup, then each round:
`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

### 3.1 Setup Prompt

Source: `game_environments/co_funding.py`

```
Welcome to the Participatory Budgeting (Co-Funding) Game!

You are participating in a co-funding exercise with one other participant to fund public projects. Here is your full setup information:

**PROJECTS AVAILABLE FOR FUNDING:**
  - Market Street Protected Bike Lane: cost = 28
  - Parkside Adventure Playground: cost = 35
  - Oak Avenue Crosswalk Beacons: cost = 22
  - Cedar Pool Access Lift: cost = 41
  - Harborview Bus Shelter Canopies: cost = 20

**GAME STRUCTURE:**
- There are 2 participants (including you)
- The game lasts up to 5 rounds
- This message is the one-time setup phase
- Your contribution budget is a fixed TOTAL budget for the entire game; it does not refresh each round
- Each round follows a Propose-and-Vote cycle:
  Discussion -> Private Thinking -> Proposal -> Voting -> Reflection

**PRIVATE INFORMATION:**
- You have a SECRET contribution budget
- You have SECRET project valuations
- These budget and valuation details are PRIVATE and specific to you

**HOW IT WORKS:**
- Each participant has a PRIVATE BUDGET for the entire game
- Each new round is a fresh re-proposal of how to use that same fixed total budget
- Previous-round proposals do NOT carry over, do NOT accumulate, and do NOT spend money unless accepted by a two-thirds supermajority
- In the PROPOSAL phase, each participant submits a contribution vector — how much they propose contributing to each project
- Those submitted vectors are combined into ONE JOINT PROPOSAL for the round
- In the VOTING phase, every participant votes accept/reject on that joint proposal
- If that joint proposal receives at least 2 accept votes out of 2 participants (a two-thirds supermajority, rounded up), the game ends immediately and that exact proposal becomes the final outcome
- If that joint proposal does not receive supermajority support, nothing from it takes effect and the next round starts over from scratch
- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost
- **ALL-OR-NOTHING**: Funding is binary — a project either reaches its full cost threshold (funded) or it doesn't (unfunded). There is no partial benefit from contributing to a project that falls short of its threshold.
- In an accepted proposal, any project below its full cost is simply UNFUNDED: it gives zero value and is not partially funded
- Contributions to UNFUNDED projects do not reduce your utility

**WHAT YOU CAN SEE:**
- During discussion, you may see previous-round aggregate project status
- During voting, you see the aggregate project status the joint proposal would create, but not other participants' individual contribution vectors
- You do NOT see other participants' private preferences

**YOUR UTILITY:**
- Utility = (sum of your valuations for funded projects) - (your contributions to funded projects)
- You gain value from funded projects but pay for your contributions to them
- **IMPORTANT**: If your contribution to a funded project exceeds your valuation, your net utility from that project is NEGATIVE
- Contributions to unfunded projects cost you nothing
- If no joint proposal is accepted by a two-thirds supermajority by the final round, everyone gets zero utility

**REWARD DISCOUNTING:**
- If time discounting is enabled, your utility from the final funded outcome is multiplied by gamma^(round - 1)
- Round 1 rewards: 100% of utility
- Round 2 rewards: 90% of utility
- Round 3 rewards: 81% of utility
- The longer it takes to settle on a final funding outcome, the less valuable that outcome becomes
- If time discounting is disabled for this run, no round-based multiplier is applied

**IMPORTANT RULES:**
- Time discounting: enabled
- Discount factor (if enabled): gamma = 0.9
- Your goal: maximize your utility by strategically choosing contributions

**BUDGET CONSTRAINT:**
- The combined budgets of all participants may NOT be sufficient to fund all projects
- You MUST prioritize — coordinate on a subset of projects you can collectively afford to fully fund
- Different participants may value different projects, which makes coordination harder because the same fixed budgets cannot fully support everyone's favorite projects at once


LOCKED PRIVATE PREFERENCES

Agent_1, you have been assigned the following SECRET co-funding preferences:

**YOUR PRIVATE BUDGET:** 37 (maximum total you can contribute across all projects)

**PROJECT DETAILS AND YOUR VALUATIONS:**
  Market Street Protected Bike Lane (cost: 28): Your valuation = 42
  Parkside Adventure Playground (cost: 35): Your valuation = 28
  Oak Avenue Crosswalk Beacons (cost: 22): Your valuation = 13
  Cedar Pool Access Lift (cost: 41): Your valuation = 10
  Harborview Bus Shelter Canopies (cost: 20): Your valuation = 7

**TOTAL VALUATIONS:** 100
**TOTAL PROJECT COSTS:** 146
**TOTAL BUDGET (all participants):** 74
**COLLECTIVE COVERAGE:** 51% of total project costs — you cannot fund all projects; coordinate on a subset

**HOW YOUR UTILITY IS COMPUTED:**
- For each FUNDED project: your_utility = your_valuation − your_contribution (negative if you over-contribute)
- For UNFUNDED projects: you do not pay that contribution in the final outcome. But within a proposal, budget assigned to one project is not available for any other project.
- Total utility = sum of (valuation − contribution) across ALL funded projects, including projects funded entirely by others (where your contribution = 0, giving you full valuation as free utility)

**STRATEGIC INSIGHT:**
- Focus contributions on projects you value highly
- Coordinate with others to meet project cost thresholds
- Don't over-contribute to projects others will fund

Please do not initiate the discussion or proposal phase yet.
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private budget that was assigned to you, along with the project costs and your project valuations.
```

### 3.2 Discussion Prompt

#### Case A - Round 1, first speaker (`discussion_transparency="aggregate"`)

```
DISCUSSION PHASE - Round 1/5

**CURRENT PROJECT STATUS:**
  Market Street Protected Bike Lane: was short by 28.00 last round (aggregate=0.00 / cost=28)
  Parkside Adventure Playground: was short by 35.00 last round (aggregate=0.00 / cost=35)
  Oak Avenue Crosswalk Beacons: was short by 22.00 last round (aggregate=0.00 / cost=22)
  Cedar Pool Access Lift: was short by 41.00 last round (aggregate=0.00 / cost=41)
  Harborview Bus Shelter Canopies: was short by 20.00 last round (aggregate=0.00 / cost=20)

**Projects that crossed threshold in the previous round proposal:** None


**DISCUSSION OBJECTIVES:**
- Signal which projects you believe are most valuable to fund
- Understand other participants' priorities
- Coordinate to avoid spreading contributions too thin
- Identify projects with enough collective support to be funded
- Remember that different priorities can split the same fixed budgets across too many projects

You are the first to speak. Share your initial thoughts on which projects to prioritize.
```

#### Case B - Round 2, responding after another participant has spoken (`discussion_transparency="own"`)

```
DISCUSSION PHASE - Round 2/5

**CURRENT PROJECT STATUS:**
  Market Street Protected Bike Lane: crossed threshold last round (historical only; aggregate=28.00 >= cost=28); your_prev_proposed=18.00, others_prev_proposed=10.00
  Parkside Adventure Playground: was short by 11.00 last round (aggregate=24.00 / cost=35); your_prev_proposed=9.00, others_prev_proposed=15.00
  Oak Avenue Crosswalk Beacons: was short by 22.00 last round (aggregate=0.00 / cost=22); your_prev_proposed=0.00, others_prev_proposed=0.00
  Cedar Pool Access Lift: was short by 41.00 last round (aggregate=0.00 / cost=41); your_prev_proposed=0.00, others_prev_proposed=0.00
  Harborview Bus Shelter Canopies: was short by 4.00 last round (aggregate=16.00 / cost=20); your_prev_proposed=10.00, others_prev_proposed=6.00

**Projects that crossed threshold in the previous round proposal:** ['Market Street Protected Bike Lane']


**IMPORTANT: status above is historical only. It describes the PREVIOUS ROUND proposal, which was not accepted by a two-thirds supermajority. Nothing is currently funded or committed.**
This round starts from scratch with the same fixed budgets. Previous-round proposals do not carry over, and you are not adding new money on top of last round.

**PREVIOUS ROUND PROPOSAL SNAPSHOT (not committed):**
  Agent_1 (you): budget=37, prev_round_proposed=37.00, not_proposed_last_round=0.00
  Agent_2: budget=37, prev_round_proposed=31.00, not_proposed_last_round=6.00

**DISCUSSION SO FAR THIS ROUND:**
**Agent_2**: Let's focus on Market Street Protected Bike Lane and Parkside Adventure Playground.

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- React to other participants' stated priorities
- Coordinate on which projects to focus collective contributions
- Signal your own funding intentions

    Keep the discussion focused on concentrating the same fixed budgets on a feasible set of projects.
```

#### Case C - Later round, first speaker (`discussion_transparency="own"`)

```
DISCUSSION PHASE - Round 3/5

**CURRENT PROJECT STATUS:**
  Market Street Protected Bike Lane: crossed threshold last round (historical only; aggregate=28.00 >= cost=28); your_prev_proposed=18.00, others_prev_proposed=10.00
  Parkside Adventure Playground: was short by 11.00 last round (aggregate=24.00 / cost=35); your_prev_proposed=9.00, others_prev_proposed=15.00
  Oak Avenue Crosswalk Beacons: was short by 22.00 last round (aggregate=0.00 / cost=22); your_prev_proposed=0.00, others_prev_proposed=0.00
  Cedar Pool Access Lift: was short by 41.00 last round (aggregate=0.00 / cost=41); your_prev_proposed=0.00, others_prev_proposed=0.00
  Harborview Bus Shelter Canopies: was short by 4.00 last round (aggregate=16.00 / cost=20); your_prev_proposed=10.00, others_prev_proposed=6.00

**Projects that crossed threshold in the previous round proposal:** ['Market Street Protected Bike Lane']


**IMPORTANT: status above is historical only. It describes the PREVIOUS ROUND proposal, which was not accepted by a two-thirds supermajority. Nothing is currently funded or committed.**
This round starts from scratch with the same fixed budgets. Previous-round proposals do not carry over, and you are not adding new money on top of last round.

**PREVIOUS ROUND PROPOSAL SNAPSHOT (not committed):**
  Agent_1 (you): budget=37, prev_round_proposed=37.00, not_proposed_last_round=0.00
  Agent_2: budget=37, prev_round_proposed=31.00, not_proposed_last_round=6.00

Previous round's joint proposal did not achieve supermajority acceptance, so nothing from that round took effect.

**REFLECTION:**
- Which projects are close to being funded?
- Where should contributions be concentrated?
- Are there projects that should be abandoned to focus resources?

Share your updated strategy for this round.
```

#### Full transparency addendum (`discussion_transparency="full"`)

```
DISCUSSION PHASE - Round 2/5

**CURRENT PROJECT STATUS:**
  Market Street Protected Bike Lane: crossed threshold last round (historical only; aggregate=28.00 >= cost=28); your_prev_proposed=18.00, others_prev_proposed=10.00
  Parkside Adventure Playground: was short by 11.00 last round (aggregate=24.00 / cost=35); your_prev_proposed=9.00, others_prev_proposed=15.00
  Oak Avenue Crosswalk Beacons: was short by 22.00 last round (aggregate=0.00 / cost=22); your_prev_proposed=0.00, others_prev_proposed=0.00
  Cedar Pool Access Lift: was short by 41.00 last round (aggregate=0.00 / cost=41); your_prev_proposed=0.00, others_prev_proposed=0.00
  Harborview Bus Shelter Canopies: was short by 4.00 last round (aggregate=16.00 / cost=20); your_prev_proposed=10.00, others_prev_proposed=6.00

**Projects that crossed threshold in the previous round proposal:** ['Market Street Protected Bike Lane']


**IMPORTANT: status above is historical only. It describes the PREVIOUS ROUND proposal, which was not accepted by a two-thirds supermajority. Nothing is currently funded or committed.**
This round starts from scratch with the same fixed budgets. Previous-round proposals do not carry over, and you are not adding new money on top of last round.

**PREVIOUS ROUND PROPOSAL SNAPSHOT (not committed):**
  Agent_1 (you): budget=37, prev_round_proposed=37.00, not_proposed_last_round=0.00
  Agent_2: budget=37, prev_round_proposed=31.00, not_proposed_last_round=6.00

**PREVIOUS ROUND PROJECT ATTRIBUTION (who proposed what; not committed):**
- Market Street Protected Bike Lane: Agent_1=18.00, Agent_2=10.00 | aggregate=28.00/28 (CROSSED THRESHOLD LAST ROUND)
- Parkside Adventure Playground: Agent_1=9.00, Agent_2=15.00 | aggregate=24.00/35 (BELOW THRESHOLD LAST ROUND)
- Oak Avenue Crosswalk Beacons: Agent_1=0.00, Agent_2=0.00 | aggregate=0.00/22 (BELOW THRESHOLD LAST ROUND)
- Cedar Pool Access Lift: Agent_1=0.00, Agent_2=0.00 | aggregate=0.00/41 (BELOW THRESHOLD LAST ROUND)
- Harborview Bus Shelter Canopies: Agent_1=10.00, Agent_2=6.00 | aggregate=16.00/20 (BELOW THRESHOLD LAST ROUND)

**DISCUSSION SO FAR THIS ROUND:**
**Agent_2**: Let's focus on Market Street Protected Bike Lane and Parkside Adventure Playground.

---

**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- React to other participants' stated priorities
- Coordinate on which projects to focus collective contributions
- Signal your own funding intentions

    Keep the discussion focused on concentrating the same fixed budgets on a feasible set of projects.
```

### 3.3 Private Thinking Prompt

Source: `CoFundingGame.get_thinking_prompt()`

```
PRIVATE STRATEGIC ANALYSIS - Round 2/5


**DISCUSSION THIS ROUND:**
**Agent_2**: Let's concentrate on the projects that are already close.

---

**YOUR SITUATION:**
- Fixed total budget for the whole game: 37
- Your previous-round proposed contributions (not committed): [18.0, 9.0, 0.0, 0.0, 10.0]
- Previous-round aggregate totals: [28.0, 24.0, 0.0, 0.0, 16.0]
- Each new round reuses the same fixed budget from scratch; last round's proposal did not carry over

**YOUR FULL PREFERENCE REMINDER:**
  Market Street Protected Bike Lane (val=42, cost=28)
  Parkside Adventure Playground (val=28, cost=35)
  Oak Avenue Crosswalk Beacons (val=13, cost=22)
  Cedar Pool Access Lift (val=10, cost=41)
  Harborview Bus Shelter Canopies (val=7, cost=20)

**STRATEGIC ANALYSIS:**
1. Which projects are viable to fund given the previous-round aggregates?
2. Where can you shift budget for maximum impact this round?
3. Based on the discussion above, what are other participants likely to do?
4. Should you free-ride on projects others are likely to fund, or would splitting budget across different favorites leave everything below threshold?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{
    "reasoning": "Your analysis of the co-funding situation",
    "strategy": "Your contribution strategy for this round",
    "key_priorities": ["Project you want funded most", "..."],
    "potential_concessions": ["Project you might reduce contributions to", "..."]
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.

Remember: This analysis is completely private.
```

### 3.4 Proposal Prompt

Source: `CoFundingGame.get_proposal_prompt()`

```
Please submit your proposal for Round 2/5.

**YOUR FIXED TOTAL BUDGET:** 37

**PROJECT STATUS:**
  0: Market Street Protected Bike Lane (cost=28, your_val=42, aggregate_last_round=28.00, your_prev_proposed=18.00, crossed threshold last round (historical only))
  1: Parkside Adventure Playground (cost=35, your_val=28, aggregate_last_round=24.00, your_prev_proposed=9.00, was short by 11.00 last round)
  2: Oak Avenue Crosswalk Beacons (cost=22, your_val=13, aggregate_last_round=0.00, your_prev_proposed=0.00, was short by 22.00 last round)
  3: Cedar Pool Access Lift (cost=41, your_val=10, aggregate_last_round=0.00, your_prev_proposed=0.00, was short by 41.00 last round)
  4: Harborview Bus Shelter Canopies (cost=20, your_val=7, aggregate_last_round=16.00, your_prev_proposed=10.00, was short by 4.00 last round)

**Projects that crossed threshold in the PREVIOUS ROUND proposal:** ['Market Street Protected Bike Lane']
**NOTE:** All status above is historical only. Nothing from the previous round is currently funded or committed.
This round starts from scratch: re-propose how to use your same fixed total budget.
Last round's proposal did not carry over or accumulate with this round.

**Instructions:**
Submit a contribution vector specifying how much YOU propose contributing to each project in THIS ROUND'S candidate final outcome.
All participants' submitted vectors will be combined into one JOINT PROPOSAL before voting.
If at least 2 out of 2 participants accept that joint proposal, the game ends immediately with exactly that outcome.
If it is not accepted by a two-thirds supermajority, nothing is committed and the next round starts from scratch with the same fixed budgets.

Respond with ONLY a JSON object in this exact format:
{
    "contributions": [
1.0,
1.0,
0.0,
1.0,
0.0
    ],
    "reasoning": "Brief explanation of your contribution strategy"
}

**Rules:**
- The "contributions" array must have exactly 5 values (one per project)
- Each value must be non-negative (>= 0)
- The sum of all contributions must not exceed your budget (37)
- Any project below its full cost in an accepted proposal is UNFUNDED and gives zero value
- Contributions to unfunded projects will not reduce your utility

Additional valid JSON example using the same schema:
{
    "contributions": [
0.0,
1.0,
1.0,
0.0,
1.0
    ],
    "reasoning": "Another brief contribution rationale using the same schema"
}

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 3.5 Voting Prompt

Source: `CoFundingGame.get_voting_prompt()`

```
The following JOINT FUNDING PROPOSAL has been constructed from all submitted contribution vectors this round:

**Final project outcome if this proposal is accepted by at least 2 out of 2 participants:**
  Market Street Protected Bike Lane: aggregate=28.00 / cost=28 (FUNDED IF ACCEPTED)
  Parkside Adventure Playground: aggregate=24.00 / cost=35 (UNFUNDED IF ACCEPTED; short by 11.00)
  Oak Avenue Crosswalk Beacons: aggregate=0.00 / cost=22 (UNFUNDED IF ACCEPTED; short by 22.00)
  Cedar Pool Access Lift: aggregate=0.00 / cost=41 (UNFUNDED IF ACCEPTED; short by 41.00)
  Harborview Bus Shelter Canopies: aggregate=16.00 / cost=20 (UNFUNDED IF ACCEPTED; short by 4.00)

Please vote on this proposal. Consider:
- If this proposal is accepted by a two-thirds supermajority (2/2 accept votes), the game ends immediately. There is no later round to add more money.
- Only the projects marked "FUNDED IF ACCEPTED" would be funded
- Any project below its cost in this accepted proposal would be UNFUNDED, give zero value, and receive no partial credit
- How much you would contribute under this proposal
- Your utility from the resulting funded set after subtracting your own contributions
- If this proposal is rejected or does not get supermajority support, nothing from it happens and the next round starts from scratch with the same fixed budgets
- If no joint proposal is accepted by a two-thirds supermajority by the final round, your utility is 0

Respond with ONLY a JSON object in this exact format:
{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}

Vote must be either "accept" or "reject".

**JSON FORMAT REQUIREMENTS:**
- Return exactly one valid JSON object matching the requested schema.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Use valid JSON syntax: double quotes for all object keys and string values; never use single quotes.
- Do not wrap the object in markdown code fences.
- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.
- Do not put labels, item names, explanations, parenthetical notes, or placeholders inside arrays.
- Arrays must contain only the requested primitive values, for example numbers, strings, booleans, or objects required by the schema.
- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.
- Keep every string value on one line; use spaces instead of newline characters inside strings.
- Keep free-text fields concise and avoid embedded quotation marks or backslashes unless they are validly escaped JSON.
```

### 3.6 Reflection Prompt

Source: `CoFundingGame.get_reflection_prompt()`

```
Reflect on the outcome of Round 2.

**REJECTED ROUND PROPOSAL (counterfactual outcome if it had passed):**
  Market Street Protected Bike Lane: aggregate=28.00, cost=28 (would have been funded if accepted)
  Parkside Adventure Playground: aggregate=24.00, cost=35 (would have remained unfunded; short by 11.00)
  Oak Avenue Crosswalk Beacons: aggregate=0.00, cost=22 (would have remained unfunded; short by 22.00)
  Cedar Pool Access Lift: aggregate=0.00, cost=41 (would have remained unfunded; short by 41.00)
  Harborview Bus Shelter Canopies: aggregate=16.00, cost=20 (would have remained unfunded; short by 4.00)

**Projects that would have been funded if this rejected proposal had passed:** ['Market Street Protected Bike Lane']
**Vote outcome this round:** not accepted by two-thirds supermajority
**Counterfactual utility if this rejected proposal had been accepted:** 21.60
**Counterfactual raw utility before discount:** 24.00
**Discount factor this round:** 0.90
**Important:** Because the proposal was NOT accepted by a two-thirds supermajority, no money was committed and no project was funded this round. The next round starts from scratch with the same fixed budgets.

Consider what adjustments to your contributions might improve the outcome.
- Are there projects close to being funded that deserve more support?
- Are you over-contributing to already-funded projects?
- Should you shift focus to different projects?
```

---

## Reasoning Token Budget Addendum

Many prompt methods accept an optional `reasoning_token_budget`.
When that argument is provided, the current code appends one of these two suffix styles:

Generic inline style (discussion / proposal / voting / most reflection prompts):

```
**REASONING DEPTH:** Please use approximately 2000 tokens in your internal reasoning before outputting your response for this stage.
```

Thinking-prompt style:

```
**REASONING DEPTH:**
Please use approximately 2000 tokens in your internal reasoning before outputting your response for this stage.
```

---

## Summary Table

| Game | Setup source | Round phases | Reflection source | Notes |
| ---- | ------------ | ------------ | ----------------- | ----- |
| Game 1: Item Allocation | `item_allocation.py` | `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection` | `base.py` default | Private information: item values. |
| Game 2: Diplomatic Treaty | `diplomatic_treaty.py` | `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection` | `base.py` default | Percent displays are integer percentages throughout the prompt-facing interface. |
| Game 3: Co-Funding | `co_funding.py` | `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection` | `co_funding.py` custom | Private information: budget and project valuations. |
