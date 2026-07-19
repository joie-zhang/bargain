#!/usr/bin/env python3
"""Build strategy/persuasion qualitative tags for all multi-agent rollouts.

This is a second-pass codebook focused on strategic pressure, persuasion,
coalition formation, exploitation, compromise, and formalization tactics. It
uses the canonical 2,730-rollout table from the prior qualitative pass as the
manifest, then re-opens every raw result JSON and assigns the new tags from the
conversation logs plus a few saved structural fields.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUAL_DIR = PROJECT_ROOT / "analysis/qualitative_rollout_dynamics_20260628"
QUAL_CSV = QUAL_DIR / "refined_rollout_dynamics_coding.csv"
EXISTING_CODEBOOK_CSV = QUAL_DIR / "refined_dynamics_codebook.csv"
OUT_DIR = PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628"


@dataclass(frozen=True)
class StrategicTag:
    code: str
    title: str
    category: str
    description: str
    paper_value: str
    patterns: tuple[str, ...] = ()
    games: tuple[str, ...] = ()
    min_agents: int | None = None
    structural: bool = False


TAGS: list[StrategicTag] = [
    StrategicTag(
        "ultimatum_language",
        "Ultimatum language",
        "pressure",
        "The speaker frames acceptance as the only viable option, a final offer, or a take-it-or-leave-it demand.",
        "Measures coercive bargaining rather than merely firm preferences.",
        (r"\btake it or leave it\b", r"\bfinal offer\b", r"\blast offer\b", r"\bonly (?:acceptable|viable|workable) (?:deal|path|option)\b", r"\bmust accept\b"),
    ),
    StrategicTag(
        "conditional_veto_threat",
        "Conditional veto threat",
        "pressure",
        "The speaker ties acceptance or rejection to a concrete package condition.",
        "A sharper subtype of redline behavior that directly threatens the formal vote.",
        (r"\bI (?:will|would|must) (?:vote no|reject|block|oppose)\b", r"\bI'?ll (?:vote no|reject|block|oppose)\b", r"\bwill reject any\b", r"\bvote NO on any\b", r"\bcannot support\b", r"\bcan'?t support\b", r"\bwill not support\b", r"\bwon'?t support\b", r"\bif and only if\b"),
    ),
    StrategicTag(
        "fairness_accusation_pressure",
        "Fairness accusation pressure",
        "pressure",
        "The speaker pressures others by calling a proposal unfair, unreasonable, greedy, imbalanced, or one-sided.",
        "Captures norm enforcement and reputational pressure in mixed-motive bargaining.",
        (r"\bunfair\b", r"\bunreasonable\b", r"\bone[- ]sided\b", r"\btoo greedy\b", r"\bgreedy\b", r"\bskewed\b", r"\bimbalanced\b", r"\bnot balanced\b"),
    ),
    StrategicTag(
        "frustration_disappointment_display",
        "Frustration or disappointment display",
        "pressure",
        "The speaker explicitly displays frustration, disappointment, irritation, or exasperation.",
        "Directly targets the user's interest in anger/emotional tactics.",
        (r"\bfrustrat", r"\bdisappoint", r"\birritat", r"\bexasperat", r"\bnot productive\b", r"\bgoing in circles\b"),
    ),
    StrategicTag(
        "round_saving_urgency_pressure",
        "Round-saving urgency pressure",
        "pressure",
        "The speaker pressures closure by invoking wasted rounds, discounting/decay, final rounds, or immediate lock-in.",
        "Separates genuine compromise from deadline-driven pressure.",
        (r"\brunning out of (?:time|rounds)\b", r"\bfinal round\b", r"\blast round\b", r"\bneed to (?:close|lock|settle|confirm) (?:this )?now\b", r"\bwe need to move\b", r"\bbefore time runs out\b", r"\bdelaying risks\b", r"\bmove straight to the proposal\b", r"\block this in\b"),
    ),
    StrategicTag(
        "trust_repair_accounting",
        "Trust-repair accounting",
        "pressure",
        "The speaker uses trust, good faith, betrayal, or explicit repair language to stabilize a deal after tension or failed rounds.",
        "Useful for distinguishing strategic commitment problems from simple preference conflict.",
        (r"\brebuild trust\b", r"\bgood faith\b", r"\bburned me\b", r"\bbetray", r"\bfinal test\b", r"\btrust[- ]building\b", r"\bno deviations\b", r"\bshow trust\b", r"\brestore trust\b"),
    ),
    StrategicTag(
        "lock_in_confirmation_pressure",
        "Lock-in confirmation pressure",
        "pressure",
        "The speaker repeatedly asks others to confirm, lock in, or choose a named path, using procedural pressure more than new concessions.",
        "Tracks whether agents understand that cheap talk must become credible action.",
        (r"\bplease confirm\b", r"\bconfirm (?:your|that|which)\b", r"\block in\b", r"\block this in\b", r"\bchoose (?:Option|Path|Track)\b", r"\bready to submit\b", r"\bneed (?:a|an|your) (?:clear |explicit |firm )?commit"),
    ),
    StrategicTag(
        "adversarial_callout",
        "Adversarial callout",
        "pressure",
        "The speaker directly calls out another agent's behavior as evasive, nonconceding, noisy, or strategically unhelpful.",
        "Identifies confrontational persuasion that may correlate with model capability or competition.",
        (r"\bcut through the noise\b", r"\byou (?:haven'?t|didn'?t|do not|don'?t)\b.{0,80}\bconced", r"\bnot actually\b.{0,60}\bconced", r"\bthe real conflict\b", r"\bstop\b.{0,50}\b(?:dancing|avoiding|circling|pretending)", r"\bevasive\b", r"\bbroken promises\b", r"\bexploitative deals\b"),
    ),
    StrategicTag(
        "mandate_constituency_appeal",
        "Mandate or constituency appeal",
        "emotional persuasion",
        "The speaker justifies firmness by invoking a delegation, mandate, domestic constraint, stakeholder group, or ratification feasibility.",
        "Captures role-legitimacy pressure distinct from simple redlines.",
        (r"\bmandate\b", r"\bdelegation\b", r"\bstakeholder", r"\bconstituenc", r"\bdomestic\b", r"\bratification\b", r"\bpolitical feasibility\b", r"\bindustry\b", r"\bimplementation capacity\b"),
    ),
    StrategicTag(
        "rapport_before_pressure",
        "Rapport before pressure",
        "emotional persuasion",
        "The speaker uses warmth, appreciation, or praise immediately before disagreement, firmness, or a strategic ask.",
        "Measures soft persuasion and face-saving language that can mask pressure.",
        (r"\b(?:appreciate|thanks? for|great point|thoughtful|clear framing|constructive tone)\b.{0,220}\b(?:but|however|must|cannot|can'?t|need|I propose|my ask)\b",),
    ),
    StrategicTag(
        "utility_arithmetic_receipts",
        "Utility arithmetic receipts",
        "logical persuasion",
        "The speaker uses explicit arithmetic over values, utilities, costs, or totals to justify the deal.",
        "Directly tests whether stronger models persuade through better game-relevant reasoning.",
        (r"\b\d+(?:\.\d+)?\s*(?:points?|utility|value|units?|budget)\b", r"\btotal (?:utility|value|payoff|cost)\b", r"\badds? up to\b", r"\bsum\b", r"\bnet (?:gain|benefit|value)\b"),
    ),
    StrategicTag(
        "agent_specific_payoff_accounting",
        "Agent-specific payoff accounting",
        "logical persuasion",
        "The speaker calculates or narrates what named agents get under a proposal.",
        "Captures strategic perspective-taking in payoff space, not just generic fairness talk.",
        (r"\bAgent[_ ]?\d+\s+(?:gets|receives|would get|ends up with|values)\b", r"\bfor Agent[_ ]?\d+\b.{0,120}\b(?:value|utility|points?|benefit)\b", r"\byou get\b.{0,80}\bI get\b", r"\bI get\b.{0,80}\byou get\b"),
    ),
    StrategicTag(
        "threshold_gap_calculation",
        "Threshold-gap calculation",
        "logical persuasion",
        "The speaker identifies exactly how far a vote/project/deal is from passing.",
        "A more mechanical version of near-threshold rescue that can be plotted as strategic numeracy.",
        (r"\bshort by\b", r"\bneeds? only\b", r"\bjust \d+(?:\.\d+)? (?:more|units?|votes?)\b", r"\bone more (?:vote|acceptor|unit)\b", r"\bgap of\b", r"\bclosest to (?:funding|passing)\b"),
    ),
    StrategicTag(
        "overfunding_waste_policing",
        "Overfunding/waste policing",
        "logical persuasion",
        "The speaker calls out waste from overfunding, partial funding, already-funded projects, or resources split too thinly.",
        "Useful because it predicts repair of inefficient or scattered proposals.",
        (r"\boverfund", r"\balready funded\b", r"\bavoid (?:waste|wasting)\b", r"\bwasteful\b", r"\bwasted\b", r"\bsplit too thin\b", r"\btoo thinly\b", r"\bpartial funding yields zero\b", r"\bleave value on the table\b"),
    ),
    StrategicTag(
        "normative_risk_escalation",
        "Normative risk escalation",
        "logical persuasion",
        "The speaker pressures agreement through high-stakes moral, safety, security, health, climate, or catastrophic-risk language.",
        "Distinguishes value-threat persuasion from generic efficiency/fairness language.",
        (r"\bexistential\b", r"\bsystemic risks?\b", r"\bunacceptable .*risk", r"\bcatastrophic\b", r"\bpublic health (?:crisis|emergency|risk|threat)\b", r"\bethical (?:floor|standards?|imperative)\b", r"\bglobal stability\b", r"\bhigh ethical floor\b"),
    ),
    StrategicTag(
        "fairness_ledger_argument",
        "Fairness ledger argument",
        "logical persuasion",
        "The speaker itemizes balance, proportionality, equal burden, or equal gains in a ledger-like way.",
        "Distinguishes concrete fairness accounting from generic fairness rhetoric.",
        (r"\bbalanc(?:e|ed|ing)\b.{0,90}\b(?:because|by|with|against)\b", r"\bproportional\b", r"\bequal (?:share|split|burden|contribution)\b", r"\broughly equal\b", r"\beven out\b", r"\boffset\b"),
    ),
    StrategicTag(
        "empathy_then_pivot",
        "Empathy-then-pivot",
        "emotional persuasion",
        "The speaker acknowledges another agent's position, then pivots to their own ask or constraint.",
        "Captures a subtle persuasion style: validation before strategic pressure.",
        (r"\bI (?:understand|recognize|appreciate|hear) (?:your|that)\b.{0,180}\b(?:but|however|given|still|need)\b", r"\bI see why\b.{0,180}\b(?:but|however|still)\b"),
    ),
    StrategicTag(
        "technical_authority_claim",
        "Technical authority claim",
        "logical persuasion",
        "The speaker cites analysis, evidence, benchmarks, data, or arithmetic as decisive bargaining authority.",
        "Useful for measuring technocratic persuasion and pseudo-empirical bargaining.",
        (r"\bour analysis\b", r"\bthe data\b", r"\bevidence\b", r"\bbenchmarks?\b", r"\barithmetic\b", r"\bcalculation\b", r"\bmodel(?:ing)? shows\b", r"\btechnically\b"),
    ),
    StrategicTag(
        "consistency_prior_statement_appeal",
        "Consistency/prior-statement appeal",
        "logical persuasion",
        "The speaker invokes what another agent previously said, promised, valued, or agreed to.",
        "Measures whether agents use conversation history as strategic leverage.",
        (r"\bas you (?:said|noted|mentioned|stated)\b", r"\byou (?:said|noted|mentioned|stated)\b", r"\bearlier you\b", r"\bpreviously\b", r"\byour own (?:priority|valuation|proposal)\b", r"\byou already\b"),
    ),
    StrategicTag(
        "formal_artifact_frameworking",
        "Formal artifact frameworking",
        "logical persuasion",
        "The speaker turns persuasion into named baselines, tables, option sets, scorecards, or structured frameworks.",
        "Captures formatting artifacts becoming coordination devices.",
        (r"\bofficial baseline\b", r"\bscorecard\b", r"\boption [ABC]\b", r"\bTrack [ABC]\b", r"\|[^\\n]*\|", r"\btable of\b", r"\bconcrete (?:package|framework|baseline)\b"),
    ),
    StrategicTag(
        "conditional_quid_pro_quo",
        "Conditional quid pro quo",
        "trade/compromise",
        "The speaker offers an explicit if-you-do-X-I-will-do-Y exchange.",
        "This is the clearest horse-trading tag.",
        (r"\bif you\b.{0,120}\bI (?:will|can|would|am willing to)\b", r"\bin return for\b", r"\bin exchange for\b", r"\bprovided that\b", r"\bcontingent on\b"),
    ),
    StrategicTag(
        "conditional_support_ledger",
        "Conditional support ledger",
        "trade/compromise",
        "Agents maintain an explicit I-back-yours-if-you-back-mine support ledger.",
        "Measures alliance-like exchange rather than abstract logrolling.",
        (r"\bsupport (?:your|you getting|your claim|your proposal)\b.{0,180}\bsupport (?:my|me getting|my claim|my proposal)\b", r"\bI'?ll back\b.{0,160}\bif you back\b", r"\byou back\b.{0,160}\bI'?ll back\b", r"\bmutual support\b", r"\bin exchange for .* support\b"),
    ),
    StrategicTag(
        "concession_laddering",
        "Concession laddering",
        "trade/compromise",
        "The speaker marks a movement from an initial position toward a less demanding one.",
        "Useful for seeing whether stronger models actively manage bargaining gradients.",
        (r"\bI can move\b", r"\bwilling to move\b", r"\bmove from\b", r"\breduce (?:my|the)\b", r"\bincrease (?:my|the)\b", r"\bcome down\b", r"\bI can concede\b", r"\bconcede\b"),
    ),
    StrategicTag(
        "compensation_for_concession",
        "Compensation for concession",
        "trade/compromise",
        "The speaker asks for or offers compensation to offset giving up a valued item, issue, or contribution.",
        "Captures economically coherent compromise, not just politeness.",
        (r"\bcompensat", r"\bmake up for\b", r"\boffset\b", r"\bin return\b.{0,80}\bgiv(?:e|ing) up\b", r"\bif I give up\b", r"\bfor losing\b"),
    ),
    StrategicTag(
        "low_weight_concession_leverage",
        "Low-weight concession leverage",
        "trade/compromise",
        "The speaker explicitly turns a low-priority issue/item/project into a chip for gains on core priorities.",
        "A concrete compromise mechanism often hidden by broad package tags.",
        (r"\blow[- ]priority\b.{0,160}\b(?:concede|trade|chip|flexible|leverage)\b", r"\bnot a battleground\b", r"\blow[- ]value\b.{0,160}\b(?:concede|trade|chip|support)\b", r"\bsmall concession\b", r"\bsweetener\b"),
    ),
    StrategicTag(
        "fallback_with_reservation",
        "Fallback with reservation",
        "trade/compromise",
        "The speaker names a backup path while preserving their preferred outcome or caveat.",
        "Separates flexible compromise from unconditional agreement.",
        (r"\bfallback\b", r"\bbackup\b", r"\balternative\b", r"\bif .* fails\b", r"\bif .* can'?t pass\b", r"\bsecond[- ]best\b", r"\bas a reserve\b"),
    ),
    StrategicTag(
        "vote_history_diagnostics",
        "Vote-history diagnostics",
        "trade/compromise",
        "Agents use prior proposal vote counts or failed-round evidence to infer which package can pass.",
        "Separates adaptive learning from generic coalition talk.",
        (r"\b(?:Round \d+|previous round|last round|vote results?|voting patterns?)\b.{0,220}\b(?:votes?|supermajority|failed|support|foundation|convergence|diagnostic|illuminating)\b", r"\bproposal #?\d+.{0,120}\b(?:votes?|supporters?)\b"),
    ),
    StrategicTag(
        "top_bottom_disclosure_protocol",
        "Top/bottom disclosure protocol",
        "trade/compromise",
        "Agents propose a protocol where everyone names top priorities plus low/zero-value concessions.",
        "Turns bargaining into explicit information elicitation and should scale with N.",
        (r"\beveryone\b.{0,120}\b(?:top|highest).{0,120}\b(?:low|zero|don'?t care|least)\b", r"\btop \d+(?:-\d+)?\b.{0,160}\b(?:low|zero|bottom)\b", r"\bname\b.{0,120}\b(?:top|priority).{0,120}\b(?:low|zero|bottom)\b"),
    ),
    StrategicTag(
        "concession_accounting_ledger",
        "Concession-accounting ledger",
        "trade/compromise",
        "Agents itemize how much they moved, what they sacrificed, and what reciprocal payment is owed.",
        "Tracks reciprocity accounting, not just the existence of a tradeoff.",
        (r"\bconcession from\b", r"\bI moved from\b", r"\bI have moved\b", r"\bsignificant concession\b", r"\babsorbs? loss\b", r"\bnot capitulation\b", r"\breciprocal compromise\b", r"\bwhat I'?m giving up\b"),
    ),
    StrategicTag(
        "vector_lock_formalization",
        "Vector-lock formalization",
        "trade/compromise",
        "Agents convert a verbal deal into exact contribution vectors, proposal numbers, or vote instructions.",
        "Positive counterpart to verbal/formal drift.",
        (r"\bsubmit\s+\[", r"\bcontribution vector\b", r"\bexactly\s+\[", r"\bvote (?:accept|yes)\b", r"\bBOTH VOTE ACCEPT\b", r"\bproposal #?\d+\b.{0,80}\baccept\b", r"\bno further discussion\b"),
    ),
    StrategicTag(
        "named_microcoalition_slate",
        "Named microcoalition slate",
        "coalition",
        "A speaker proposes a named subset of agents and an exact slate/path, often bypassing the full group.",
        "Directly captures coalition formation in N>=4 settings.",
        (r"\bAgent[_ ]?\d+\s*,\s*Agent[_ ]?\d+(?:\s*,?\s*(?:and )?Agent[_ ]?\d+)?\b.{0,160}\b(?:each contribute|please confirm|coalition|slate|package)\b", r"\b(?:coalition|alliance|bloc|slate).{0,120}\bAgent[_ ]?\d+.{0,80}\bAgent[_ ]?\d+", r"\bteam up with Agent[_ ]?\d+\b"),
        min_agents=4,
    ),
    StrategicTag(
        "vote_bloc_counting",
        "Vote-bloc counting",
        "coalition",
        "Agents count yes votes, bloc size, or the supermajority math needed to pass.",
        "Plot-able against N and model strength as coalition arithmetic.",
        (r"\b\d+\s*(?:yes|accept|support) votes\b", r"\bneed \d+ (?:votes|supporters|acceptors)\b", r"\bwe have \d+\b.{0,80}\bvotes\b", r"\btwo[- ]thirds (?:supermajority|vote|votes|acceptance|threshold)\b", r"\b(?:supermajority|threshold)\b.{0,80}\b(?:votes|supporters|acceptors)\b"),
        min_agents=4,
    ),
    StrategicTag(
        "bespoke_agent_or_bloc_recruitment",
        "Bespoke agent/bloc recruitment",
        "coalition",
        "The speaker targets named agents or blocs with customized concessions, questions, or asks.",
        "Captures personalization beyond broad coalition talk.",
        (r"\bdirect question to Agent", r"\bto (?:the )?[A-Z][A-Za-z -]+ coalition\b", r"\bAgent[_ ]?\d+\b.{0,180}\b(?:what specific adjustment would secure|what would secure|I can offer|for your support|your vote)\b", r"\bswing votes? needed\b"),
        min_agents=4,
    ),
    StrategicTag(
        "holdout_bypass_minimum_coalition",
        "Holdout-bypass minimum coalition",
        "coalition",
        "The speaker proposes passing a deal without a refusing or low-value holdout.",
        "Captures exploitative or hard-nosed minimum-winning coalition behavior.",
        (r"\b(?:pass|proceed|move forward) without (?:Agent[_ ]?\d+|unanimity|the holdout|holdouts)\b", r"\bdon'?t need unanimous\b", r"\bnot need unanimity\b", r"\bif .* won'?t support\b.{0,120}\b(?:pass|move|proceed)\b"),
        min_agents=4,
    ),
    StrategicTag(
        "coalition_integrity_warning",
        "Coalition-integrity warning",
        "coalition",
        "The speaker warns that reopening an anchor or moving too far will fracture the viable coalition.",
        "Captures path-dependence and pressure to preserve an emergent deal.",
        (r"\bfractur(?:e|ing) (?:the )?coalition\b", r"\bbreak (?:our|the) coalition\b", r"\bdestabili[sz]e\b", r"\brisks? losing support\b", r"\bonly viable coalition\b", r"\bcoalition momentum\b"),
        min_agents=4,
    ),
    StrategicTag(
        "silent_agent_inclusion_guard",
        "Silent-agent inclusion guard",
        "coalition",
        "Agents resist premature lock-in by explicitly calling for quiet or not-yet-speaking agents.",
        "Shows anti-capture governance in larger groups and should be strongly N-dependent.",
        (r"\bAgent[_ ]?\d+(?:[-, and_0-9 ]+)? haven'?t spoken\b", r"\b(?:hear from|include|wait for).{0,80}\b(?:quiet|silent|not-yet-speaking|remaining) agents?\b", r"\bquiet agents?\b", r"\bnot-yet-speaking agents?\b"),
        min_agents=4,
    ),
    StrategicTag(
        "third_party_mediation",
        "Third-party mediation",
        "coalition",
        "A speaker summarizes others' positions and proposes a bridge deal between conflicting agents.",
        "Useful for detecting mediator/synthesizer roles that may rise with capability.",
        (r"\bsummariz(?:e|ing) (?:the|everyone|what)\b", r"\bsynthesize\b", r"\bbridge\b.{0,80}\b(?:gap|positions|between)\b", r"\bmediate\b", r"\bcommon ground between\b"),
        min_agents=3,
    ),
    StrategicTag(
        "cross_agent_conflict_mapping",
        "Cross-agent conflict mapping",
        "coalition",
        "The speaker explicitly maps which agents have overlapping or conflicting claims.",
        "A concrete sign of strategic situational awareness in larger games.",
        (r"\bAgent[_ ]?\d+\b.{0,120}\bAgent[_ ]?\d+\b.{0,120}\b(?:both|overlap|conflict|contested|competing)\b", r"\bconflict between Agent[_ ]?\d+ and Agent[_ ]?\d+\b", r"\boverlap between\b"),
        min_agents=3,
    ),
    StrategicTag(
        "nonoverlap_lane_setting",
        "Non-overlap lane setting",
        "coalition",
        "Agents assign lanes or non-aggression zones so different players can claim different items/issues/projects.",
        "Captures coordination architecture in larger groups.",
        (r"\bclear lanes\b", r"\bnon[- ]overlapping\b", r"\bnon[- ]aggression\b", r"\byou take .* I take\b", r"\bI'?ll stay out of\b", r"\bdefer to .* on\b", r"\bavoid competing\b"),
        min_agents=3,
    ),
    StrategicTag(
        "budget_carryover_hallucination",
        "Budget carryover hallucination",
        "formalization",
        "Agents talk as if failed-round spending, rejected proposals, or unfunded partial contributions carry forward.",
        "A formal reasoning failure that can explain bad Game 3 outcomes.",
        (r"\balready (?:funded|in the books)\b", r"\bfrom Round \d+\b.{0,180}\b(?:carr(?:y|ies|ied)|already|still funded|in place)\b", r"\bcarry(?:over| forward)\b.{0,80}\b(?:contributions?|budget|funding|spending|pledges?)\b", r"\bprevious contributions? remain\b"),
        games=("game3",),
    ),
    StrategicTag(
        "self_advocacy_value_maximization",
        "Self-advocacy/value maximization",
        "self-interest/exploitation",
        "The speaker explicitly argues for maximizing their own payoff or preserving their own best outcome.",
        "Central to whether agents are strategically self-interested rather than merely cooperative.",
        (r"\bmaximize my\b", r"\bmy (?:utility|payoff|value|score)\b", r"\bbest for me\b", r"\bfrom my perspective\b", r"\bI need to preserve\b", r"\bI want to keep\b"),
    ),
    StrategicTag(
        "valuation_as_budget_error",
        "Valuation-as-budget error",
        "self-interest/exploitation",
        "Agents confuse private valuations with spendable budget, project cost, or contribution capacity.",
        "Identifies arithmetic failures that can masquerade as strategy.",
        (r"\bvaluations?\b.{0,120}\buse up\b.{0,80}\b(?:budget|combined budget)\b", r"\buse up\b.{0,120}\b(?:out of|of) (?:our|the) \d+ budget\b", r"\bmy value\b.{0,120}\b(?:spendable|budget|contribution capacity)\b", r"\bworth \d+\b.{0,120}\b(?:so|therefore).{0,80}\b(?:contribute|budget|spend)\b"),
    ),
    StrategicTag(
        "partial_progress_signal_spend",
        "Partial-progress signal spend",
        "self-interest/exploitation",
        "Agents deliberately put money into under-threshold projects as progress, signal, or momentum despite all-or-nothing funding.",
        "A Game 3 misconception that can be plotted against failures and model strength.",
        (r"\bbring us closer\b", r"\bmake progress\b", r"\bsignal(?:ing)? support\b", r"\bincrement\b", r"\bpartial contribution\b", r"\btoward eventual funding\b"),
        games=("game3",),
    ),
    StrategicTag(
        "leverage_preservation",
        "Leverage preservation",
        "self-interest/exploitation",
        "The speaker refuses to give up a valuable asset/position because it preserves bargaining leverage.",
        "Directly game-theoretic: agents reason about strategic leverage, not only final value.",
        (r"\bleverage\b", r"\bpreserve .* leverage\b", r"\bnot give up\b.{0,120}\b(?:top|valuable|priority)\b", r"\bhold onto\b.{0,100}\b(?:leverage|top|priority)\b"),
    ),
    StrategicTag(
        "zero_value_reciprocity_offer",
        "Zero-value reciprocity offer",
        "self-interest/exploitation",
        "An agent offers support for a zero/low-value target as a bargaining chip for reciprocal support.",
        "Distinguishes compromise from efficient preference overlap.",
        (r"\b(?:despite|even though).{0,80}\bzero value to me\b.{0,180}\b(?:support|contribute|back|cooperation|reciprocal)\b", r"\bzero value to me\b.{0,180}\b(?:in exchange|secure your cooperation|reciprocal support)\b", r"\bno value to me\b.{0,180}\b(?:in exchange|secure your cooperation|reciprocal support)\b"),
        games=("game3",),
    ),
    StrategicTag(
        "zero_value_subsidy",
        "Zero-value subsidy",
        "self-interest/exploitation",
        "A Game 3 agent pays positive cost for a funded project they privately value at zero.",
        "A structural exploitation/capitulation tag, auditable without relying on rhetoric.",
        (),
        games=("game3",),
        structural=True,
    ),
    StrategicTag(
        "silent_free_beneficiary",
        "Silent free beneficiary",
        "self-interest/exploitation",
        "A Game 3 funded project gives positive private value to an agent who contributes nothing to it.",
        "Detects realized free-riding even when the transcript never names it.",
        (),
        games=("game3",),
        structural=True,
    ),
    StrategicTag(
        "accepted_loss_capitulation",
        "Accepted-loss capitulation",
        "self-interest/exploitation",
        "The rollout reaches consensus even though at least one agent ends with negative final utility.",
        "A structural marker of strategic failure or capitulation under pressure.",
        (),
        structural=True,
    ),
    StrategicTag(
        "overfunded_full_budget_dump",
        "Overfunded full-budget dump",
        "formalization",
        "Accepted Game 3 proposals overfund a project because agents dump full budgets into the focal project.",
        "Auditable waste/exploitation marker from saved formal contribution vectors.",
        (),
        games=("game3",),
        structural=True,
    ),
    StrategicTag(
        "counter_anchor_cost_policing",
        "Counter-anchor cost policing",
        "formalization",
        "A speaker corrects an impossible or confused anchor by re-centering on actual costs, budgets, or constraints.",
        "Measures useful strategic correction rather than persuasion alone.",
        (r"\bkey constraint\b", r"\bproject costs? \(?not valuations?\)?", r"\bactual costs?\b", r"\bwithin (?:our|the) budget\b", r"\bcosts?, not valuations?\b", r"\bwe cannot fund\b.{0,120}\bcost"),
    ),
]


assert len(TAGS) == 50
TAG_BY_CODE = {tag.code: tag for tag in TAGS}
PATTERNS = {
    tag.code: [re.compile(pattern, re.IGNORECASE) for pattern in tag.patterns]
    for tag in TAGS
}
SPACE_RE = re.compile(r"\s+")
SEARCH_CHAR_LIMIT = 180_000


def clean(text: object, limit: int | None = None) -> str:
    value = SPACE_RE.sub(" ", str(text or "")).strip()
    if limit is not None and len(value) > limit:
        return value[: limit - 1].rstrip() + "..."
    return value


def link(path: str) -> str:
    return f"[json]({path})"


def load_existing_codes() -> set[str]:
    if not EXISTING_CODEBOOK_CSV.exists():
        return set()
    with EXISTING_CODEBOOK_CSV.open(newline="") as handle:
        return {row["dynamic_code"] for row in csv.DictReader(handle)}


def read_manifest() -> list[dict[str, str]]:
    with QUAL_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def discussion_and_proposal_text(logs: list[dict]) -> str:
    pieces: list[str] = []
    for entry in logs:
        if entry.get("from") == "system":
            continue
        if entry.get("phase") not in {"discussion", "proposal"}:
            continue
        pieces.append(clean(entry.get("content")))
        proposal = entry.get("proposal")
        if isinstance(proposal, dict):
            pieces.append(clean(proposal.get("reasoning")))
            pieces.append(clean(proposal.get("raw_response")))
    return "\n".join(piece for piece in pieces if piece)


def first_round_text(logs: list[dict]) -> str:
    pieces = [
        clean(entry.get("content"))
        for entry in logs
        if entry.get("from") != "system" and entry.get("phase") == "discussion" and entry.get("round") == 1
    ]
    return "\n".join(piece for piece in pieces if piece)


def opening_text(logs: list[dict], max_messages: int = 2) -> str:
    pieces = []
    for entry in logs:
        if entry.get("from") == "system" or entry.get("phase") != "discussion":
            continue
        pieces.append(clean(entry.get("content")))
        if len(pieces) >= max_messages:
            break
    return "\n".join(pieces)


def split_evidence_units(text: str) -> list[str]:
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text):
        piece = clean(piece, 420)
        if piece:
            out.append(piece)
    return out


def find_quote(text: str, code: str) -> str:
    regexes = PATTERNS.get(code, [])
    if not regexes:
        return ""
    search_text = text[:SEARCH_CHAR_LIMIT]
    for regex in regexes:
        match = regex.search(search_text)
        if match:
            start = max(0, match.start() - 140)
            end = min(len(search_text), match.end() + 220)
            return clean(search_text[start:end], 420)
    return ""


def proposal_validation_error(data: dict) -> bool:
    for entry in data.get("conversation_logs") or []:
        proposal = entry.get("proposal")
        if not isinstance(proposal, dict):
            continue
        if proposal.get("validation_error") or proposal.get("recovered_after_error"):
            return True
        raw = clean(proposal.get("raw_response"))
        if re.search(r"\bvalidation error\b|\binvalid proposal\b|\bmalformed\b", raw, re.IGNORECASE):
            return True
    return False


def divergent_final_proposals(logs: list[dict], final_round: int | None) -> bool:
    rows = [entry for entry in logs if entry.get("phase") == "proposal_enumeration"]
    if final_round is not None:
        rows = [entry for entry in rows if entry.get("round") == final_round] or rows
    if not rows:
        return False
    enumerated = rows[-1].get("enumerated_proposals") or []
    proposals = []
    for proposal in enumerated:
        payload = proposal.get("allocation") or proposal.get("contributions") or proposal.get("original_proposal")
        if payload is not None:
            proposals.append(json.dumps(payload, sort_keys=True))
    return len(set(proposals)) > 1


def final_game3_proposal(data: dict, final_round: int | None) -> dict:
    cfg = data.get("config") or {}
    if cfg.get("game_label") != "game3":
        return {}
    logs = data.get("conversation_logs") or []
    rows = [entry for entry in logs if entry.get("phase") == "proposal_enumeration"]
    if final_round is not None:
        rows = [entry for entry in rows if entry.get("round") == final_round] or rows
    if not rows:
        return {}
    enumerated = rows[-1].get("enumerated_proposals") or []
    if not enumerated:
        return {}
    proposal = enumerated[0]
    original = proposal.get("original_proposal") if isinstance(proposal.get("original_proposal"), dict) else {}
    return {
        "contributions_by_agent": proposal.get("contributions_by_agent") or original.get("contributions_by_agent") or {},
        "aggregate_totals": proposal.get("aggregate_totals") or original.get("aggregate_totals") or [],
        "funded_projects": proposal.get("funded_projects") or original.get("funded_projects") or [],
    }


def funded_indexes(raw_funded: object) -> set[int]:
    indexes: set[int] = set()
    if not isinstance(raw_funded, list):
        return indexes
    for item in raw_funded:
        if isinstance(item, int):
            indexes.add(item)
        elif isinstance(item, dict) and isinstance(item.get("index"), int):
            indexes.add(item["index"])
    return indexes


def project_costs(data: dict) -> list[float]:
    items = (data.get("config") or {}).get("items") or []
    costs = []
    for item in items:
        try:
            costs.append(float(item.get("cost")))
        except (AttributeError, TypeError, ValueError):
            costs.append(math.nan)
    return costs


def agent_budget(cfg: dict, agent: str) -> float:
    budgets = cfg.get("agent_budgets") or {}
    try:
        return float(budgets.get(agent))
    except (AttributeError, TypeError, ValueError):
        try:
            return float(cfg.get("total_budget")) / float(cfg.get("n_agents"))
        except (TypeError, ValueError, ZeroDivisionError):
            return math.nan


def game3_zero_value_subsidy(data: dict, final_round: int | None) -> bool:
    proposal = final_game3_proposal(data, final_round)
    funded = funded_indexes(proposal.get("funded_projects"))
    prefs = data.get("agent_preferences") or {}
    contributions = proposal.get("contributions_by_agent") or {}
    for agent, vector in contributions.items():
        values = prefs.get(agent) or []
        for idx in funded:
            try:
                if float(vector[idx]) > 1e-9 and float(values[idx]) <= 1e-9:
                    return True
            except (IndexError, TypeError, ValueError):
                continue
    return False


def game3_silent_free_beneficiary(data: dict, final_round: int | None) -> bool:
    proposal = final_game3_proposal(data, final_round)
    funded = funded_indexes(proposal.get("funded_projects"))
    prefs = data.get("agent_preferences") or {}
    contributions = proposal.get("contributions_by_agent") or {}
    for agent, values in prefs.items():
        vector = contributions.get(agent) or []
        for idx in funded:
            try:
                if float(values[idx]) > 1e-9 and float(vector[idx]) <= 1e-9:
                    return True
            except (IndexError, TypeError, ValueError):
                continue
    return False


def game3_overfunded_full_budget_dump(data: dict, final_round: int | None) -> bool:
    proposal = final_game3_proposal(data, final_round)
    funded = funded_indexes(proposal.get("funded_projects"))
    if not funded:
        return False
    costs = project_costs(data)
    aggregate = proposal.get("aggregate_totals") or []
    overfunded = False
    for idx in funded:
        try:
            overfunded = overfunded or float(aggregate[idx]) > float(costs[idx]) + 1e-9
        except (IndexError, TypeError, ValueError):
            continue
    if not overfunded:
        return False
    cfg = data.get("config") or {}
    for agent, vector in (proposal.get("contributions_by_agent") or {}).items():
        budget = agent_budget(cfg, agent)
        if not math.isfinite(budget):
            continue
        try:
            total = sum(float(x) for x in vector)
            max_funded = max(float(vector[idx]) for idx in funded)
        except (IndexError, TypeError, ValueError):
            continue
        if total > 1e-9 and max_funded >= budget - 1e-6 and total - max_funded <= 1e-6:
            return True
    return False


def accepted_negative_utility(data: dict) -> bool:
    if not bool(data.get("consensus_reached")):
        return False
    utilities = data.get("final_utilities") or {}
    for value in utilities.values():
        try:
            if float(value) < -1e-9:
                return True
        except (TypeError, ValueError):
            continue
    return False


def structural_quote(code: str) -> str:
    return {
        "zero_value_subsidy": "Saved final proposal: an agent contributed positive budget to a funded project with private value 0.",
        "silent_free_beneficiary": "Saved final proposal: a funded project gave positive private value to an agent who contributed 0 to that project.",
        "accepted_loss_capitulation": "Saved final utilities include at least one negative payoff in an accepted consensus outcome.",
        "overfunded_full_budget_dump": "Saved final proposal overfunded a funded project and at least one agent put their full budget into that project.",
    }.get(code, "")


def tag_scope_text(tag: StrategicTag, scopes: dict[str, str]) -> str:
    if tag.code in {"top_bottom_disclosure_protocol", "self_advocacy_value_maximization"}:
        return scopes["first_round"] or scopes["full"]
    return scopes["full"]


def tag_applies(tag: StrategicTag, cfg: dict, data: dict, text: str, final_round: int | None) -> bool:
    if tag.games and cfg.get("game_label") not in tag.games:
        return False
    if tag.min_agents is not None:
        try:
            if int(cfg.get("n_agents") or 0) < tag.min_agents:
                return False
        except ValueError:
            return False
    if tag.code == "zero_value_subsidy":
        return game3_zero_value_subsidy(data, final_round)
    if tag.code == "silent_free_beneficiary":
        return game3_silent_free_beneficiary(data, final_round)
    if tag.code == "accepted_loss_capitulation":
        return accepted_negative_utility(data)
    if tag.code == "overfunded_full_budget_dump":
        return bool(data.get("consensus_reached")) and game3_overfunded_full_budget_dump(data, final_round)
    scope = text[:SEARCH_CHAR_LIMIT]
    return any(regex.search(scope) for regex in PATTERNS.get(tag.code, []))


def classify_row(row: dict[str, str]) -> dict:
    path = Path(row["result_path"])
    with path.open() as handle:
        data = json.load(handle)
    cfg = data.get("config") or {}
    logs = data.get("conversation_logs") or []
    final_round = data.get("final_round")
    try:
        final_round_int = int(final_round)
    except (TypeError, ValueError):
        final_round_int = None

    scopes = {
        "full": discussion_and_proposal_text(logs),
        "first_round": first_round_text(logs),
        "opening": opening_text(logs, max_messages=4),
    }
    codes: list[str] = []
    quotes: dict[str, str] = {}
    for tag in TAGS:
        text = tag_scope_text(tag, scopes)
        if tag_applies(tag, cfg, data, text, final_round_int):
            codes.append(tag.code)
            quote = find_quote(text, tag.code)
            if not quote and tag.structural:
                quote = structural_quote(tag.code)
            if quote:
                quotes[tag.code] = quote

    return {
        "result_path": str(path),
        "config_id": row.get("config_id") or cfg.get("config_id"),
        "experiment_family": row.get("experiment_family") or cfg.get("experiment_family"),
        "game_label": row.get("game_label") or cfg.get("game_label"),
        "n_agents": row.get("n_agents") or cfg.get("n_agents"),
        "setting": row.get("setting") or "",
        "model_order": row.get("model_order") or cfg.get("model_order"),
        "models": row.get("models") or "+".join(cfg.get("models") or []),
        "adversary_model": row.get("adversary_model") or cfg.get("adversary_model"),
        "adversary_position": row.get("adversary_position") or cfg.get("adversary_position"),
        "consensus_reached": row.get("consensus_reached") or data.get("consensus_reached"),
        "final_round": row.get("final_round") or final_round,
        "divergent_final_proposals": divergent_final_proposals(logs, final_round_int),
        "new_strategy_tags": ";".join(codes),
        "new_strategy_tag_count": len(codes),
        "new_strategy_tag_quotes_json": json.dumps(quotes, sort_keys=True),
    }


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frequency_tables(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    overall = []
    by_family_game = []
    for tag in TAGS:
        count = sum(tag.code in str(row["new_strategy_tags"]).split(";") for row in rows)
        overall.append(
            {
                "tag_code": tag.code,
                "tag_title": tag.title,
                "category": tag.category,
                "count": count,
                "denominator": len(rows),
                "share": count / len(rows) if rows else math.nan,
            }
        )
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment_family"]), str(row["game_label"]))].append(row)
    for (family, game), subset in sorted(groups.items()):
        for tag in TAGS:
            count = sum(tag.code in str(row["new_strategy_tags"]).split(";") for row in subset)
            by_family_game.append(
                {
                    "experiment_family": family,
                    "game_label": game,
                    "tag_code": tag.code,
                    "tag_title": tag.title,
                    "category": tag.category,
                    "count": count,
                    "denominator": len(subset),
                    "share": count / len(subset) if subset else math.nan,
                }
            )
    return overall, by_family_game


def long_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        quotes = json.loads(row.get("new_strategy_tag_quotes_json") or "{}")
        for code in [code for code in str(row["new_strategy_tags"]).split(";") if code]:
            tag = TAG_BY_CODE[code]
            out.append(
                {
                    "result_path": row["result_path"],
                    "config_id": row["config_id"],
                    "experiment_family": row["experiment_family"],
                    "game_label": row["game_label"],
                    "n_agents": row["n_agents"],
                    "tag_code": code,
                    "tag_title": tag.title,
                    "category": tag.category,
                    "quote": quotes.get(code, ""),
                }
            )
    return out


def evidence_rows(rows: list[dict], max_examples: int = 5) -> list[dict]:
    out = []
    for tag in TAGS:
        examples = []
        seen_groups = set()
        tagged = [row for row in rows if tag.code in str(row["new_strategy_tags"]).split(";")]
        for row in tagged:
            group = (row["experiment_family"], row["game_label"])
            if group in seen_groups and len(examples) < 3:
                continue
            quotes = json.loads(row.get("new_strategy_tag_quotes_json") or "{}")
            quote = quotes.get(tag.code, "")
            if not quote:
                continue
            examples.append(
                {
                    "tag_code": tag.code,
                    "tag_title": tag.title,
                    "category": tag.category,
                    "config_id": row["config_id"],
                    "experiment_family": row["experiment_family"],
                    "game_label": row["game_label"],
                    "n_agents": row["n_agents"],
                    "result_path": row["result_path"],
                    "quote": quote,
                }
            )
            seen_groups.add(group)
            if len(examples) >= max_examples:
                break
        out.extend(examples)
    return out


def write_codebook(evidence: list[dict], frequencies: list[dict]) -> None:
    evidence_by_code: dict[str, list[dict]] = defaultdict(list)
    for row in evidence:
        evidence_by_code[row["tag_code"]].append(row)
    freq_by_code = {row["tag_code"]: row for row in frequencies}

    fields = [
        "tag_code",
        "tag_title",
        "category",
        "description",
        "paper_value",
        "games",
        "min_agents",
        "structural",
        "patterns",
        "count",
        "share",
    ]
    write_csv(
        OUT_DIR / "new_strategy_tag_codebook.csv",
        [
            {
                "tag_code": tag.code,
                "tag_title": tag.title,
                "category": tag.category,
                "description": tag.description,
                "paper_value": tag.paper_value,
                "games": ";".join(tag.games),
                "min_agents": tag.min_agents or "",
                "structural": tag.structural,
                "patterns": "; ".join(tag.patterns),
                "count": freq_by_code[tag.code]["count"],
                "share": freq_by_code[tag.code]["share"],
            }
            for tag in TAGS
        ],
        fields,
    )

    lines = [
        "# New Strategy/Persuasion Tag Codebook",
        "",
        "These 50 tags are intentionally distinct from the earlier refined dynamics codebook. They focus on pressure, persuasion, coalition formation, exploitative/self-interested tactics, compromise mechanics, and formalization reliability.",
        "",
    ]
    for tag in TAGS:
        freq = freq_by_code[tag.code]
        lines.extend(
            [
                f"## {tag.title} (`{tag.code}`)",
                "",
                f"- Category: {tag.category}",
                f"- Definition: {tag.description}",
                f"- Why it matters: {tag.paper_value}",
                f"- Frequency: {freq['count']} / {freq['denominator']} ({freq['share']:.1%})",
                "- Examples:",
            ]
        )
        examples = evidence_by_code.get(tag.code, [])
        if not examples:
            lines.append("  - No examples matched under the current reproducible classifier.")
        for ex in examples:
            quote = clean(ex["quote"], 300).replace("|", "\\|")
            lines.append(
                f"  - `{ex['config_id']}` {ex['experiment_family']} {ex['game_label']} n={ex['n_agents']}: {quote} {link(ex['result_path'])}"
            )
        lines.append("")
    (OUT_DIR / "new_strategy_tag_codebook.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(rows: list[dict], frequencies: list[dict], by_family_game: list[dict]) -> None:
    top = sorted(frequencies, key=lambda row: (-int(row["count"]), row["tag_code"]))[:15]
    rare = sorted(frequencies, key=lambda row: (int(row["count"]), row["tag_code"]))[:10]
    category_counts = Counter()
    for row in rows:
        for code in [code for code in str(row["new_strategy_tags"]).split(";") if code]:
            category_counts[TAG_BY_CODE[code].category] += 1

    lines = [
        "# Strategic Qualitative Tags: Second-Pass Coding",
        "",
        f"- Rollouts re-read from raw JSON: **{len(rows)}**.",
        f"- New tags: **{len(TAGS)}**.",
        "- Unit of analysis: rollout-level tag presence. A tag means the behavior appears at least once in discussion/proposal text or saved structural fields.",
        "- Important caveat: this is a reproducible high-recall coding scaffold, not a final human adjudication. The codebook and evidence snippets are designed for audit/refinement before paper claims.",
        "- Validation caveat: explicit/structural tags are strongest. Broad lexical tags such as lock-in pressure, generic concession language, fairness ledgering, vote-history diagnostics, and support-ledger language should be treated as candidate labels until manually adjudicated or tightened for a specific paper figure.",
        "- Structural Game 3 caveat: `zero_value_subsidy` and `silent_free_beneficiary` are computed on the final submitted proposal, including no-consensus final-round proposals; use `consensus_reached == True` when analyzing realized outcomes only.",
        "",
        "## Highest-Frequency Tags",
        "",
    ]
    for row in top:
        lines.append(f"- `{row['tag_code']}`: {row['count']} / {row['denominator']} ({row['share']:.1%})")
    lines.extend(["", "## Rare/High-Specificity Tags", ""])
    for row in rare:
        lines.append(f"- `{row['tag_code']}`: {row['count']} / {row['denominator']} ({row['share']:.1%})")
    lines.extend(["", "## Category Totals", ""])
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count} rollout-tag assignments")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `new_strategy_tag_codebook.md` and `.csv`: definitions, rationale, patterns, frequencies, and example transcript links.",
            "- `strategic_tag_assignments.csv`: one row per rollout with semicolon-separated new tags.",
            "- `strategic_tag_long.csv`: one row per rollout-tag assignment, easier for plotting/modeling.",
            "- `new_strategy_tag_frequencies.csv`: overall frequency table.",
            "- `new_strategy_tag_frequencies_by_family_game.csv`: family x game frequency table.",
            "- `new_strategy_tag_evidence.csv`: supporting transcript snippets for each tag.",
        ]
    )
    (OUT_DIR / "strategic_qualitative_tagging_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_codes()
    overlap = sorted(existing & {tag.code for tag in TAGS})
    if overlap:
        raise SystemExit(f"new tag code(s) duplicate existing codebook: {overlap}")

    manifest = read_manifest()
    rows = []
    for idx, row in enumerate(manifest, start=1):
        rows.append(classify_row(row))
        if idx % 250 == 0:
            print(f"classified={idx}/{len(manifest)}", flush=True)
    frequencies, by_family_game = frequency_tables(rows)
    long = long_rows(rows)
    evidence = evidence_rows(rows)

    assignment_fields = [
        "result_path",
        "config_id",
        "experiment_family",
        "game_label",
        "n_agents",
        "setting",
        "model_order",
        "models",
        "adversary_model",
        "adversary_position",
        "consensus_reached",
        "final_round",
        "divergent_final_proposals",
        "new_strategy_tags",
        "new_strategy_tag_count",
        "new_strategy_tag_quotes_json",
    ]
    write_csv(OUT_DIR / "strategic_tag_assignments.csv", rows, assignment_fields)
    write_csv(
        OUT_DIR / "strategic_tag_long.csv",
        long,
        ["result_path", "config_id", "experiment_family", "game_label", "n_agents", "tag_code", "tag_title", "category", "quote"],
    )
    write_csv(
        OUT_DIR / "new_strategy_tag_frequencies.csv",
        sorted(frequencies, key=lambda row: (-int(row["count"]), row["tag_code"])),
        ["tag_code", "tag_title", "category", "count", "denominator", "share"],
    )
    write_csv(
        OUT_DIR / "new_strategy_tag_frequencies_by_family_game.csv",
        by_family_game,
        ["experiment_family", "game_label", "tag_code", "tag_title", "category", "count", "denominator", "share"],
    )
    write_csv(
        OUT_DIR / "new_strategy_tag_evidence.csv",
        evidence,
        ["tag_code", "tag_title", "category", "config_id", "experiment_family", "game_label", "n_agents", "result_path", "quote"],
    )
    write_codebook(evidence, frequencies)
    write_report(rows, frequencies, by_family_game)

    zero_example = [row["tag_code"] for row in frequencies if int(row["count"]) == 0]
    print(
        f"rollouts={len(rows)} tags={len(TAGS)} assignments={len(long)} "
        f"zero_frequency_tags={len(zero_example)}"
    )
    if zero_example:
        print("zero_frequency_tag_codes=" + ",".join(zero_example))


if __name__ == "__main__":
    main()
