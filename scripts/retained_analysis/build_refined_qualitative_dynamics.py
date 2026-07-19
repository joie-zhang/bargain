#!/usr/bin/env python3
"""Build a refined qualitative dynamics coding table.

This complements ``build_qualitative_rollout_dynamics_report.py``. The first
script intentionally uses a broad, reproducible scaffold. This one adds a
paper-facing tag set distilled from the sub-agent deep reads, using only agent
discussion/proposal text for textual tags and saved proposal/vote objects for
structural tags.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/qualitative_rollout_dynamics_20260628"

PRODUCTION_ROOT = (
    PROJECT_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255"
)
HETEROGENEOUS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)
ROOTS = [PRODUCTION_ROOT, HETEROGENEOUS_ROOT]


@dataclass(frozen=True)
class RefinedDynamic:
    code: str
    title: str
    description: str
    patterns: tuple[str, ...] = ()
    games: tuple[str, ...] = ()
    families: tuple[str, ...] = ()


DYNAMICS = [
    RefinedDynamic(
        "priority_disclosure",
        "Priority disclosure",
        "Agents reveal ranked priorities, numeric values, target bands, or top projects.",
        (r"\btop priorit", r"\bmy priorit", r"\bI value\b", r"\bvaluation", r"\bworth .* to me", r"\bhighest[- ]value", r"\btarget band"),
    ),
    RefinedDynamic(
        "anchor_package",
        "Anchor package",
        "The discussion organizes around a named item, policy basket, project core, or focal package.",
        (r"\banchor\b", r"\bcore\b", r"\bpackage\b", r"\bbundle\b", r"\bprimary target\b", r"\bfocal\b", r"\bspine\b"),
    ),
    RefinedDynamic(
        "explicit_tradeoff_package",
        "Explicit tradeoff package",
        "Agents propose swaps, concessions, bundles, issue tradeoffs, or cross-dimensional compensation.",
        (r"\btrade[- ]?off", r"\bswap\b", r"\bconcession", r"\bin exchange\b", r"\bif you .* I", r"\bI can give\b", r"\bmovement on\b"),
    ),
    RefinedDynamic(
        "side_payment_smoothing",
        "Side-payment smoothing",
        "Agents add low-value items, small concessions, or smoothing adjustments to make a deal acceptable.",
        (r"\bsweetener", r"\bsmoothing", r"\bbonus item", r"\blow[- ]value", r"\bzero[- ]value", r"\bbargaining chip", r"\bminor adjustment"),
    ),
    RefinedDynamic(
        "sequenced_or_contingent_deal",
        "Sequenced or contingent deal",
        "The bargain relies on staged movement, milestones, future rounds, audits, reviews, or conditional promises.",
        (r"\btwo[- ]step", r"\bstaged", r"\bcontingent", r"\bmilestone", r"\baudit", r"\bverification", r"\breview", r"\bfuture round", r"\bnext round", r"\bRound 2"),
    ),
    RefinedDynamic(
        "coalition_threshold_management",
        "Coalition and threshold management",
        "Agents reason about votes, acceptor counts, supermajorities, swing agents, commitments, or project thresholds.",
        (r"\bcoalition", r"\bsupermajority", r"\bthreshold", r"\bvotes?\b", r"\bacceptors?\b", r"\bswing", r"\bcommit", r"\bpublicly back"),
    ),
    RefinedDynamic(
        "hard_anchor_or_redline",
        "Hard anchor or redline",
        "An agent marks a claim as non-negotiable, a non-starter, not acceptable, or impossible to support.",
        (r"\bnon[- ]negotiable", r"\bnon[- ]starter", r"\bred ?line", r"\bcannot support", r"\bcan't support", r"\bnot acceptable", r"\bwill not support", r"\bveto"),
    ),
    RefinedDynamic(
        "fairness_frame",
        "Fairness frame",
        "The proposal is justified with fairness, balance, proportionality, equity, or mutual acceptability.",
        (r"\bfair", r"\bequitable", r"\bbalanced", r"\bproportional", r"\bequal", r"\bmutually acceptable", r"\bcompromise"),
    ),
    RefinedDynamic(
        "efficiency_frame",
        "Efficiency frame",
        "Agents invoke efficiency, total value, optimality, avoiding waste, or win-win outcomes.",
        (r"\befficient", r"\bmaximize", r"\boptimal", r"\btotal value", r"\bavoid waste", r"\bwin[- ]win", r"\bPareto"),
    ),
    RefinedDynamic(
        "single_item_mutual_support_pact",
        "Single-item mutual-support pact",
        "Game 1 agents agree to support each person's most-valued item when top claims do not conflict.",
        (r"\bmost valued item", r"\btop item", r"\bI support you getting", r"\bhelp you obtain", r"\bmutual support", r"\beach.*top"),
        games=("game1",),
    ),
    RefinedDynamic(
        "shared_top_item_deadlock",
        "Shared-top-item deadlock",
        "Game 1 agents identify a contested top item or shared high-value claim that slows agreement.",
        (r"\bboth want", r"\bsame .*item", r"\bcontested", r"\bcompeting for", r"\boverlap.*top", r"\bshared top"),
        games=("game1",),
    ),
    RefinedDynamic(
        "low_value_trading_pool",
        "Low-value trading pool",
        "Game 1 agents use unwanted or zero-value objects as bargaining chips or clearing goods.",
        (r"\blow[- ]value", r"\bzero[- ]value", r"\bbargaining chip", r"\bitems I don't need", r"\bno value to me", r"\bindifferent"),
        games=("game1",),
    ),
    RefinedDynamic(
        "policy_basket_logrolling",
        "Policy-basket logrolling",
        "Game 2 agents trade movement across issues, target bands, guardrails, and package-level concessions.",
        (r"\bissue", r"\bpackage", r"\btrade[- ]?off", r"\bconcession", r"\bguardrail", r"\btarget band", r"\bmovement on", r"\bmiddle ground"),
        games=("game2",),
    ),
    RefinedDynamic(
        "staged_verification_governance",
        "Staged verification governance",
        "Game 2 bargains shift from percentages to implementation mechanisms such as audits, milestones, reviews, or verification.",
        (r"\bstaged", r"\bmilestone", r"\baudit", r"\bverification", r"\breview mechanism", r"\bimplementation", r"\bgovernance", r"\bjoint verification"),
        games=("game2",),
    ),
    RefinedDynamic(
        "spine_focal_point",
        "Spine focal point",
        "Game 2 discussion names a security, climate, or other issue spine that organizes the package.",
        (r"\bspine\b", r"\bfocal", r"\bcentral package", r"\bsecurity .* climate", r"\bclimate .* security"),
        games=("game2",),
    ),
    RefinedDynamic(
        "midpoint_closure",
        "Midpoint closure",
        "Game 2 settlement is framed as splitting the difference, a midpoint, or final middle-ground closure.",
        (r"\bmidpoint", r"\bsplit the difference", r"\bmiddle ground", r"\bfinal compromise", r"\bmeet halfway"),
        games=("game2",),
    ),
    RefinedDynamic(
        "feasibility_first_budget_math",
        "Feasibility-first budget math",
        "Game 3 agents focus on costs, budgets, minimum top-ups, and whether a project can actually clear threshold.",
        (r"\bbudget", r"\bcost", r"\bfund", r"\bthreshold", r"\bneeds? \d", r"\bunits?", r"\baggregate", r"\btop[- ]up"),
        games=("game3",),
    ),
    RefinedDynamic(
        "numeric_pledge_split",
        "Numeric pledge split",
        "Game 3 agents state concrete contribution amounts or per-agent splits.",
        (r"\bI will contribute", r"\bI can contribute", r"\bcommit \d", r"\b\d+(?:\.\d+)? units?", r"\bsplit\b", r"\bcontribution vector"),
        games=("game3",),
    ),
    RefinedDynamic(
        "single_project_rally",
        "Single-project rally",
        "Game 3 agents rally around one flagship, primary, or anchor project as the easiest win.",
        (r"\bsingle project", r"\bone project", r"\bprimary target", r"\banchor project", r"\bflagship", r"\bcore project", r"\block in .*project"),
        games=("game3",),
    ),
    RefinedDynamic(
        "near_threshold_rescue",
        "Near-threshold rescue",
        "Game 3 later-round bargaining pivots to a project needing only a small gap-filling contribution.",
        (r"\btop[- ]up", r"\bgap", r"\bneeds? only", r"\bjust \d", r"\bone more", r"\bshort by", r"\bclosest to funding", r"\bnear[- ]threshold"),
        games=("game3",),
    ),
    RefinedDynamic(
        "zero_value_holdout",
        "Zero-value holdout",
        "Game 3 refusal is justified by a project giving the agent zero or too little private value.",
        (r"\bzero utility", r"\bzero value", r"\bno value", r"\bdoes not benefit me", r"\bnot worth", r"\bconsume.*budget"),
        games=("game3",),
    ),
    RefinedDynamic(
        "self_fund_fallback",
        "Self-funding fallback",
        "Game 3 agents fall back to funding their own preferred project or a low-dependence coalition path.",
        (r"\bself[- ]fund", r"\bfund my", r"\bmy own project", r"\blower[- ]dependence", r"\bwithout relying"),
        games=("game3",),
    ),
    RefinedDynamic(
        "free_riding",
        "Free-riding discourse",
        "An agent raises free-riding as a strategic concern or openly frames zero contribution as free-riding.",
        (r"\bfree[- ]rid", r"\b0 units .*strategic", r"\bzero contribution", r"\bbenefit without"),
        games=("game3",),
    ),
    RefinedDynamic(
        "adversary_agenda_anchor",
        "Adversary agenda anchor",
        "In homogeneous-adversary runs, the inserted model supplies the early frame, focal proposal, or strategic anchor.",
        (r"\bI propose", r"\bopening", r"\banchor", r"\bpackage", r"\bmy priorities", r"\bI recommend"),
        families=("homogeneous_adversary",),
    ),
    RefinedDynamic(
        "adversary_last_synthesizer_or_veto",
        "Adversary last-position synthesizer or veto",
        "A last-position adversary summarizes, trims, ratifies, or vetoes an already-emerging baseline coalition.",
        (r"\bbuilding on", r"\bsummar", r"\bI can support", r"\bI cannot support", r"\bfinal package", r"\bveto"),
        families=("homogeneous_adversary",),
    ),
    RefinedDynamic(
        "baseline_mirroring_or_deference",
        "Baseline mirroring or deference",
        "GPT-5-nano baselines echo, ratify, lightly modify, or copy a proposed frame rather than introducing an independent frame.",
        (r"\bI agree", r"\bI support", r"\balign", r"\bbuilding on", r"\bthat works", r"\bI can back", r"\bconfirm"),
        families=("homogeneous_adversary",),
    ),
    RefinedDynamic(
        "template_role_artifact",
        "Template or role artifact",
        "Messages contain draft-like instructions, wrong self-labels, impossible round references, or role/scaffold leakage.",
        (r"\breply you can use", r"\bRound 11", r"\bpost-round", r"\bas an AI", r"\bStrategy Response", r"\bwrong agent", r"\bAgent_1 \| Round"),
    ),
]

PATTERNS = {
    dynamic.code: [re.compile(pattern, flags=re.IGNORECASE) for pattern in dynamic.patterns]
    for dynamic in DYNAMICS
}
COMBINED_PATTERNS = {
    dynamic.code: re.compile("|".join(f"(?:{pattern})" for pattern in dynamic.patterns), flags=re.IGNORECASE)
    for dynamic in DYNAMICS
    if dynamic.patterns
}
CODE_TO_DYNAMIC = {dynamic.code: dynamic for dynamic in DYNAMICS}


def clean(text: object, limit: int = 5000) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def split_sentences(text: str) -> list[str]:
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text):
        value = clean(piece, 320)
        if value:
            out.append(value)
    return out


def agent_text(logs: list[dict]) -> str:
    pieces = []
    for entry in logs:
        if entry.get("from") == "system":
            continue
        if entry.get("phase") not in {"discussion", "proposal"}:
            continue
        pieces.append(clean(entry.get("content", "")))
        proposal = entry.get("proposal")
        if isinstance(proposal, dict):
            pieces.append(clean(proposal.get("reasoning", "")))
    return "\n".join(pieces)


def adversary_text(logs: list[dict], adversary_agent: str | None) -> str:
    if not adversary_agent:
        return ""
    return "\n".join(clean(e.get("content", "")) for e in logs if e.get("from") == adversary_agent)


def baseline_text(logs: list[dict], adversary_agent: str | None) -> str:
    return "\n".join(
        clean(e.get("content", ""))
        for e in logs
        if e.get("from") not in {"system", adversary_agent} and e.get("phase") in {"discussion", "proposal"}
    )


def find_quote(text: str, code: str) -> str:
    regexes = PATTERNS.get(code, [])
    # Full transcripts can be very long. A short first matching quote is only
    # illustrative, so cap the search window to keep full-corpus coding fast.
    for sentence in split_sentences(text[:80_000]):
        if any(regex.search(sentence) for regex in regexes):
            return sentence
    return ""


def utility_range(final_utilities: object) -> float:
    if not isinstance(final_utilities, dict):
        return math.nan
    values = []
    for value in final_utilities.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return max(values) - min(values) if values else math.nan


def utility_mean(final_utilities: object) -> float:
    if not isinstance(final_utilities, dict):
        return math.nan
    values = []
    for value in final_utilities.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return mean(values) if values else math.nan


def setting_label(cfg: dict) -> str:
    game = cfg.get("game_label")
    if game == "game1":
        return f"comp={cfg.get('competition_level', '')}"
    if game == "game2":
        return f"rho={cfg.get('rho', '')}, theta={cfg.get('theta', '')}"
    if game == "game3":
        return f"sigma={cfg.get('sigma', '')}, alpha={cfg.get('alpha', '')}"
    return ""


def final_vote_stats(logs: list[dict], final_round: int | None) -> dict[str, int | None]:
    rows = [e for e in logs if e.get("phase") == "vote_tabulation"]
    if final_round is not None:
        rows = [e for e in rows if e.get("round") == final_round] or rows
    if not rows:
        return {"max_accept": None, "min_reject": None, "threshold": None, "voters": None}
    content = str(rows[-1].get("content", ""))
    vote_pairs = [(int(a), int(r)) for a, r in re.findall(r"(\d+)\s+accept,\s+(\d+)\s+reject", content)]
    threshold_match = re.search(r"threshold:\s*(\d+)/(\d+)", content)
    threshold = int(threshold_match.group(1)) if threshold_match else None
    voters = int(threshold_match.group(2)) if threshold_match else None
    if not vote_pairs:
        return {"max_accept": None, "min_reject": None, "threshold": threshold, "voters": voters}
    return {
        "max_accept": max(a for a, _ in vote_pairs),
        "min_reject": min(r for _, r in vote_pairs),
        "threshold": threshold,
        "voters": voters,
    }


def final_game3_scatter(logs: list[dict], cfg: dict, final_round: int | None) -> dict[str, object]:
    if cfg.get("game_label") != "game3":
        return {"proposal_scatter": False, "positive_nonfunded_count": 0, "funded_count": None}
    rows = [e for e in logs if e.get("phase") == "proposal_enumeration"]
    if final_round is not None:
        rows = [e for e in rows if e.get("round") == final_round] or rows
    if not rows:
        return {"proposal_scatter": False, "positive_nonfunded_count": 0, "funded_count": None}
    enumerated = rows[-1].get("enumerated_proposals") or []
    if not enumerated:
        return {"proposal_scatter": False, "positive_nonfunded_count": 0, "funded_count": None}
    proposal = enumerated[0]
    aggregate = proposal.get("aggregate_totals") or proposal.get("original_proposal", {}).get("aggregate_totals") or []
    funded = proposal.get("funded_projects") or proposal.get("original_proposal", {}).get("funded_projects") or []
    funded_indexes = set()
    for item in funded:
        if isinstance(item, int):
            funded_indexes.add(item)
        elif isinstance(item, dict) and isinstance(item.get("index"), int):
            funded_indexes.add(item["index"])
    positive_nonfunded = 0
    for idx, value in enumerate(aggregate):
        try:
            is_positive = float(value) > 0
        except (TypeError, ValueError):
            is_positive = False
        if is_positive and idx not in funded_indexes:
            positive_nonfunded += 1
    return {
        "proposal_scatter": positive_nonfunded >= 2,
        "positive_nonfunded_count": positive_nonfunded,
        "funded_count": len(funded),
    }


def adversary_agent(cfg: dict) -> str | None:
    if cfg.get("experiment_family") != "homogeneous_adversary":
        return None
    agents = cfg.get("agents") or []
    if cfg.get("adversary_position") == "first" and agents:
        return str(agents[0])
    if cfg.get("adversary_position") == "last" and agents:
        return str(agents[-1])
    return None


def tag_textual(dynamic: RefinedDynamic, cfg: dict, text: str) -> bool:
    if dynamic.games and cfg.get("game_label") not in dynamic.games:
        return False
    if dynamic.families and cfg.get("experiment_family") not in dynamic.families:
        return False
    if dynamic.code == "adversary_last_synthesizer_or_veto" and cfg.get("adversary_position") != "last":
        return False
    regex = COMBINED_PATTERNS.get(dynamic.code)
    if regex is None:
        return False
    # These dynamics almost always appear in the opening agenda-setting and
    # repair discussion. Capping protects against very long scaffolded logs.
    return bool(regex.search(text[:200_000]))


def classify(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    cfg = data.get("config") or {}
    logs = data.get("conversation_logs") or []
    text = agent_text(logs)
    adv_agent = adversary_agent(cfg)
    adv_text = adversary_text(logs, adv_agent)
    base_text = baseline_text(logs, adv_agent)
    final_round = data.get("final_round")
    final_round_int = int(final_round) if str(final_round).isdigit() else None
    consensus = bool(data.get("consensus_reached"))
    vote_stats = final_vote_stats(logs, final_round_int)
    scatter = final_game3_scatter(logs, cfg, final_round_int)

    tags: set[str] = set()
    for dynamic in DYNAMICS:
        scope_text = text
        if dynamic.code.startswith("adversary_"):
            scope_text = adv_text
        elif dynamic.code == "baseline_mirroring_or_deference":
            scope_text = base_text
        if tag_textual(dynamic, cfg, scope_text):
            tags.add(dynamic.code)

    if consensus and final_round_int == 1:
        tags.add("outcome_consensus_r1")
    elif consensus and final_round_int is not None and 2 <= final_round_int <= 3:
        tags.add("outcome_consensus_r2_r3")
    elif consensus and final_round_int is not None and 4 <= final_round_int <= 9:
        tags.add("outcome_late_consensus_r4_r9")
    elif not consensus:
        tags.add("outcome_no_consensus_r10")

    max_accept = vote_stats.get("max_accept")
    threshold = vote_stats.get("threshold")
    if consensus and max_accept is not None and threshold is not None and max_accept == threshold:
        tags.add("minimum_winning_supermajority")
    if not consensus and max_accept is not None and threshold is not None and max_accept == threshold - 1:
        tags.add("near_miss_vote")

    u_range = utility_range(data.get("final_utilities"))
    u_mean = utility_mean(data.get("final_utilities"))
    if math.isfinite(u_range) and (not math.isfinite(u_mean) or u_range >= max(20.0, abs(u_mean) * 0.75)):
        tags.add("high_inequality_outcome")

    if not consensus and re.search(r"\balign|\bagree|\bconsensus|\bconfirm|\block in|\bshared\b|\bcore\b|\banchor\b", text, re.IGNORECASE):
        tags.add("verbal_convergence_vote_failure")

    if cfg.get("game_label") == "game3":
        if scatter["proposal_scatter"]:
            tags.add("proposal_scatter")
            tags.add("accepted_with_scatter" if consensus else "failed_with_scatter")
        if not consensus and re.search(r"\banchor|\bcore|\bvector|\bRound 11|future round|post-round", text, re.IGNORECASE):
            tags.add("semantic_vector_or_ballot_drift")

    if cfg.get("game_label") == "game2" and {"hard_anchor_or_redline", "explicit_tradeoff_package"} <= tags:
        tags.add("redline_then_package")

    return {
        "result_path": str(path),
        "config_id": cfg.get("config_id"),
        "experiment_family": cfg.get("experiment_family"),
        "game_label": cfg.get("game_label"),
        "n_agents": cfg.get("n_agents"),
        "setting": setting_label(cfg),
        "model_order": cfg.get("model_order"),
        "models": "+".join(cfg.get("models") or []),
        "adversary_model": cfg.get("adversary_model"),
        "adversary_position": cfg.get("adversary_position"),
        "consensus_reached": consensus,
        "final_round": final_round,
        "max_accept_final_vote": max_accept,
        "threshold_final_vote": threshold,
        "positive_nonfunded_count": scatter["positive_nonfunded_count"],
        "funded_count": scatter["funded_count"],
        "utility_range": u_range,
        "refined_dynamic_codes": ";".join(sorted(tags)),
        "refined_dynamic_count": len(tags),
        "matched_quotes_json": "{}",
    }


def iter_paths() -> list[Path]:
    paths: list[Path] = []
    for root in ROOTS:
        paths.extend(sorted(root.glob("runs/*/experiment_results.json")))
    return paths


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frequency_rows(rows: list[dict], by: tuple[str, ...] = ()) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    if by:
        for row in rows:
            groups[tuple(str(row.get(key)) for key in by)].append(row)
    else:
        groups[()].extend(rows)

    out = []
    codes = sorted(
        {code for row in rows for code in str(row.get("refined_dynamic_codes", "")).split(";") if code}
    )
    for group_key, subset in sorted(groups.items()):
        denominator = len(subset)
        counts = Counter(code for row in subset for code in str(row.get("refined_dynamic_codes", "")).split(";") if code)
        for code in codes:
            count = counts[code]
            item = {
                "dynamic_code": code,
                "dynamic_title": CODE_TO_DYNAMIC.get(code, RefinedDynamic(code, code, "")).title,
                "count": count,
                "denominator": denominator,
                "share": count / denominator if denominator else 0,
            }
            for idx, key in enumerate(by):
                item[key] = group_key[idx]
            out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for path in iter_paths():
        try:
            row = classify(path)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": repr(exc)})
            continue
        if row.get("experiment_family") in {"heterogeneous_random", "homogeneous_adversary", "homogeneous_control"}:
            rows.append(row)

    coding_fields = [
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
        "max_accept_final_vote",
        "threshold_final_vote",
        "positive_nonfunded_count",
        "funded_count",
        "utility_range",
        "refined_dynamic_codes",
        "refined_dynamic_count",
        "matched_quotes_json",
    ]
    write_csv(OUT_DIR / "refined_rollout_dynamics_coding.csv", rows, coding_fields)

    frequency = frequency_rows(rows)
    write_csv(
        OUT_DIR / "refined_dynamic_frequencies.csv",
        sorted(frequency, key=lambda row: (-row["count"], row["dynamic_code"])),
        ["dynamic_code", "dynamic_title", "count", "denominator", "share"],
    )

    by_family_game = frequency_rows(rows, ("experiment_family", "game_label"))
    write_csv(
        OUT_DIR / "refined_dynamic_frequencies_by_family_game.csv",
        sorted(
            by_family_game,
            key=lambda row: (row["experiment_family"], row["game_label"], -row["count"], row["dynamic_code"]),
        ),
        ["experiment_family", "game_label", "dynamic_code", "dynamic_title", "count", "denominator", "share"],
    )

    codebook_rows = [
        {
            "dynamic_code": dynamic.code,
            "dynamic_title": dynamic.title,
            "description": dynamic.description,
            "games": ";".join(dynamic.games),
            "families": ";".join(dynamic.families),
            "patterns": "; ".join(dynamic.patterns),
        }
        for dynamic in DYNAMICS
    ]
    for code, title, description in [
        ("outcome_consensus_r1", "Round-1 consensus", "Accepted supermajority in the first round."),
        ("outcome_consensus_r2_r3", "Short repair consensus", "Consensus in rounds 2-3 after a brief repair cycle."),
        ("outcome_late_consensus_r4_r9", "Late consensus", "Consensus in rounds 4-9."),
        ("outcome_no_consensus_r10", "No consensus by round 10", "No accepted supermajority by the final round."),
        ("minimum_winning_supermajority", "Minimum-winning supermajority", "Final accepted vote passed exactly at the saved threshold."),
        ("near_miss_vote", "Near-miss vote", "Final failed vote was one acceptor short of the saved threshold."),
        ("high_inequality_outcome", "High-inequality outcome", "Final utility spread is large relative to mean utility."),
        ("verbal_convergence_vote_failure", "Verbal convergence with vote failure", "Agents describe alignment but final formal vote/proposal fails."),
        ("proposal_scatter", "Proposal scatter", "Game 3 final proposal has at least two positive but nonfunded project totals."),
        ("accepted_with_scatter", "Accepted with scatter", "Game 3 consensus succeeds despite scattered nonfunded contributions."),
        ("failed_with_scatter", "Failed with scatter", "Game 3 no-consensus final proposal contains scattered nonfunded contributions."),
        ("semantic_vector_or_ballot_drift", "Semantic vector or ballot drift", "Discussion names one plan but formal vectors/votes fail to realize it."),
        ("redline_then_package", "Redline then package", "Game 2 redline language is paired with package/tradeoff language."),
    ]:
        codebook_rows.append(
            {
                "dynamic_code": code,
                "dynamic_title": title,
                "description": description,
                "games": "",
                "families": "",
                "patterns": "",
            }
        )
    write_csv(
        OUT_DIR / "refined_dynamics_codebook.csv",
        sorted(codebook_rows, key=lambda row: row["dynamic_code"]),
        ["dynamic_code", "dynamic_title", "description", "games", "families", "patterns"],
    )
    write_csv(OUT_DIR / "refined_parse_errors.csv", errors, ["path", "error"])

    print(f"wrote refined dynamics for {len(rows)} rollouts; errors={len(errors)}")


if __name__ == "__main__":
    main()
