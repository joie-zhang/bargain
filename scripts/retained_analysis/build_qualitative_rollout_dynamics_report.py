#!/usr/bin/env python3
"""Build qualitative coding artifacts for the N>2 rollout corpus.

This script reads every raw rollout JSON in the canonical heterogeneous and
homogeneous multi-agent roots, extracts transcript-level signals from
conversation_logs, assigns a first-pass qualitative codebook, and writes a
markdown report plus per-rollout appendix.
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
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/qualitative_rollout_dynamics_20260628"
SUBAGENT_DIR = OUT_DIR / "subagent_reports"
CHUNK_DIR = OUT_DIR / "subagent_chunk_manifests"

PRODUCTION_ROOT = (
    PROJECT_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255"
)
HETEROGENEOUS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)

ROOTS = [
    ("production_homogeneous_and_control", PRODUCTION_ROOT),
    ("heterogeneous_repair", HETEROGENEOUS_ROOT),
]

GAME_NAMES = {
    "game1": "item allocation",
    "game2": "diplomacy/issues",
    "game3": "public-goods cofunding",
}


@dataclass(frozen=True)
class Dynamic:
    code: str
    title: str
    description: str
    patterns: tuple[str, ...]
    games: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    structural: bool = False


DYNAMICS: list[Dynamic] = [
    Dynamic(
        "preference_transparency",
        "Preference transparency and priority broadcasting",
        "Agents explicitly reveal top items/issues/projects, numeric values, or priority order to make bargaining legible.",
        (r"\btop priorit", r"\bpriority\b", r"\bpriorities\b", r"\bvalue to me\b", r"\bmy values?\b", r"\btransparent"),
    ),
    Dynamic(
        "overlap_conflict_detection",
        "Early detection of overlapping claims",
        "Agents notice that several parties want the same scarce item, issue outcome, or project and frame it as a conflict to resolve.",
        (r"\boverlap\b", r"\bconflict\b", r"\bwe both\b", r"\bsame .*priority", r"\bcompeting for\b", r"\bcontested\b"),
    ),
    Dynamic(
        "coalition_vote_counting",
        "Coalition formation and vote counting",
        "Agents talk in terms of assembling enough supporters, votes, majorities, or thresholds to pass a proposal.",
        (r"\bcoalition\b", r"\b\d+[- ]?vote\b", r"\bvotes?\b", r"\bmajority\b", r"\bsupermajority\b", r"\bthreshold\b", r"\bcommit"),
    ),
    Dynamic(
        "package_deal_bundling",
        "Package deals and bundle construction",
        "Agents bundle several items/issues/projects together, or make explicit trade-offs across dimensions.",
        (r"\bpackage\b", r"\bbundle\b", r"\btrade[- ]?off\b", r"\bswap\b", r"\bpairing\b", r"\bcombination\b", r"\bconcession"),
    ),
    Dynamic(
        "fallback_pivoting",
        "Fallback plans and rapid pivoting",
        "Agents maintain backup proposals and pivot when a preferred target cannot gather enough support.",
        (r"\bfallback\b", r"\bbackup\b", r"\bpivot\b", r"\balternative\b", r"\bif .*can't pass\b", r"\bif .*fails\b"),
    ),
    Dynamic(
        "fairness_norms",
        "Fairness and proportionality norms",
        "Agents justify proposals with fairness, balance, proportional sharing, equal treatment, or mutually acceptable compromise.",
        (r"\bfair\b", r"\bequitable\b", r"\bproportional\b", r"\bbalanced\b", r"\bequal\b", r"\bmutually acceptable\b", r"\bcompromise"),
    ),
    Dynamic(
        "efficiency_norms",
        "Efficiency and total-surplus framing",
        "Agents appeal to efficiency, avoiding waste, maximizing total value, or finding high-impact outcomes.",
        (r"\befficient\b", r"\befficiency\b", r"\bmaximize\b", r"\boptimal\b", r"\bhigh[- ]impact\b", r"\bwin[- ]win\b", r"\bpareto"),
    ),
    Dynamic(
        "low_value_trading_pool",
        "Low-value item trading pool",
        "Item-allocation agents offer zero/low-value goods as sweeteners or use unwanted items to clear disputes.",
        (r"\blow[- ]value\b", r"\bzero[- ]value\b", r"\bhappy to trade\b", r"\bsweetener", r"\bfree for the taking\b"),
        games=("game1",),
    ),
    Dynamic(
        "agenda_setting_first_mover",
        "First-mover agenda setting",
        "An early speaker names a concrete proposal that becomes the focal reference point for the rest of the discussion.",
        (r"\bi propose\b", r"\bmy proposal\b", r"\bopening stance\b", r"\bI'?ll start\b", r"\bI recommend\b"),
    ),
    Dynamic(
        "adversary_anchor",
        "Inserted-model anchoring in homogeneous-adversary runs",
        "The inserted adversary supplies a focal proposal, rationale, or framing that the GPT-5-nano agents respond to or orbit around.",
        (r"\bAgent_1\b", r"\bagree with Agent_", r"\bbuilding on\b", r"\bas .* suggested\b", r"\bI support\b"),
        families=("homogeneous_adversary",),
    ),
    Dynamic(
        "baseline_deference",
        "Baseline deference or proposal copying",
        "Baseline agents echo, accept, or lightly modify a proposal instead of introducing an independent alternative.",
        (r"\bagree\b", r"\bsupport\b", r"\bsounds good\b", r"\balign\b", r"\bI can back\b", r"\bI will support\b"),
        families=("homogeneous_adversary",),
    ),
    Dynamic(
        "distributional_tension",
        "Efficiency-distribution tension",
        "Discussion recognizes the difference between a good group outcome and who captures the surplus.",
        (r"\bmy own utility\b", r"\bmaximize my\b", r"\bfor me\b", r"\bfor everyone\b", r"\bgroup\b", r"\bsurplus\b", r"\bshare\b"),
    ),
    Dynamic(
        "holdout_refusal",
        "Holdout or refusal to support",
        "Agents explicitly decline to support a target because it has low value to them or does not fit their coalition interests.",
        (r"\bI don't\b", r"\bI do not\b", r"\bunlikely to back\b", r"\bcan't support\b", r"\bcannot support\b", r"\bnot willing\b", r"\boppose\b"),
    ),
    Dynamic(
        "public_good_budget_threshold",
        "Public-good budget and threshold coordination",
        "Game 3 agents reason about costs, budgets, pledges, and the minimum coalition needed to fund a project.",
        (r"\bbudget\b", r"\bcost\b", r"\bfund\b", r"\bpledge\b", r"\bcontribute\b", r"\bthreshold\b", r"\bproject\b"),
        games=("game3",),
    ),
    Dynamic(
        "public_good_single_project_rally",
        "Single-project rally",
        "Game 3 discussion converges on one flagship project as the easiest fundable win.",
        (r"\bflagship\b", r"\bsole item\b", r"\bsingle\b", r"\bone project\b", r"\bclear win\b", r"\bprimary target\b"),
        games=("game3",),
    ),
    Dynamic(
        "issue_logrolling",
        "Issue-by-issue logrolling",
        "Game 2 agents trade movement on one issue against gains on another.",
        (r"\bissue\b", r"\bconcede\b", r"\btrade\b", r"\bmovement\b", r"\bmiddle ground\b", r"\bpackage\b"),
        games=("game2",),
    ),
    Dynamic(
        "multi_round_revision",
        "Multi-round revision after failed or partial agreement",
        "The rollout goes beyond the first round and agents revise proposals after feedback, votes, or failed consensus.",
        (),
        structural=True,
    ),
    Dynamic(
        "fast_consensus",
        "Fast consensus",
        "The rollout reaches agreement in the first round.",
        (),
        structural=True,
    ),
    Dynamic(
        "no_consensus",
        "No consensus",
        "The rollout ends without consensus.",
        (),
        structural=True,
    ),
    Dynamic(
        "high_inequality_outcome",
        "High-inequality outcome",
        "The final utilities have a wide spread, suggesting one or a few agents captured much more value.",
        (),
        structural=True,
    ),
    Dynamic(
        "clean_vote_integrity",
        "Clean procedural completion",
        "The saved vote-integrity flags show no synthetic votes, contamination, or hard vote failure.",
        (),
        structural=True,
    ),
    Dynamic(
        "procedural_vote_problem",
        "Procedural vote anomaly",
        "The saved vote-integrity flags show synthetic voting, contamination, or hard failure.",
        (),
        structural=True,
    ),
]

PATTERNS = {
    dynamic.code: [re.compile(pattern, flags=re.IGNORECASE) for pattern in dynamic.patterns]
    for dynamic in DYNAMICS
}


def clean_one_line(text: object, limit: int = 260) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [clean_one_line(piece, 320) for piece in pieces if clean_one_line(piece, 320)]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def utility_stats(final_utilities: object) -> dict[str, float]:
    if not isinstance(final_utilities, dict):
        return {"utility_min": math.nan, "utility_max": math.nan, "utility_mean": math.nan, "utility_range": math.nan}
    values = []
    for value in final_utilities.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    if not values:
        return {"utility_min": math.nan, "utility_max": math.nan, "utility_mean": math.nan, "utility_range": math.nan}
    return {
        "utility_min": min(values),
        "utility_max": max(values),
        "utility_mean": mean(values),
        "utility_range": max(values) - min(values),
    }


def competition_value(cfg: dict) -> str:
    for key in ("competition_level", "rho", "theta", "sigma", "alpha"):
        value = cfg.get(key)
        if value is not None and value != "":
            try:
                return f"{key}={float(value):.3g}"
            except (TypeError, ValueError):
                return f"{key}={value}"
    return ""


def setting_label(cfg: dict) -> str:
    game = cfg.get("game_label", "")
    if game == "game1":
        return f"comp={cfg.get('competition_level', '')}"
    if game == "game2":
        return f"rho={cfg.get('rho', '')}, theta={cfg.get('theta', '')}"
    if game == "game3":
        return f"sigma={cfg.get('sigma', '')}, alpha={cfg.get('alpha', '')}"
    return competition_value(cfg)


def transcript_text(logs: list[dict]) -> str:
    lines = []
    for entry in logs:
        speaker = entry.get("from", "")
        phase = entry.get("phase", "")
        round_no = entry.get("round", "")
        content = clean_one_line(entry.get("content", ""), 5000)
        lines.append(f"[round={round_no} phase={phase} speaker={speaker}] {content}")
    return "\n".join(lines)


def find_quote(text: str, regexes: list[re.Pattern]) -> str:
    for sentence in split_sentences(text):
        if any(regex.search(sentence) for regex in regexes):
            return sentence
    return ""


def quote_by_code(text: str, codes: Iterable[str]) -> dict[str, str]:
    out = {}
    for code in codes:
        quote = find_quote(text, PATTERNS.get(code, []))
        if quote:
            out[code] = quote
    return out


def dynamic_applies(dynamic: Dynamic, cfg: dict, text: str, record: dict) -> bool:
    family = cfg.get("experiment_family")
    game = cfg.get("game_label")
    if dynamic.games and game not in dynamic.games:
        return False
    if dynamic.families and family not in dynamic.families:
        return False

    if dynamic.code == "multi_round_revision":
        return (record.get("final_round") or 0) and int(record["final_round"]) > 1
    if dynamic.code == "fast_consensus":
        return bool(record.get("consensus_reached")) and int(record.get("final_round") or 99) <= 1
    if dynamic.code == "no_consensus":
        return not bool(record.get("consensus_reached"))
    if dynamic.code == "high_inequality_outcome":
        utility_range = record.get("utility_range")
        utility_mean = record.get("utility_mean")
        if utility_range is None or utility_mean in (None, 0) or not math.isfinite(float(utility_range)):
            return False
        return float(utility_range) >= max(20.0, abs(float(utility_mean)) * 0.75)
    if dynamic.code == "clean_vote_integrity":
        vi = record.get("vote_integrity") or {}
        return not any(bool(vi.get(key)) for key in ("synthetic_vote_used", "contaminated", "hard_failed"))
    if dynamic.code == "procedural_vote_problem":
        vi = record.get("vote_integrity") or {}
        return any(bool(vi.get(key)) for key in ("synthetic_vote_used", "contaminated", "hard_failed"))

    regexes = PATTERNS.get(dynamic.code, [])
    return any(regex.search(text) for regex in regexes)


def classify_rollout(path: Path, root_label: str, root: Path) -> dict:
    data = load_json(path)
    cfg = data.get("config") or {}
    logs = data.get("conversation_logs") or []
    text = transcript_text(logs)
    stats = utility_stats(data.get("final_utilities"))
    phases = Counter(str(entry.get("phase", "")) for entry in logs)
    rounds = sorted({int(entry.get("round") or 0) for entry in logs if str(entry.get("round") or "").isdigit()})

    record: dict[str, object] = {
        "root_label": root_label,
        "root": str(root),
        "result_path": str(path),
        "run_dir": str(path.parent),
        "config_id": cfg.get("config_id"),
        "experiment_family": cfg.get("experiment_family"),
        "experiment_type": cfg.get("experiment_type"),
        "game_label": cfg.get("game_label"),
        "game_type": cfg.get("game_type"),
        "n_agents": cfg.get("n_agents"),
        "setting": setting_label(cfg),
        "models": "+".join(cfg.get("models") or []),
        "model_order": cfg.get("model_order"),
        "adversary_model": cfg.get("adversary_model"),
        "adversary_position": cfg.get("adversary_position"),
        "heterogeneous_run_index": cfg.get("heterogeneous_run_index"),
        "consensus_reached": bool(data.get("consensus_reached")),
        "final_round": data.get("final_round"),
        "conversation_log_count": len(logs),
        "rounds_observed": ",".join(str(value) for value in rounds),
        "discussion_count": phases.get("discussion", 0),
        "proposal_count": phases.get("proposal", 0),
        "vote_count": phases.get("vote", 0) + phases.get("vote_tabulation", 0),
        "vote_integrity": data.get("vote_integrity") or {},
    }
    record.update(stats)

    codes: list[str] = []
    for dynamic in DYNAMICS:
        if dynamic_applies(dynamic, cfg, text, record):
            codes.append(dynamic.code)
    record["dynamic_codes"] = ";".join(codes)
    record["dynamic_count"] = len(codes)

    quotes = quote_by_code(text, codes)
    record["matched_quotes_json"] = json.dumps(quotes, ensure_ascii=False, sort_keys=True)
    record["primary_quote"] = next(iter(quotes.values()), "")
    record["tldr"] = build_tldr(record, codes)
    return record


def build_tldr(record: dict, codes: list[str]) -> str:
    family = str(record.get("experiment_family") or "")
    game = str(record.get("game_label") or "")
    n_agents = record.get("n_agents")
    outcome = "consensus" if record.get("consensus_reached") else "no consensus"
    final_round = record.get("final_round")
    dynamics = [CODE_TO_TITLE.get(code, code) for code in codes if code not in {"clean_vote_integrity"}]
    dynamics_short = "; ".join(dynamics[:4]) if dynamics else "no strong textual dynamic detected"
    setting = record.get("setting") or ""
    return (
        f"{family} {game} N={n_agents} ({setting}) ended in {outcome} by round {final_round}; "
        f"main dynamics: {dynamics_short}."
    )


CODE_TO_TITLE = {dynamic.code: dynamic.title for dynamic in DYNAMICS}


def iter_result_paths() -> Iterable[tuple[str, Path, Path]]:
    for root_label, root in ROOTS:
        for path in sorted(root.glob("runs/*/experiment_results.json")):
            yield root_label, root, path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def link(path: str) -> str:
    return f"[{Path(path).name}]({path})"


def frequency_tables(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(rows)
    by_code: list[dict] = []
    for dynamic in DYNAMICS:
        count = sum(dynamic.code in str(row["dynamic_codes"]).split(";") for row in rows)
        by_code.append(
            {
                "dynamic_code": dynamic.code,
                "dynamic_title": dynamic.title,
                "count": count,
                "share": count / total if total else 0,
            }
        )
    by_code.sort(key=lambda row: (-row["count"], row["dynamic_code"]))

    by_family_game: list[dict] = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("experiment_family")), str(row.get("game_label")))].append(row)
    for (family, game), subset in sorted(groups.items()):
        for dynamic in DYNAMICS:
            count = sum(dynamic.code in str(row["dynamic_codes"]).split(";") for row in subset)
            by_family_game.append(
                {
                    "experiment_family": family,
                    "game_label": game,
                    "dynamic_code": dynamic.code,
                    "dynamic_title": dynamic.title,
                    "count": count,
                    "denominator": len(subset),
                    "share": count / len(subset) if subset else 0,
                }
            )

    by_family: list[dict] = []
    family_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        family_groups[str(row.get("experiment_family"))].append(row)
    for family, subset in sorted(family_groups.items()):
        for dynamic in DYNAMICS:
            count = sum(dynamic.code in str(row["dynamic_codes"]).split(";") for row in subset)
            by_family.append(
                {
                    "experiment_family": family,
                    "dynamic_code": dynamic.code,
                    "dynamic_title": dynamic.title,
                    "count": count,
                    "denominator": len(subset),
                    "share": count / len(subset) if subset else 0,
                }
            )
    return by_code, by_family_game, by_family


def choose_examples(rows: list[dict], code: str, max_examples: int = 5) -> list[dict]:
    candidates = []
    for row in rows:
        codes = str(row["dynamic_codes"]).split(";")
        if code not in codes:
            continue
        quote = ""
        try:
            quote = json.loads(str(row["matched_quotes_json"])).get(code, "")
        except json.JSONDecodeError:
            pass
        if not quote and code in {"multi_round_revision", "fast_consensus", "no_consensus", "high_inequality_outcome"}:
            quote = row.get("primary_quote", "")
        candidates.append({**row, "example_quote": quote})
    # Prefer diversity across family/game/n.
    selected: list[dict] = []
    seen = set()
    for row in sorted(candidates, key=lambda r: (str(r.get("experiment_family")), str(r.get("game_label")), str(r.get("n_agents")), str(r.get("result_path")))):
        key = (row.get("experiment_family"), row.get("game_label"), row.get("n_agents"))
        if key in seen and len(selected) < max_examples - 1:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= max_examples:
            break
    return selected


def write_report(rows: list[dict], by_code: list[dict], by_family_game: list[dict]) -> None:
    report = OUT_DIR / "qualitative_rollout_dynamics_report.md"
    lines: list[str] = []
    lines.append("# Qualitative Rollout Dynamics Report")
    lines.append("")
    lines.append("This report is generated from every raw `experiment_results.json` in the canonical N>2 heterogeneous and homogeneous roots.")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    fam_counts = Counter(str(row.get("experiment_family")) for row in rows)
    game_counts = Counter(str(row.get("game_label")) for row in rows)
    lines.append(f"- Total rollouts read: **{len(rows)}**")
    for family, count in sorted(fam_counts.items()):
        lines.append(f"- `{family}`: **{count}**")
    for game, count in sorted(game_counts.items()):
        lines.append(f"- `{game}` ({GAME_NAMES.get(game, game)}): **{count}**")
    lines.append("")
    lines.append("Raw roots:")
    for _, root in ROOTS:
        lines.append(f"- `{root}`")
    lines.append("")
    lines.append("The appendix file `appendix_per_rollout_tldrs.md` contains one short pull-out for every rollout. Machine-readable labels are in `rollout_dynamics_coding.csv`.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("The first pass uses transcript-grounded qualitative codes. Textual codes are assigned when the `conversation_logs` contain matching language; structural codes are assigned from saved outcome fields such as `final_round`, `consensus_reached`, final utility spread, and vote-integrity flags. This should be treated as a reproducible coding scaffold: the codebook is explicit, the per-rollout assignments are inspectable, and downstream work can refine or collapse codes.")
    lines.append("")
    lines.append("## Dynamics Codebook")
    lines.append("")
    freq_by_code = {row["dynamic_code"]: row for row in by_code}
    for dynamic in DYNAMICS:
        freq = freq_by_code[dynamic.code]
        lines.append(f"### {dynamic.title} (`{dynamic.code}`)")
        lines.append("")
        lines.append(dynamic.description)
        lines.append("")
        lines.append(f"- Corpus frequency: **{freq['count']} / {len(rows)}** ({freq['share']:.1%})")
        examples = choose_examples(rows, dynamic.code, max_examples=4)
        if examples:
            lines.append("- Illustrative samples:")
            for ex in examples:
                quote = clean_one_line(ex.get("example_quote", ""), 220)
                quote_text = f" Quote: \"{quote}\"" if quote else ""
                lines.append(
                    f"  - `{ex.get('experiment_family')}` `{ex.get('game_label')}` N={ex.get('n_agents')} "
                    f"{ex.get('setting')}: {link(str(ex.get('result_path')))}.{quote_text}"
                )
        lines.append("")
    lines.append("## Frequency By Family And Game")
    lines.append("")
    top_rows = sorted(by_family_game, key=lambda row: (row["experiment_family"], row["game_label"], -row["count"], row["dynamic_code"]))
    current = None
    for row in top_rows:
        key = (row["experiment_family"], row["game_label"])
        if key != current:
            current = key
            lines.append(f"### {key[0]} / {key[1]}")
            lines.append("")
        if row["count"]:
            lines.append(f"- `{row['dynamic_code']}`: {row['count']} / {row['denominator']} ({row['share']:.1%})")
    lines.append("")
    report.write_text("\n".join(lines))


def write_appendix(rows: list[dict]) -> None:
    path = OUT_DIR / "appendix_per_rollout_tldrs.md"
    lines = [
        "# Appendix: Per-Rollout TLDR Pull-Outs",
        "",
        "One line is included for every raw rollout read by the qualitative coding pass.",
        "",
        "| # | Family | Game | N | Setting | Outcome | Dynamics | TLDR | Raw file |",
        "|---:|---|---|---:|---|---|---|---|---|",
    ]
    for idx, row in enumerate(sorted(rows, key=lambda r: (str(r.get("experiment_family")), str(r.get("game_label")), int(r.get("config_id") or 0), str(r.get("result_path")))), start=1):
        outcome = "consensus" if row.get("consensus_reached") else "no consensus"
        dynamics = clean_one_line(str(row.get("dynamic_codes", "")).replace(";", ", "), 180)
        tldr = clean_one_line(row.get("tldr", ""), 260).replace("|", "\\|")
        lines.append(
            f"| {idx} | `{row.get('experiment_family')}` | `{row.get('game_label')}` | {row.get('n_agents')} | "
            f"{clean_one_line(row.get('setting', ''), 80)} | {outcome} r{row.get('final_round')} | "
            f"{dynamics} | {tldr} | {link(str(row.get('result_path')))} |"
        )
    path.write_text("\n".join(lines))


def write_subagent_chunk_manifests(rows: list[dict], chunk_size: int = 10) -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("chunk_*.jsonl"):
        old.unlink()
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("experiment_family")), str(r.get("game_label")), int(r.get("config_id") or 0)))
    for chunk_idx, start in enumerate(range(0, len(rows_sorted), chunk_size), start=1):
        chunk = rows_sorted[start : start + chunk_size]
        path = CHUNK_DIR / f"chunk_{chunk_idx:04d}.jsonl"
        with path.open("w") as handle:
            for row in chunk:
                handle.write(
                    json.dumps(
                        {
                            "result_path": row["result_path"],
                            "experiment_family": row["experiment_family"],
                            "game_label": row["game_label"],
                            "n_agents": row["n_agents"],
                            "setting": row["setting"],
                            "tldr": row["tldr"],
                            "dynamic_codes": row["dynamic_codes"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBAGENT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[dict] = []
    for root_label, root, path in iter_result_paths():
        try:
            rows.append(classify_rollout(path, root_label, root))
        except Exception as exc:  # noqa: BLE001 - report corrupt files, keep corpus running
            errors.append({"path": str(path), "error": repr(exc)})

    rows = [row for row in rows if row.get("experiment_family") in {"heterogeneous_random", "homogeneous_adversary", "homogeneous_control"}]

    by_code, by_family_game, by_family = frequency_tables(rows)

    coding_fields = [
        "root_label",
        "root",
        "result_path",
        "run_dir",
        "config_id",
        "experiment_family",
        "experiment_type",
        "game_label",
        "game_type",
        "n_agents",
        "setting",
        "models",
        "model_order",
        "adversary_model",
        "adversary_position",
        "heterogeneous_run_index",
        "consensus_reached",
        "final_round",
        "conversation_log_count",
        "rounds_observed",
        "discussion_count",
        "proposal_count",
        "vote_count",
        "utility_min",
        "utility_max",
        "utility_mean",
        "utility_range",
        "dynamic_codes",
        "dynamic_count",
        "primary_quote",
        "matched_quotes_json",
        "tldr",
    ]
    write_csv(OUT_DIR / "rollout_dynamics_coding.csv", rows, coding_fields)
    write_csv(OUT_DIR / "dynamic_frequencies.csv", by_code, ["dynamic_code", "dynamic_title", "count", "share"])
    write_csv(
        OUT_DIR / "dynamic_frequencies_by_family_game.csv",
        by_family_game,
        ["experiment_family", "game_label", "dynamic_code", "dynamic_title", "count", "denominator", "share"],
    )
    write_csv(
        OUT_DIR / "dynamic_frequencies_by_family.csv",
        by_family,
        ["experiment_family", "dynamic_code", "dynamic_title", "count", "denominator", "share"],
    )
    write_csv(OUT_DIR / "parse_errors.csv", errors, ["path", "error"])

    codebook_rows = [
        {
            "dynamic_code": dynamic.code,
            "dynamic_title": dynamic.title,
            "description": dynamic.description,
            "patterns": "; ".join(dynamic.patterns),
            "games": ";".join(dynamic.games),
            "families": ";".join(dynamic.families),
            "structural": dynamic.structural,
        }
        for dynamic in DYNAMICS
    ]
    write_csv(
        OUT_DIR / "dynamics_codebook.csv",
        codebook_rows,
        ["dynamic_code", "dynamic_title", "description", "patterns", "games", "families", "structural"],
    )

    write_report(rows, by_code, by_family_game)
    write_appendix(rows)
    write_subagent_chunk_manifests(rows)

    print(f"wrote {OUT_DIR}")
    print(f"rollouts_read={len(rows)} parse_errors={len(errors)} chunks={len(list(CHUNK_DIR.glob('chunk_*.jsonl')))}")


if __name__ == "__main__":
    main()
