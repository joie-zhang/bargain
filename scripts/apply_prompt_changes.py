#!/usr/bin/env python3
"""
=============================================================================
Apply Accepted Prompt Changes
=============================================================================

Reads the decisions file produced by the Streamlit reviewer and applies all
accepted changes to the game environment Python source files.

Usage:
    python scripts/apply_prompt_changes.py
    python scripts/apply_prompt_changes.py --decisions path/to/decisions.json
    python scripts/apply_prompt_changes.py --dry-run   # print diffs, no writes

What it modifies:
    game_environments/co_funding.py
    game_environments/item_allocation.py
    game_environments/diplomatic_treaty.py

What it reads:
    docs/reference/prompt_change_decisions.json   (Streamlit output)

Each change is implemented as a named function. If a before-string is not found
in the target file the script prints an error and skips that change rather than
silently doing nothing.

=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DECISIONS_FILE = BASE_DIR / "docs/reference/prompt_change_decisions.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_file(rel_path: str) -> str:
    return (BASE_DIR / rel_path).read_text()


def write_file(rel_path: str, content: str, dry_run: bool) -> None:
    path = BASE_DIR / rel_path
    if dry_run:
        print(f"  [DRY-RUN] Would write {path.relative_to(BASE_DIR)}")
        return
    path.write_text(content)


def apply_single(rel_path: str, before: str, after: str, change_id: str, dry_run: bool) -> bool:
    content = read_file(rel_path)
    # Idempotency: find the first non-empty line in "after" that is absent from
    # "before". If that line is already in the file the change has been applied.
    unique_marker = next(
        (line for line in after.split("\n") if line.strip() and line not in before),
        None,
    )
    if unique_marker and unique_marker in content:
        print(f"  ✓ {change_id}: already applied (idempotent skip)")
        return True
    if before not in content:
        print(f"  ✗ {change_id}: pattern not found in {rel_path}")
        print(f"    First 80 chars of pattern: {before[:80]!r}")
        return False
    new_content = content.replace(before, after, 1)
    write_file(rel_path, new_content, dry_run)
    action = "Would apply" if dry_run else "Applied"
    print(f"  ✓ {change_id}: {action}")
    return True


# ── Individual change implementations ────────────────────────────────────────
# Each function returns (file_path, before, after) for apply_single().

def change_G3_01():
    before = (
        "- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost\n"
        "- Contributions to UNFUNDED projects are REFUNDED (you don't lose that money)"
    )
    after = (
        "- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost\n"
        "- **ALL-OR-NOTHING**: Funding is binary — a project either reaches its full cost threshold (funded) or it doesn't (unfunded). There is no partial benefit from contributing to a project that falls short of its threshold.\n"
        "- Contributions to UNFUNDED projects are REFUNDED (you don't lose that money)"
    )
    return "game_environments/co_funding.py", before, after


def change_G3_02():
    before = (
        "- Your goal: maximize your utility by strategically choosing contributions\n"
        "\n"
        "Please acknowledge that you understand these rules and are ready to participate!"
    )
    after = (
        "- Your goal: maximize your utility by strategically choosing contributions\n"
        "\n"
        "**BUDGET CONSTRAINT:**\n"
        "- The combined budgets of all participants may NOT be sufficient to fund all projects\n"
        "- You MUST prioritize — coordinate on a subset of projects you can collectively afford to fully fund\n"
        "\n"
        "Please acknowledge that you understand these rules and are ready to participate!"
    )
    return "game_environments/co_funding.py", before, after


def change_G3_03():
    # Modification: removed the prescriptive "never contribute more than your valuation"
    # line per user review — keeps only the factual negative-utility warning.
    before = (
        "- You gain value from funded projects but pay for your contributions to them\n"
        "- Contributions to unfunded projects cost you nothing (refunded)"
    )
    after = (
        "- You gain value from funded projects but pay for your contributions to them\n"
        "- **IMPORTANT**: If your contribution to a funded project exceeds your valuation, your net utility from that project is NEGATIVE\n"
        "- Contributions to unfunded projects cost you nothing (refunded)"
    )
    return "game_environments/co_funding.py", before, after


def change_G3_04():
    # Modification: uses dynamic computation (total_budget / total_cost * 100),
    # correct for all sigma values — not a hardcoded percentage.
    before = (
        '        lines.append(f"**TOTAL BUDGET (all participants):** {game_state[\'total_budget\']:.2f}")\n'
        '        lines.append("")\n'
        '        lines.append("**STRATEGIC INSIGHT:**")'
    )
    after = (
        '        lines.append(f"**TOTAL BUDGET (all participants):** {game_state[\'total_budget\']:.2f}")\n'
        "        _coverage = round(game_state['total_budget'] / game_state['total_cost'] * 100) if game_state['total_cost'] > 0 else 0\n"
        '        lines.append(f"**COLLECTIVE COVERAGE:** {_coverage}% of total project costs — you cannot fund all projects; coordinate on a subset")\n'
        '        lines.append("")\n'
        '        lines.append("**STRATEGIC INSIGHT:**")'
    )
    return "game_environments/co_funding.py", before, after


def change_G3_05():
    # Modification per user review:
    # (1) Unfunded refund framed as opportunity cost within a round
    # (2) Utility formula scoped to ALL funded projects (free-rider benefit included)
    # NOTE: before pattern anchors after G3-04's COLLECTIVE COVERAGE line so both
    # changes compose correctly when applied fresh.
    before = (
        '        lines.append(f"**COLLECTIVE COVERAGE:** {_coverage}% of total project costs — you cannot fund all projects; coordinate on a subset")\n'
        '        lines.append("")\n'
        '        lines.append("**STRATEGIC INSIGHT:**")'
    )
    after = (
        '        lines.append(f"**COLLECTIVE COVERAGE:** {_coverage}% of total project costs — you cannot fund all projects; coordinate on a subset")\n'
        '        lines.append("")\n'
        '        lines.append("**HOW YOUR UTILITY IS COMPUTED:**")\n'
        '        lines.append("- For each FUNDED project: your_utility = your_valuation \u2212 your_contribution (negative if you over-contribute)")\n'
        '        lines.append("- For UNFUNDED projects: your contribution is refunded at end of game, but within a round money pledged to one project cannot be reallocated to another \u2014 choose carefully and coordinate to ensure your highest-value projects get funded")\n'
        '        lines.append("- Total utility = sum of (valuation \u2212 contribution) across ALL funded projects, including projects funded entirely by others (where your contribution = 0, giving you full valuation as free utility)")\n'
        '        lines.append("")\n'
        '        lines.append("**STRATEGIC INSIGHT:**")'
    )
    return "game_environments/co_funding.py", before, after


def change_G3_06():
    # Adds last-round warning after "Currently funded projects" in pledge prompt
    before = (
        "**Currently funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}{reasoning_instruction}\n"
        "\n"
        "{format_section}"
    )
    after = (
        "**Currently funded projects (LAST ROUND):** {[projects[j]['name'] for j in funded] if funded else 'None'}\n"
        "**NOTE:** All status above reflects LAST ROUND's results. This round starts fresh — reaffirm your contributions or previously funded projects will become unfunded.{reasoning_instruction}\n"
        "\n"
        "{format_section}"
    )
    return "game_environments/co_funding.py", before, after


def change_G3_07():
    before = (
        "Vote **yay** if you are willing to finalize this exact profile now.\n"
        "Vote **nay** if you want another revision round."
    )
    after = (
        "Vote **yay** if you are satisfied with the current pledge profile and want to finalize it now.\n"
        "Vote **nay** if you want one more revision round to improve contributions.\n"
        "\n"
        "**CONSEQUENCE:** If ALL participants vote yay, the game ends immediately with this pledge profile as the final outcome. If ANY participant votes nay, one more revision round occurs."
    )
    return "game_environments/co_funding.py", before, after


def change_G3_08():
    before = '        lines.append("Consider adjusting your contributions based on these aggregate results.")'
    after = (
        '        lines.append("When deciding how to adjust your contributions next round:")\n'
        '        lines.append("- **Near threshold** (>70% funded): consider increasing your contribution to push it over")\n'
        '        lines.append("- **Far from threshold** (<30% funded): consider reallocating those contributions to better-funded projects")\n'
        '        lines.append("- **Already funded**: you may reduce slightly if others will maintain coverage \u2014 but do not drop below what is needed")\n'
        '        lines.append("- **Over-contributed**: if you contributed more than your valuation to a funded project, reduce to your valuation")'
    )
    return "game_environments/co_funding.py", before, after


def change_G1_01():
    before = (
        "Please vote on this proposal. Consider:\n"
        "- How this allocation affects your utility\n"
        "- Whether you might get a better deal by continuing negotiation\n"
        "- The strategic implications of accepting vs. rejecting{reasoning_instruction}"
    )
    after = (
        "**REMINDER — YOUR UTILITY:**\n"
        "- Your utility = sum of preference values for items you receive, multiplied by the round discount\n"
        "- Round 1: 100% | Round 2: {self.config.gamma_discount * 100:.0f}% | Round 3: {self.config.gamma_discount**2 * 100:.0f}% (\\u03b3={self.config.gamma_discount} per round)\n"
        "- If no deal is reached by the final round, your utility is 0\n"
        "\n"
        "Please vote on this proposal. Consider:\n"
        "- How this allocation affects your utility\n"
        "- Whether you might get a better deal by continuing negotiation\n"
        "- The strategic implications of accepting vs. rejecting{reasoning_instruction}"
    )
    return "game_environments/item_allocation.py", before, after


def change_G1_02():
    before = "- Your maximum possible utility: {max_utility:.2f} points (if you get ALL items)"
    after = "- Your theoretical maximum utility: {max_utility:.2f} points (if you received ALL items — unrealistic in negotiation; use this only as an upper bound)"
    return "game_environments/item_allocation.py", before, after


def change_G2_01():
    before = (
        "- Utility is discounted by {self.config.gamma_discount} per round — early agreement is better\n"
        "\n"
        "Please acknowledge that you understand these rules and are ready to negotiate!"
    )
    after = (
        "- Utility is discounted by {self.config.gamma_discount} per round — early agreement is better\n"
        "- **If the final round ends without unanimous agreement, all parties receive ZERO utility** — any positive-utility compromise is strictly better than no deal\n"
        "\n"
        "Please acknowledge that you understand these rules and are ready to negotiate!"
    )
    return "game_environments/diplomatic_treaty.py", before, after


def change_G2_02():
    before = (
        "- A score of 0.0 means fully opposed; 1.0 means fully supportive on each proposition\n"
        "- Maximum utility = 100.0 (every proposition at your exact ideal score)"
    )
    after = (
        "- A rate of 0.0 means 0% (minimum policy level); 1.0 means 100% (maximum policy level) on each issue\n"
        "- Maximum utility = 100.0 (every issue resolved at your exact ideal rate)"
    )
    return "game_environments/diplomatic_treaty.py", before, after


# ── Registry ──────────────────────────────────────────────────────────────────
CHANGE_REGISTRY = {
    "G3-01": change_G3_01,
    "G3-02": change_G3_02,
    "G3-03": change_G3_03,
    "G3-04": change_G3_04,
    "G3-05": change_G3_05,
    "G3-06": change_G3_06,
    "G3-07": change_G3_07,
    "G3-08": change_G3_08,
    "G1-01": change_G1_01,
    "G1-02": change_G1_02,
    "G2-01": change_G2_01,
    "G2-02": change_G2_02,
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Apply accepted prompt changes.")
    parser.add_argument(
        "--decisions",
        default=str(DECISIONS_FILE),
        help="Path to prompt_change_decisions.json (default: docs/reference/prompt_change_decisions.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files",
    )
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        print(f"ERROR: decisions file not found: {decisions_path}")
        print("Run the Streamlit reviewer first and save your decisions.")
        sys.exit(1)

    with open(decisions_path) as f:
        decisions = json.load(f)

    accepted = {k: v for k, v in decisions.items() if v.get("decision") == "accept"}
    skipped = {k: v for k, v in decisions.items() if v.get("decision") == "skip"}
    declined = {k: v for k, v in decisions.items() if v.get("decision") == "decline"}

    print(f"Decisions: {len(accepted)} accepted, {len(skipped)} skipped, {len(declined)} declined")
    print("─" * 60)

    if args.dry_run:
        print("[DRY-RUN MODE — no files will be written]\n")

    applied = []
    errors = []

    for change_id in accepted:
        if change_id not in CHANGE_REGISTRY:
            print(f"  ✗ {change_id}: no implementation found in registry")
            errors.append(change_id)
            continue
        rel_path, before, after = CHANGE_REGISTRY[change_id]()
        success = apply_single(rel_path, before, after, change_id, args.dry_run)
        if success:
            applied.append(change_id)
        else:
            errors.append(change_id)
        note = accepted[change_id].get("notes", "")
        if note:
            print(f"    Note: {note}")

    print("─" * 60)
    print(f"Applied: {len(applied)}  |  Errors: {len(errors)}  |  Skipped/Declined: {len(skipped) + len(declined)}")

    if errors:
        print(f"\nFailed changes: {errors}")
        print("These patterns were not found in the source files.")
        print("Check that the source files have not been modified since the audit was generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
