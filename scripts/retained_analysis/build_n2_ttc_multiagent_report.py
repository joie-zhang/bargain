#!/usr/bin/env python3
"""Build the canonical N=2 + TTC + N>2 report bundle."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = PROJECT_ROOT / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
EXPANDED_REPORT = EXPANDED_DIR / "n2_plus_multiagent_comparison_report.md"
TTC_ROOT = PROJECT_ROOT / "experiments/results/ttc_native_scaling_20260502_212943"
TTC_MONITORING = TTC_ROOT / "monitoring"
TTC_PLOT_ROOT = TTC_MONITORING / "full_analysis_plots"

OUT_DIR = PROJECT_ROOT / "experiments/results/n2_ttc_multiagent_comparison_analysis_20260505"
OUT_REPORT = OUT_DIR / "n2_ttc_multiagent_comparison_report.md"

SELECTED_TTC_PLOTS = [
    "overall_by_effort.png",
    "overall_by_compute_per_call.png",
    "individual_scatter_compute_tokens_vs_payoff.png",
    "by_game_by_compute_per_call.png",
    "by_game_cell_utility_gap_compute_per_call.png",
]

SPLIT_MARKER = "\n---\n\n# Second Half: N > 2 Multi-Agent Analysis\n"
IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def copy_expanded_bundle() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(EXPANDED_DIR, OUT_DIR)
    copied_report = OUT_DIR / EXPANDED_REPORT.name
    if copied_report.exists():
        copied_report.unlink()


def copy_ttc_assets() -> None:
    target_dir = OUT_DIR / "ttc/full_analysis_plots"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in SELECTED_TTC_PLOTS:
        shutil.copy2(TTC_PLOT_ROOT / filename, target_dir / filename)


def ttc_section() -> str:
    return """
---

# Test-Time Compute Scaling

The next controlled N=2 variant asks a different scaling question: if the opponent remains `gpt-5-nano` with low reasoning, does giving the target model more requested test-time compute translate into better bargaining outcomes? The run contains 216 completed samples across GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash; all 216 result files completed, with 10 no-consensus outcomes.

The compute x-axis is normalized to mean target-model tokens per LLM call rather than total tokens per run. That avoids rewarding a low-reasoning condition for taking more negotiation rounds. GPT-5 has provider-reported reasoning tokens; Claude and Gemini use output-token proxies, so their token plots should be interpreted as visible-output scaling rather than confirmed hidden-reasoning scaling.

## TTC Summary

| family | weak -> strong levels | n comparable cells | mean target utility delta | improved | worsened | flat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 | minimal -> high | 18 | +3.26 | 6 | 7 | 5 |
| Claude Sonnet 4.6 | low -> max | 18 | +2.27 | 7 | 3 | 8 |
| Gemini 3 Flash | minimal -> high | 18 | -0.05 | 7 | 5 | 6 |

The headline is therefore not monotone compute scaling. Extra reasoning sometimes helps, but the median pattern is that the model reaches one of a small number of structurally determined deals; additional compute changes wording, timing, risk posture, or proposal repair more than it reliably changes payoff.

The individual-sample token slope tests tell the same story. Raw OLS slopes look mixed, and the more conservative game-cell/order fixed-effect slopes are small and statistically weak: GPT-5 `+0.71` utility per 1k tokens, Claude `+1.02`, Gemini `-1.69`, with fixed-effect p-values `0.667`, `0.473`, and `0.064`, respectively.

## TTC Plots

### Requested Effort

![Overall trend by requested effort](ttc/full_analysis_plots/overall_by_effort.png)

### Effective Compute Per Call

![Overall trend by effective compute tokens per call](ttc/full_analysis_plots/overall_by_compute_per_call.png)

### Individual Samples With OLS Fits

![Individual samples: reasoning tokens versus adversary payoff](ttc/full_analysis_plots/individual_scatter_compute_tokens_vs_payoff.png)

### Game Breakdowns

![Breakdown by game using effective compute tokens per call](ttc/full_analysis_plots/by_game_by_compute_per_call.png)

### Where Compute Moves Relative Payoff

![Game-cell utility gap by effective compute tokens per call](ttc/full_analysis_plots/by_game_cell_utility_gap_compute_per_call.png)

## Qualitative Readout

The qualitative audit argues that TTC is best read as a bargaining-protocol stress test rather than a scalar capability benchmark. Higher reasoning helps when the bottleneck is local arithmetic, vector correction, or recognizing a simple feasible deal. It does not reliably help when the bottleneck is first-mover anchoring, symmetric contested value, stochastic proposal selection, discount timing, negative-utility incentive compatibility, or misunderstanding the cofunding protocol.

Several recurring mechanisms explain the weak scaling:

- Payoff ceilings cap improvements. Perfectly complementary item-allocation cells and identical-ideal diplomacy cells already allow maximum or equal utility, so additional reasoning cannot create relative advantage after the obvious deal is found.
- Order and focal-bundle assignment often dominate reasoning depth. In mirrored contested cells, the side that anchors the focal bundle can win while the other side reasons correctly and still accepts the lower-value package.
- More reasoning can become discount-aware satisficing. Stronger reasoning sometimes accepts a worse immediate agreement because the per-round discount makes delay unattractive.
- Protocol mechanics add noise. Multiple accepted proposals, random tie-breaking, proposal parsing, and cofunding's single-round contribution semantics can swamp small improvements in deliberation quality.
- Some high-reasoning wins are opponent failures. Several large target advantages come from the baseline accepting bad or negative-utility contributions, not from a robust monotone benefit of more test-time compute.

The conclusion is that requested reasoning level should not be treated as an Elo-like scalar. It changes deliberation style and failure modes, and sometimes improves local execution, but realized payoff remains strongly mediated by game geometry and protocol execution. The full per-sample TTC audit is available in [test_time_compute_scaling_full_analysis.md](../ttc_native_scaling_20260502_212943/monitoring/test_time_compute_scaling_full_analysis.md).

"""


def add_figure_captions(markdown: str) -> str:
    """Add stable figure numbers below every Markdown image."""
    lines = markdown.splitlines()
    out: list[str] = []
    figure_number = 0
    for line in lines:
        out.append(line)
        match = IMAGE_RE.match(line.strip())
        if not match:
            continue
        figure_number += 1
        alt_text = match.group(1).strip() or f"Figure {figure_number}"
        terminal = "" if alt_text.endswith((".", "?", "!")) else "."
        caption = f"*Figure {figure_number}. {alt_text}{terminal}*"
        out.extend(["", caption])
    return "\n".join(out).rstrip() + "\n"


def build_report() -> None:
    expanded_text = EXPANDED_REPORT.read_text(encoding="utf-8")
    if SPLIT_MARKER not in expanded_text:
        raise RuntimeError(f"Could not find split marker in {EXPANDED_REPORT}")
    n2_text, n_gt_2_text = expanded_text.split(SPLIT_MARKER, maxsplit=1)
    n_gt_2_text = "# N > 2 Multi-Agent Analysis\n" + n_gt_2_text
    report = n2_text.rstrip() + "\n\n" + ttc_section().strip() + "\n\n---\n\n" + n_gt_2_text.lstrip()
    report = add_figure_captions(report)
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    copy_expanded_bundle()
    copy_ttc_assets()
    build_report()
    print(f"Wrote report: {OUT_REPORT}")
    print(f"Copied TTC plots: {len(SELECTED_TTC_PLOTS)}")


if __name__ == "__main__":
    main()
