#!/usr/bin/env python3
"""Recover TTC hidden-reasoning usage across the five completed seeds.

GPT-5 artifacts contain provider-reported reasoning token counts.  The Claude
and Gemini artifacts predate capture of their provider-specific thinking-token
detail fields, but they retain both:

1. the provider-reported inclusive output-token count for every call; and
2. the exact visible response text.

For Claude and Gemini, this script sends the visible response blocks to the
corresponding provider's token-counting endpoint and reports

    estimated hidden tokens = inclusive output tokens - visible output tokens.

The estimate is exact with respect to the stored usage counts and the current
provider tokenizer, but it is labeled an estimate because historical tokenizer
versioning and provider accounting semantics are not reconstructable from the
saved artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from negotiation.provider_key_rotation import discover_provider_keys  # noqa: E402


DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_seed1024_20260725_211500"
    / "analysis/seeds42_984_526_423_1024/final_audit_all_five.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments/analysis/reviewer_ttc_hidden_tokens_20260726"
)

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
LEVEL_ORDER = {
    "gpt-5": ["minimal", "low", "medium", "high"],
    "claude-sonnet-4-6": ["low", "medium", "high", "max"],
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
}
COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_text_blocks(family: str, texts: list[str]) -> str:
    payload = json.dumps(
        {"family": family, "texts": texts},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_row(family: str, row: dict[str, Any]) -> bool:
    model = str(row.get("model_name") or "").lower()
    if family == "gpt-5":
        return "gpt-5-2025" in model
    if family == "claude-sonnet-4-6":
        return "claude" in model
    if family == "gemini-3-flash":
        return "gemini" in model
    raise ValueError(f"Unknown family: {family}")


def discover_run_rows(audit_path: Path) -> pd.DataFrame:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for seed_text, seed_info in audit["by_seed"].items():
        seed = int(seed_text)
        root = Path(seed_info["results_root"])
        for family in FAMILY_ORDER:
            paths = sorted(
                (root / family).glob(
                    "level_*/*/*/seed_*/run_1_all_interactions.json"
                )
            )
            if len(paths) != 72:
                raise RuntimeError(
                    f"Expected 72 {family} runs for seed {seed}, found {len(paths)}"
                )
            for path in paths:
                level_part = next(
                    part for part in path.parts if part.startswith("level_")
                )
                level = level_part.removeprefix("level_")
                interactions = json.loads(path.read_text(encoding="utf-8"))
                selected = [
                    row
                    for row in interactions
                    if isinstance(row, dict) and target_row(family, row)
                ]
                if not selected:
                    raise RuntimeError(f"No target interactions found in {path}")
                texts = [str(row.get("response") or "") for row in selected]
                if any(not text for text in texts):
                    raise RuntimeError(f"Empty saved target response in {path}")
                usages = [row.get("token_usage") or {} for row in selected]
                output_tokens = sum(
                    float(usage.get("output_tokens") or 0.0) for usage in usages
                )
                reported_reasoning = sum(
                    float(
                        usage.get("reasoning_tokens")
                        or usage.get("thinking_tokens")
                        or 0.0
                    )
                    for usage in usages
                )
                records.append(
                    {
                        "seed": seed,
                        "family": family,
                        "level": level,
                        "level_index": LEVEL_ORDER[family].index(level),
                        "run_path": str(path),
                        "target_calls": len(selected),
                        "provider_output_tokens": output_tokens,
                        "reported_reasoning_tokens": reported_reasoning,
                        "texts": texts,
                        "text_hash": sha256_text_blocks(family, texts),
                    }
                )
    frame = pd.DataFrame(records)
    if len(frame) != 1080:
        raise RuntimeError(f"Expected 1,080 run records, found {len(frame)}")
    for family in FAMILY_ORDER:
        family_rows = frame[frame["family"].eq(family)]
        if len(family_rows) != 360:
            raise RuntimeError(
                f"Expected 360 {family} records, found {len(family_rows)}"
            )
    return frame


def retry_count(call: Callable[[], int], label: str) -> int:
    last_error: BaseException | None = None
    for attempt in range(6):
        try:
            return int(call())
        except BaseException as exc:  # provider SDKs expose varied error bases
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Token counting failed for {label}") from last_error


def load_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def save_cache(path: Path, cache: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def count_claude_visible(
    items: list[tuple[str, list[str]]],
    workers: int,
) -> dict[str, int]:
    import anthropic

    keys = discover_provider_keys("anthropic")
    if not keys:
        raise RuntimeError(
            "No Anthropic key discovered; source the paper experiment key file first"
        )
    api_key = keys[0].value
    local = threading.local()

    def client() -> anthropic.Anthropic:
        if not hasattr(local, "client"):
            local.client = anthropic.Anthropic(api_key=api_key)
        return local.client

    overhead = retry_count(
        lambda: client()
        .messages.count_tokens(
            model="claude-sonnet-4-6",
            messages=[{"role": "assistant", "content": ""}],
        )
        .input_tokens,
        "Claude empty-message overhead",
    )

    def one(item: tuple[str, list[str]]) -> tuple[str, int]:
        digest, texts = item
        blocks = [{"type": "text", "text": text} for text in texts]
        count = retry_count(
            lambda: client()
            .messages.count_tokens(
                model="claude-sonnet-4-6",
                messages=[{"role": "assistant", "content": blocks}],
            )
            .input_tokens,
            f"Claude {digest}",
        )
        return digest, max(0, count - overhead)

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item[0] for item in items}
        for index, future in enumerate(as_completed(futures), start=1):
            digest, count = future.result()
            results[digest] = count
            if index % 50 == 0:
                print(f"Counted Claude visible tokens for {index}/{len(items)} runs")
    return results


def count_gemini_visible(
    items: list[tuple[str, list[str]]],
    workers: int,
) -> dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        keys = discover_provider_keys("google")
        if keys:
            api_key = keys[0].value
    if not api_key:
        raise RuntimeError("No Google API key discovered")
    genai.configure(api_key=api_key)
    local = threading.local()

    def model() -> Any:
        if not hasattr(local, "model"):
            local.model = genai.GenerativeModel("gemini-3-flash-preview")
        return local.model

    def one(item: tuple[str, list[str]]) -> tuple[str, int]:
        digest, texts = item
        count = retry_count(
            lambda: model().count_tokens(texts).total_tokens,
            f"Gemini {digest}",
        )
        return digest, count

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item[0] for item in items}
        for index, future in enumerate(as_completed(futures), start=1):
            digest, count = future.result()
            results[digest] = count
            if index % 50 == 0:
                print(f"Counted Gemini visible tokens for {index}/{len(items)} runs")
    return results


def fill_visible_counts(
    frame: pd.DataFrame,
    cache_path: Path,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cache = load_cache(cache_path)
    for family, counter in (
        ("claude-sonnet-4-6", count_claude_visible),
        ("gemini-3-flash", count_gemini_visible),
    ):
        subset = frame[frame["family"].eq(family)]
        missing: list[tuple[str, list[str]]] = []
        for row in subset.itertuples():
            if row.text_hash not in cache:
                missing.append((row.text_hash, row.texts))
        if missing:
            print(f"Counting {len(missing)} uncached {family} runs")
            cache.update(counter(missing, workers))
            save_cache(cache_path, cache)

    output = frame.copy()
    visible: list[float] = []
    raw_hidden: list[float] = []
    for row in output.itertuples():
        if row.family == "gpt-5":
            raw_hidden_value = float(row.reported_reasoning_tokens)
            visible_value = float(row.provider_output_tokens) - raw_hidden_value
        else:
            visible_value = float(cache[row.text_hash])
            raw_hidden_value = float(row.provider_output_tokens) - visible_value
        visible.append(visible_value)
        raw_hidden.append(raw_hidden_value)
    output["visible_output_tokens"] = visible
    output["raw_hidden_token_residual"] = raw_hidden

    # Gemini's current count_tokens endpoint uses an accounting convention that
    # is consistently about 34 tokens/call above the historical candidate-token
    # field at the minimal setting.  Zero-anchor that provider-accounting offset
    # independently within each seed.  This yields *incremental* hidden usage
    # above minimal and avoids pretending that a negative token count is real.
    gemini_offsets: dict[int, float] = {}
    gemini = output[output["family"].eq("gemini-3-flash")]
    for seed, group in gemini[gemini["level"].eq("minimal")].groupby("seed"):
        calls = float(group["target_calls"].sum())
        raw = float(group["raw_hidden_token_residual"].sum())
        gemini_offsets[int(seed)] = -raw / calls

    hidden: list[float] = []
    source: list[str] = []
    for row in output.itertuples():
        if row.family == "gpt-5":
            hidden_value = float(row.raw_hidden_token_residual)
            source_value = "provider_reported_reasoning_tokens"
        elif row.family == "gemini-3-flash":
            hidden_value = float(row.raw_hidden_token_residual) + (
                gemini_offsets[int(row.seed)] * float(row.target_calls)
            )
            source_value = (
                "output_minus_provider_tokenized_visible_"
                "zero_anchored_to_minimal_within_seed"
            )
        else:
            # A small number of individual low-effort Claude runs have negative
            # residuals from historical/current tokenizer drift.  Do not clamp
            # them run-by-run: aggregation within seed/effort cancels that
            # measurement noise, and every seed-level aggregate remains positive.
            hidden_value = float(row.raw_hidden_token_residual)
            source_value = "inclusive_output_minus_provider_tokenized_visible"
        hidden.append(hidden_value)
        source.append(source_value)
    output["hidden_reasoning_tokens"] = hidden
    output["hidden_token_source"] = source
    return output, cache


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    for (seed, family, level, level_index), group in frame.groupby(
        ["seed", "family", "level", "level_index"], sort=False
    ):
        calls = int(group["target_calls"].sum())
        output_tokens = float(group["provider_output_tokens"].sum())
        visible_tokens = float(group["visible_output_tokens"].sum())
        hidden_tokens = float(group["hidden_reasoning_tokens"].sum())
        seed_rows.append(
            {
                "seed": int(seed),
                "family": family,
                "level": level,
                "level_index": int(level_index),
                "n_runs": len(group),
                "target_calls": calls,
                "output_tokens_per_call": output_tokens / calls,
                "visible_tokens_per_call": visible_tokens / calls,
                "hidden_tokens_per_call": hidden_tokens / calls,
                "hidden_fraction_of_output": (
                    hidden_tokens / output_tokens if output_tokens else math.nan
                ),
            }
        )
    by_seed = pd.DataFrame(seed_rows).sort_values(
        ["family", "level_index", "seed"]
    )

    summary_rows: list[dict[str, Any]] = []
    for (family, level, level_index), group in by_seed.groupby(
        ["family", "level", "level_index"], sort=False
    ):
        row: dict[str, Any] = {
            "family": family,
            "level": level,
            "level_index": int(level_index),
            "seed_count": len(group),
            "runs_per_seed": int(group["n_runs"].iloc[0]),
        }
        for metric in (
            "output_tokens_per_call",
            "visible_tokens_per_call",
            "hidden_tokens_per_call",
            "hidden_fraction_of_output",
        ):
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sem = float(stats.sem(values))
            half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sem"] = sem
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["family", "level_index"]
    )
    return by_seed, summary


def plot_hidden(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=False)
    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        subset = summary[summary["family"].eq(family)].sort_values("level_index")
        x = subset["level_index"].to_numpy(dtype=float)
        y = subset["hidden_tokens_per_call_mean"].to_numpy(dtype=float)
        lo = subset["hidden_tokens_per_call_ci95_low"].to_numpy(dtype=float)
        hi = subset["hidden_tokens_per_call_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            marker="o",
            markersize=6,
            linewidth=2.2,
            capsize=4,
            color=COLORS[family],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"].astype(str))
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.28)
        if family == "gpt-5":
            ax.text(
                0.03,
                0.95,
                "direct provider count",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#475569",
            )
        else:
            ax.text(
                0.03,
                0.95,
                "output − visible estimate",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#475569",
            )
    axes[0].set_ylabel("Hidden reasoning tokens per target call")
    fig.suptitle(
        "Observed hidden reasoning usage across five independent TTC seeds",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_single_seed(by_seed: pd.DataFrame, seed: int, output: Path) -> None:
    seed_frame = by_seed[by_seed["seed"].eq(seed)].copy()
    if len(seed_frame) != 12:
        raise RuntimeError(
            f"Expected 12 family/effort rows for seed {seed}, found {len(seed_frame)}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=False)
    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        subset = seed_frame[seed_frame["family"].eq(family)].sort_values(
            "level_index"
        )
        x = subset["level_index"].to_numpy(dtype=float)
        y = subset["hidden_tokens_per_call"].to_numpy(dtype=float)
        y[np.isclose(y, 0.0, atol=1e-9)] = 0.0
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=COLORS[family],
        )
        for x_value, y_value in zip(x, y, strict=True):
            is_peak = bool(np.isclose(y_value, np.max(y)))
            ax.annotate(
                f"{y_value:,.0f}",
                (x_value, y_value),
                xytext=(0, -22 if is_peak else 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#334155",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"].astype(str))
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.28)
        ax.margins(y=0.18)
        if family == "gpt-5":
            measurement = "direct provider count"
        elif family == "gemini-3-flash":
            measurement = "output − visible; minimal-anchored"
        else:
            measurement = "output − visible estimate"
        ax.set_title(
            f"{FAMILY_LABELS[family]}\n{measurement}",
            fontsize=13,
            color="#111827",
            pad=10,
        )
    axes[0].set_ylabel("Hidden reasoning tokens per target call")
    fig.suptitle(
        f"Observed hidden reasoning usage in the original TTC seed ({seed})",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_single_seed_report(
    by_seed: pd.DataFrame,
    seed: int,
    output: Path,
) -> None:
    table = (
        by_seed[by_seed["seed"].eq(seed)]
        .sort_values(["family", "level_index"])
        [
            [
                "family",
                "level",
                "n_runs",
                "target_calls",
                "hidden_tokens_per_call",
                "visible_tokens_per_call",
                "output_tokens_per_call",
                "hidden_fraction_of_output",
            ]
        ]
        .copy()
    )
    numeric = [
        column
        for column in table.columns
        if column not in {"family", "level", "n_runs", "target_calls"}
    ]
    for column in numeric:
        table.loc[np.isclose(table[column], 0.0, atol=1e-9), column] = 0.0
    table[numeric] = table[numeric].round(2)

    trend_lines: list[str] = []
    for family in FAMILY_ORDER:
        subset = table[table["family"].eq(family)]
        values = subset["hidden_tokens_per_call"].to_numpy(dtype=float)
        levels = subset["level"].astype(str).tolist()
        sequence = " → ".join(f"{value:,.0f}" for value in values)
        nondecreasing = bool(np.all(np.diff(values) >= -1e-9))
        trend_lines.append(
            f"- **{FAMILY_LABELS[family]}:** {sequence} tokens/call across "
            f"{'/'.join(levels)}; "
            f"{'monotonic' if nondecreasing else 'not monotonic'}."
        )

    lines = [
        f"# Original-seed TTC hidden-reasoning-token audit (seed {seed})",
        "",
        "This uses the same accounting as the five-seed figure: GPT-5 uses its",
        "directly reported reasoning-token field; Claude and Gemini use the",
        "post-hoc output-minus-provider-tokenized-visible estimate. Gemini is",
        "zero-anchored to its minimal setting within this seed.",
        "",
        f"![Seed {seed} hidden reasoning usage](hidden_reasoning_tokens_seed{seed}.png)",
        "",
        "## Trend check",
        "",
        *trend_lines,
        "",
        "Thus, the original seed does **not** show a clean effort-to-token increase",
        "for GPT-5 or Claude. Gemini does increase monotonically.",
        "",
        "## Values",
        "",
        table.to_markdown(index=False),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame, output: Path) -> None:
    table = summary[
        [
            "family",
            "level",
            "hidden_tokens_per_call_mean",
            "hidden_tokens_per_call_ci95_low",
            "hidden_tokens_per_call_ci95_high",
            "visible_tokens_per_call_mean",
            "output_tokens_per_call_mean",
            "hidden_fraction_of_output_mean",
        ]
    ].copy()
    numeric = [column for column in table.columns if column not in {"family", "level"}]
    table[numeric] = table[numeric].round(2)
    lines = [
        "# Five-seed TTC hidden-reasoning-token audit",
        "",
        "GPT-5 uses the directly reported reasoning-token field. Claude and Gemini",
        "use provider-billed output tokens minus the same provider's token count of",
        "the exact visible response blocks retained in the artifacts.",
        "",
        "The Claude/Gemini values are post-hoc estimates, not direct historical",
        "thinking-token fields: tokenizer versions and provider accounting semantics",
        "may have changed since the runs were generated. Gemini is reported as",
        "incremental hidden usage above its minimal setting, zero-anchored separately",
        "within each seed to remove a stable historical/current accounting offset.",
        "",
        "![Hidden reasoning usage](hidden_reasoning_tokens_five_seeds.png)",
        "",
        table.to_markdown(index=False),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "provider_visible_token_cache.json"

    frame = discover_run_rows(args.audit.resolve())
    frame, _ = fill_visible_counts(frame, cache_path, max(1, args.workers))
    by_seed, summary = summarize(frame)

    run_csv = output_dir / "run_level_hidden_tokens.csv"
    seed_csv = output_dir / "seed_level_hidden_tokens.csv"
    summary_csv = output_dir / "five_seed_hidden_tokens_summary.csv"
    plot_path = output_dir / "hidden_reasoning_tokens_five_seeds.png"
    report_path = output_dir / "hidden_reasoning_tokens_report.md"
    original_seed = 42
    original_seed_csv = output_dir / f"seed{original_seed}_hidden_tokens_summary.csv"
    original_seed_plot = output_dir / f"hidden_reasoning_tokens_seed{original_seed}.png"
    original_seed_report = output_dir / f"hidden_reasoning_tokens_seed{original_seed}.md"

    export = frame.drop(columns=["texts"])
    export.to_csv(run_csv, index=False)
    by_seed.to_csv(seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_hidden(summary, plot_path)
    write_report(summary, report_path)
    by_seed[by_seed["seed"].eq(original_seed)].to_csv(
        original_seed_csv,
        index=False,
    )
    plot_single_seed(by_seed, original_seed, original_seed_plot)
    write_single_seed_report(by_seed, original_seed, original_seed_report)

    print(f"Wrote {run_csv}")
    print(f"Wrote {seed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {original_seed_csv}")
    print(f"Wrote {original_seed_plot}")
    print(f"Wrote {original_seed_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
