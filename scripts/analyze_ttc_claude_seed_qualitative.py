#!/usr/bin/env python3
"""Validate and summarize multi-seed Claude TTC qualitative annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "analysis/ttc_claude_seed_qualitative_adjudication_20260728"
ORIGINAL_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
MANIFEST_PATH = ROOT / "all_available_rollouts_manifest.jsonl"
CODEBOOK_PATH = ROOT / "llm_tag_codebook_full50.json"
PAPER_TAGS_PATH = ROOT / "paper_tag_subset_23.json"
JUDGE_OUTPUT_ROOT = ROOT / "judge_outputs"
SUMMARY_ROOT = ROOT / "summaries"
PLOT_ROOT = ROOT / "plots"

SEED_ORDER = [42, 984, 526, 423, 1024, 128, 256, 612, 2048, 4096]
LEVEL_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
CLAUDE_LEVELS = ["low", "medium", "high", "max"]
FOCUS_GROUPS = [
    "emotional persuasion",
    "trade/compromise",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
]
GROUP_DISPLAY = {
    "emotional persuasion": "Emotional Persuasion",
    "trade/compromise": "Trade/Compromise",
    "logical persuasion": "Logical Persuasion",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-Interest/Exploitation",
    "formalization": "Formalization",
}
EVENT_REQUIRED_FIELDS = {
    "rollout_id",
    "seed",
    "source_config_id",
    "result_path",
    "interactions_path",
    "rollout_view_path",
    "family",
    "level",
    "level_index",
    "provider",
    "game_label",
    "game_cell",
    "game_type",
    "n_agents",
    "order",
    "target_agent",
    "baseline_agent",
    "speaker_agent",
    "speaker_model",
    "speaker_elo",
    "speaker_role",
    "speaker_is_target",
    "speaker_is_baseline",
    "tag_code",
    "tag_title",
    "tag_category",
    "evidence_type",
    "source_kind",
    "phase",
    "round",
    "discussion_turn",
    "log_index",
    "interaction_index",
    "speaker_order",
    "total_speakers",
    "quote",
    "rationale",
    "confidence",
    "negation_checked",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def validate_new_outputs(
    manifests: list[dict[str, Any]],
    codebook_by_code: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    errors: list[str] = []
    all_events: list[dict[str, Any]] = []
    completion_rows = []
    completed_ids = set()

    for manifest in manifests:
        rollout_id = manifest["rollout_id"]
        completion_path = JUDGE_OUTPUT_ROOT / "completions" / f"{rollout_id}.json"
        event_path = JUDGE_OUTPUT_ROOT / "events" / f"{rollout_id}.jsonl"
        if not completion_path.exists():
            continue
        completed_ids.add(rollout_id)
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            events = read_jsonl(event_path)
        except Exception as exc:
            errors.append(f"{rollout_id}: unreadable output: {type(exc).__name__}: {exc}")
            continue
        if completion.get("rollout_id") != rollout_id:
            errors.append(f"{rollout_id}: completion rollout_id mismatch")
        if completion.get("result_sha256") != manifest["result_sha256"]:
            errors.append(f"{rollout_id}: result source hash mismatch")
        if completion.get("interactions_sha256") != manifest["interactions_sha256"]:
            errors.append(f"{rollout_id}: interaction source hash mismatch")
        if int(completion.get("event_count", -1)) != len(events):
            errors.append(f"{rollout_id}: completion event_count mismatch")

        view = json.loads(Path(manifest["rollout_view_path"]).read_text(encoding="utf-8"))
        logs = {
            int(row["log_index"]): row for row in view.get("conversation_logs") or []
        }
        interactions = {
            int(row["interaction_index"]): row
            for row in view.get("target_private_interactions") or []
        }
        seen = set()
        for event_number, event in enumerate(events):
            where = f"{rollout_id}:event{event_number}"
            missing_fields = sorted(EVENT_REQUIRED_FIELDS - set(event))
            if missing_fields:
                errors.append(f"{where}: missing fields {missing_fields}")
                continue
            for field in (
                "rollout_id",
                "seed",
                "source_config_id",
                "result_path",
                "interactions_path",
                "rollout_view_path",
                "family",
                "level",
                "level_index",
                "provider",
                "game_label",
                "game_cell",
                "game_type",
                "n_agents",
                "order",
                "target_agent",
                "baseline_agent",
            ):
                if str(event.get(field)) != str(manifest.get(field)):
                    errors.append(f"{where}: {field} does not match manifest")
            tag_code = event.get("tag_code")
            tag = codebook_by_code.get(tag_code)
            if tag is None:
                errors.append(f"{where}: unknown tag {tag_code!r}")
                continue
            if event.get("tag_title") != tag["tag_title"]:
                errors.append(f"{where}: tag title mismatch")
            if event.get("tag_category") != tag["category"]:
                errors.append(f"{where}: tag category mismatch")
            if (
                event.get("speaker_agent") != manifest["target_agent"]
                or event.get("speaker_role") != "target"
                or event.get("speaker_is_target") is not True
                or event.get("speaker_is_baseline") is not False
            ):
                errors.append(f"{where}: event is not attributed to the target")
            source_kind = event.get("source_kind")
            if source_kind == "conversation_log":
                source = logs.get(event.get("log_index"))
                if source is None:
                    errors.append(f"{where}: invalid log_index")
                elif event["quote"] not in str(source.get("content") or ""):
                    errors.append(f"{where}: quote not found in conversation source")
            elif source_kind == "interaction":
                source = interactions.get(event.get("interaction_index"))
                if source is None:
                    errors.append(f"{where}: invalid interaction_index")
                elif event["quote"] not in str(source.get("response") or ""):
                    errors.append(f"{where}: quote not found in interaction source")
            elif source_kind != "formal_outcome":
                errors.append(f"{where}: invalid source_kind {source_kind!r}")
            duplicate_key = (
                rollout_id,
                tag_code,
                source_kind,
                event.get("log_index"),
                event.get("interaction_index"),
            )
            if duplicate_key in seen:
                errors.append(f"{where}: duplicate tag/source event")
            seen.add(duplicate_key)
        all_events.extend(events)
        completion_rows.append(completion)

    usage = {
        "completed_rollouts": len(completed_ids),
        "event_rows": len(all_events),
        "total_prompt_tokens": int(
            sum((row.get("usage") or {}).get("prompt_tokens") or 0 for row in completion_rows)
        ),
        "total_completion_tokens": int(
            sum(
                (row.get("usage") or {}).get("completion_tokens") or 0
                for row in completion_rows
            )
        ),
        "total_reasoning_tokens": int(
            sum(
                ((row.get("usage") or {}).get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                or 0
                for row in completion_rows
            )
        ),
        "reported_cost_usd": float(
            sum((row.get("usage") or {}).get("cost") or 0.0 for row in completion_rows)
        ),
        "rollouts_with_reported_cost": int(
            sum(
                (row.get("usage") or {}).get("cost") is not None
                for row in completion_rows
            )
        ),
        "rollouts_without_reported_cost": int(
            sum(
                (row.get("usage") or {}).get("cost") is None
                for row in completion_rows
            )
        ),
        "cost_note": (
            "OpenRouter reports per-call cost; first-party OpenAI usage does not. "
            "reported_cost_usd is therefore a partial total when both routes appear."
        ),
    }
    return all_events, errors, usage


def original_seed42_frames(
    codebook_by_code: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifests = pd.DataFrame(read_jsonl(ORIGINAL_ROOT / "all_ttc_rollouts_manifest.jsonl"))
    manifests = manifests[manifests["family"].eq("claude-sonnet-4-6")].copy()
    manifests["rollout_id"] = manifests["config_id"].map(
        lambda value: f"seed_42_config_{int(value):04d}"
    )
    manifests["seed"] = 42
    manifests["source_config_id"] = manifests["config_id"].astype(int)

    events = pd.DataFrame(read_jsonl(ORIGINAL_ROOT / "ttc_llm_event_tags.jsonl"))
    events = events[
        events["family"].eq("claude-sonnet-4-6")
        & events["speaker_is_target"].eq(True)
    ].copy()
    events["rollout_id"] = events["config_id"].map(
        lambda value: f"seed_42_config_{int(value):04d}"
    )
    events["seed"] = 42
    events["source_config_id"] = events["config_id"].astype(int)
    events["tag_category"] = events["tag_code"].map(
        {code: row["category"] for code, row in codebook_by_code.items()}
    )
    return manifests, events


def new_frames(
    manifests: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_df = pd.DataFrame(manifests)
    event_df = pd.DataFrame(events)
    if event_df.empty:
        event_df = pd.DataFrame(
            columns=[
                "rollout_id",
                "seed",
                "source_config_id",
                "family",
                "level",
                "level_index",
                "tag_code",
                "tag_category",
                "round",
                "phase",
                "discussion_turn",
            ]
        )
    return manifest_df, event_df


def intensity_tables(
    manifests: pd.DataFrame,
    events: pd.DataFrame,
    selected_tag_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_events = events[
        events["tag_code"].isin(selected_tag_codes)
        & events["tag_category"].isin(FOCUS_GROUPS)
    ].copy()
    turn_keys = [
        "rollout_id",
        "tag_category",
        "round",
        "phase",
        "discussion_turn",
    ]
    unique_turns = eligible_events.drop_duplicates(turn_keys)
    counts = (
        unique_turns.groupby(
            ["seed", "level", "level_index", "tag_category"], as_index=False
        )
        .size()
        .rename(columns={"size": "unique_turn_event_count"})
    )
    denoms = (
        manifests.groupby(["seed", "level", "level_index"], as_index=False)
        .agg(rollout_count=("rollout_id", "nunique"))
    )
    category_grid = pd.DataFrame({"tag_category": FOCUS_GROUPS})
    grid = denoms.merge(category_grid, how="cross")
    per_seed = grid.merge(
        counts,
        on=["seed", "level", "level_index", "tag_category"],
        how="left",
    )
    per_seed["unique_turn_event_count"] = (
        per_seed["unique_turn_event_count"].fillna(0).astype(int)
    )
    per_seed["unique_turn_events_per_rollout"] = (
        per_seed["unique_turn_event_count"] / per_seed["rollout_count"]
    )

    rows = []
    for keys, group in per_seed.groupby(
        ["level", "level_index", "tag_category"], sort=False
    ):
        level, level_index, category = keys
        values = group["unique_turn_events_per_rollout"].astype(float)
        n = int(group["seed"].nunique())
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        if n > 1:
            try:
                from scipy.stats import t

                critical = float(t.ppf(0.975, n - 1))
            except Exception:
                critical = 1.96
        else:
            critical = 0.0
        rows.append(
            {
                "level": level,
                "level_index": int(level_index),
                "tag_category": category,
                "seed_count": n,
                "seeds": ",".join(str(seed) for seed in sorted(group["seed"].astype(int))),
                "rollout_count_min": int(group["rollout_count"].min()),
                "rollout_count_max": int(group["rollout_count"].max()),
                "mean_unique_turn_events_per_rollout": mean,
                "seed_sem": sem,
                "ci95_low": mean - critical * sem,
                "ci95_high": mean + critical * sem,
            }
        )
    across_seed = pd.DataFrame(rows).sort_values(
        ["tag_category", "level_index"]
    )
    return per_seed.sort_values(["seed", "tag_category", "level_index"]), across_seed


def plot_intensity_ci(summary: pd.DataFrame, path: Path, title_note: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 8.65), sharex=True)
    axes = axes.ravel()
    for ax, category in zip(axes, FOCUS_GROUPS, strict=True):
        sub = summary[summary["tag_category"].eq(category)].sort_values("level_index")
        x = np.arange(len(sub))
        mean = sub["mean_unique_turn_events_per_rollout"].to_numpy(float)
        low = sub["ci95_low"].to_numpy(float)
        high = sub["ci95_high"].to_numpy(float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            color="#2ca02c",
            marker="D",
            linewidth=2.2,
            markersize=6.2,
            capsize=3.5,
        )
        ax.set_title(GROUP_DISPLAY[category], fontsize=17, pad=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["low", "med", "high", "max"], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.8)
    fig.supylabel("Average Occurrences", fontsize=22, x=0.02)
    fig.supxlabel("Requested Reasoning Effort", fontsize=20, y=0.055)
    fig.text(
        0.5,
        0.012,
        title_note,
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0.06, 0.09, 1.0, 1.0], h_pad=1.0, w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifests = read_jsonl(MANIFEST_PATH)
    codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    codebook_by_code = {row["tag_code"]: row for row in codebook}
    paper_tags = json.loads(PAPER_TAGS_PATH.read_text(encoding="utf-8"))
    paper_tag_codes = {row["tag_code"] for row in paper_tags}
    figure_compatible_codes = {
        row["tag_code"] for row in codebook if row["category"] != "coalition"
    }

    new_events, errors, usage = validate_new_outputs(manifests, codebook_by_code)
    completed_ids = {
        path.stem
        for path in (JUDGE_OUTPUT_ROOT / "completions").glob("*.json")
    }
    prepared_ids = {row["rollout_id"] for row in manifests}
    missing_outputs = sorted(prepared_ids - completed_ids)
    source_inventory = json.loads((ROOT / "source_inventory.json").read_text())
    source_missing = source_inventory["missing_source_config_ids"]

    validation = {
        "prepared_source_rollouts": len(manifests),
        "completed_annotation_rollouts": len(completed_ids & prepared_ids),
        "missing_annotation_rollout_ids": missing_outputs,
        "source_rollouts_still_missing": source_missing,
        "validation_errors": errors,
        "usage": usage,
        "current_figure_definition_tag_count": len(figure_compatible_codes),
        "paper_subset_definition_tag_count": len(paper_tag_codes),
    }
    write_json(ROOT / "validation_status.json", validation)
    if errors:
        raise RuntimeError(
            f"Annotation validation found {len(errors)} errors; "
            f"see {ROOT / 'validation_status.json'}"
        )
    if args.require_complete and (
        missing_outputs
        or any(source_missing.values())
        or len(manifests) != 648
    ):
        raise RuntimeError("The 648-rollout annotation pass is not complete")

    write_jsonl(ROOT / "ttc_claude_nine_seed_event_tags.jsonl", new_events)
    new_manifest_df, new_event_df = new_frames(manifests, new_events)
    original_manifest_df, original_event_df = original_seed42_frames(codebook_by_code)
    intensity_manifest_columns = ["rollout_id", "seed", "level", "level_index"]
    all_manifests = pd.concat(
        [
            original_manifest_df[intensity_manifest_columns],
            new_manifest_df[intensity_manifest_columns],
        ],
        ignore_index=True,
    )
    all_events = pd.concat([original_event_df, new_event_df], ignore_index=True)

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    definitions = {
        "figure_compatible_41_tags": figure_compatible_codes,
        "paper_subset_23_tags": paper_tag_codes,
    }
    for name, selected_codes in definitions.items():
        per_seed, across_seed = intensity_tables(
            all_manifests,
            all_events,
            selected_codes,
        )
        per_seed.to_csv(
            SUMMARY_ROOT / f"claude_category_intensity_by_seed_{name}.csv",
            index=False,
        )
        across_seed.to_csv(
            SUMMARY_ROOT / f"claude_category_intensity_across_seed_ci95_{name}.csv",
            index=False,
        )
        plot_intensity_ci(
            across_seed,
            PLOT_ROOT / f"claude_category_intensity_across_ten_seeds_{name}.png",
            (
                f"Claude Sonnet 4.6; mean ± 95% Student-t CI across "
                f"{int(across_seed['seed_count'].max())} seeds"
            ),
        )

    event_counts = (
        all_events.groupby(["seed", "tag_category", "tag_code"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    event_counts.to_csv(SUMMARY_ROOT / "all_event_counts_by_seed.csv", index=False)
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
