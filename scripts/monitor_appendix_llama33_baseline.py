#!/usr/bin/env python3
"""Read-only monitor for the Llama 3.3 appendix baseline Slurm sweep."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
    short_model_name,
)


RUN_ROOTS = {
    "game1": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game1_202605",
    "game2": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game2_202605",
    "game3": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game3_202605",
}
GAME_LABELS = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
ELO_DOC = (
    PROJECT_ROOT
    / "docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/results/appendix_llama33_baseline_monitoring/current"
)

MODEL_ALIASES = {
    "amazon-nova-micro": "amazon-nova-micro-v1.0",
    "amazon-nova-pro": "amazon-nova-pro-v1.0",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "deepseek-r1": "deepseek-r1-0528",
}
LOG_ERROR_PATTERN = re.compile(
    r"traceback|exception|error|failed|timeout|oom|rate limit|context length|invalid json",
    re.IGNORECASE,
)


def canon(model_name: Any) -> str:
    raw = str(model_name)
    return canonical_model_name(MODEL_ALIASES.get(raw, raw))


def parse_elo_markdown() -> dict[str, int]:
    elo_by_model: dict[str, int] = {}
    patterns = [
        re.compile(r"^\|\s*\d+\s*\|\s*`?([^|`]+?)`?\s*\|\s*(\d+)\s*\|"),
        re.compile(r"^\|\s*\d+\s*\|\s*[^|]+?\|\s*`?([^|`]+?)`?\s*\|\s*(\d+)\s*\|"),
    ]
    for line in ELO_DOC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for pattern in patterns:
            match = pattern.match(stripped)
            if match:
                elo_by_model[canon(match.group(1).strip())] = int(match.group(2))
                break
    return elo_by_model


def read_manifests() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for game, root in RUN_ROOTS.items():
        manifest = root / "slurm/submitted_jobs.txt"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            config_id, job_id, config_file = line.split(",", 2)
            rows.append(
                {
                    "game": game,
                    "config_id": config_id,
                    "job_id": job_id,
                    "config_file": config_file,
                }
            )
    return rows


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def scheduler_status(manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    job_to_row = {row["job_id"]: row for row in manifest_rows}
    queue_text = run_text(
        ["bash", "-lc", 'squeue -h -u "$USER" -o "%i|%j|%T|%M|%R" || true']
    )
    queue_state: dict[str, dict[str, str]] = {}
    for line in queue_text.splitlines():
        parts = line.split("|", 4)
        if len(parts) < 3 or parts[0] not in job_to_row:
            continue
        queue_state[parts[0]] = {
            "job_id": parts[0],
            "job_name": parts[1],
            "state": parts[2],
            "elapsed": parts[3] if len(parts) > 3 else "",
            "reason": parts[4] if len(parts) > 4 else "",
        }

    missing_ids = [row["job_id"] for row in manifest_rows if row["job_id"] not in queue_state]
    terminal_state: dict[str, dict[str, str]] = {}
    for start in range(0, len(missing_ids), 100):
        chunk = missing_ids[start : start + 100]
        if not chunk:
            continue
        sacct_text = run_text(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(chunk),
                "--format=JobIDRaw,JobName,State,ExitCode,Elapsed",
                "-P",
            ]
        )
        for line in sacct_text.splitlines():
            parts = line.split("|")
            if len(parts) < 5 or parts[0] not in job_to_row:
                continue
            terminal_state[parts[0]] = {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "exit_code": parts[3],
                "elapsed": parts[4],
            }

    state_counts = Counter()
    per_game = defaultdict(Counter)
    terminal_missing = 0
    for row in manifest_rows:
        job_id = row["job_id"]
        if job_id in queue_state:
            state = queue_state[job_id]["state"]
        elif job_id in terminal_state:
            state = terminal_state[job_id]["state"]
        else:
            state = "UNKNOWN_TERMINAL_OR_DELAYED"
            terminal_missing += 1
        state_counts[state] += 1
        per_game[row["game"]][state] += 1

    pending = sum(v for k, v in state_counts.items() if k.startswith("PENDING"))
    running = sum(v for k, v in state_counts.items() if k.startswith("RUNNING"))
    completed = state_counts.get("COMPLETED", 0)
    terminal_non_success = sum(
        v
        for k, v in state_counts.items()
        if k not in {"PENDING", "RUNNING", "COMPLETED"}
    )

    return {
        "total": len(manifest_rows),
        "started": len(manifest_rows) - pending,
        "finished": completed,
        "queued": pending,
        "in_progress": running,
        "state_counts": dict(sorted(state_counts.items())),
        "per_game": {game: dict(sorted(counts.items())) for game, counts in per_game.items()},
        "terminal_non_success": terminal_non_success,
        "terminal_missing_from_sacct": terminal_missing,
    }


def result_file(output_dir: Path) -> Path | None:
    for name in ("experiment_results.json", "run_1_experiment_results.json"):
        path = output_dir / name
        if path.exists():
            return path
    return None


def ordered_agent_ids(final_utilities: dict[str, Any]) -> list[str]:
    if {"Agent_1", "Agent_2"}.issubset(final_utilities):
        return ["Agent_1", "Agent_2"]
    if {"Agent_Alpha", "Agent_Beta"}.issubset(final_utilities):
        return ["Agent_Alpha", "Agent_Beta"]
    return sorted(final_utilities)


def utility_by_model(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    final_utilities = payload.get("final_utilities") or {}
    agent_performance = payload.get("agent_performance") or {}
    if not isinstance(final_utilities, dict):
        return {}
    expected = {canon(cfg["baseline_model"]), canon(cfg["adversary_model"])}
    mapped: dict[str, float] = {}

    for agent_id, utility in final_utilities.items():
        perf = agent_performance.get(agent_id) if isinstance(agent_performance, dict) else None
        raw_model = perf.get("model") if isinstance(perf, dict) else None
        if raw_model:
            model = canon(raw_model)
            if model in expected:
                mapped[model] = float(utility)

    if expected.issubset(mapped):
        return mapped

    agent_ids = ordered_agent_ids(final_utilities)
    models = [canon(model) for model in cfg.get("models", [])]
    for agent_id, model in zip(agent_ids, models):
        if model in expected and model not in mapped:
            mapped[model] = float(final_utilities[agent_id])
    return mapped


def transcript_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    logs = payload.get("conversation_logs") or []
    if not isinstance(logs, list):
        logs = []
    contents = [
        str(entry.get("content") or "") if isinstance(entry, dict) else str(entry or "")
        for entry in logs
    ]
    normalized = [
        re.sub(r"\s+", " ", content.strip().lower())[:500]
        for content in contents
        if content.strip()
    ]
    joined = "\n".join(contents).lower()
    return {
        "messages": len(logs),
        "empty_messages": sum(1 for content in contents if not content.strip()),
        "duplicate_messages": len(normalized) - len(set(normalized)),
        "avg_chars": sum(map(len, contents)) / len(contents) if contents else 0.0,
        "error_terms_in_transcript": sum(
            joined.count(term)
            for term in ["error", "exception", "traceback", "failed to", "invalid json", "timeout"]
        ),
    }


def collect_result_rows(elo_by_model: dict[str, int]) -> tuple[list[dict[str, Any]], Counter, list[str]]:
    rows: list[dict[str, Any]] = []
    missing = Counter()
    parse_errors: list[str] = []

    for game, root in RUN_ROOTS.items():
        index_path = root / "configs/experiment_index.csv"
        with index_path.open(newline="", encoding="utf-8") as handle:
            for index_row in csv.DictReader(handle):
                cfg_path = root / "configs" / index_row["config_file"]
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                path = result_file(PROJECT_ROOT / cfg["output_dir"])
                if path is None:
                    missing[game] += 1
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    utilities = utility_by_model(payload, cfg)
                    baseline = canon(cfg["baseline_model"])
                    adversary = canon(cfg["adversary_model"])
                    if baseline not in utilities or adversary not in utilities:
                        parse_errors.append(f"{game} {index_row['config_file']}: missing utility mapping")
                        continue

                    if game == "game1":
                        competition_value = float(cfg["competition_level"])
                        competition_name = "competition_level"
                    else:
                        competition_value = float(cfg["competition_index"])
                        competition_name = "competition_index"

                    rows.append(
                        {
                            "game": game,
                            "game_label": GAME_LABELS[game],
                            "config_file": index_row["config_file"],
                            "adversary": adversary,
                            "short": short_model_name(adversary),
                            "elo": elo_by_model.get(adversary, math.nan),
                            "advu": float(utilities[adversary]),
                            "baseu": float(utilities[baseline]),
                            "comp": competition_value,
                            "competition_name": competition_name,
                            "order": cfg.get("model_order"),
                            "cons": bool(payload.get("consensus_reached")),
                            "round": payload.get("final_round"),
                            "rho": cfg.get("rho"),
                            "theta": cfg.get("theta"),
                            "alpha": cfg.get("alpha"),
                            "sigma": cfg.get("sigma"),
                            "result_path": str(path),
                            **transcript_metrics(payload),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue monitoring.
                    parse_errors.append(
                        f"{game} {index_row['config_file']}: {type(exc).__name__}: {exc}"
                    )

    return rows, missing, parse_errors


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return math.nan
    x_arr = np.array([pair[0] for pair in pairs], dtype=float)
    y_arr = np.array([pair[1] for pair in pairs], dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_models(rows: list[dict[str, Any]], elo_by_model: dict[str, int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["game"], row["adversary"])].append(row)

    summary: list[dict[str, Any]] = []
    for (game, model), model_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], elo_by_model.get(item[0][1], 0))
    ):
        final_rounds = [
            float(row["round"]) for row in model_rows if row["round"] is not None
        ]
        summary.append(
            {
                "game": game,
                "game_label": GAME_LABELS[game],
                "model": model,
                "short": short_model_name(model),
                "elo": elo_by_model.get(model, math.nan),
                "n": len(model_rows),
                "adv": mean([float(row["advu"]) for row in model_rows]),
                "base": mean([float(row["baseu"]) for row in model_rows]),
                "cons": mean([1.0 if row["cons"] else 0.0 for row in model_rows]),
                "round": mean(final_rounds),
            }
        )
    return summary


def write_csvs(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    run_fields = [
        "game",
        "game_label",
        "config_file",
        "adversary",
        "short",
        "elo",
        "advu",
        "baseu",
        "comp",
        "competition_name",
        "order",
        "cons",
        "round",
        "rho",
        "theta",
        "alpha",
        "sigma",
        "messages",
        "empty_messages",
        "duplicate_messages",
        "avg_chars",
        "error_terms_in_transcript",
        "result_path",
    ]
    with (OUTPUT_DIR / "completed_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, run_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["game", "game_label", "model", "short", "elo", "n", "adv", "base", "cons", "round"]
    with (OUTPUT_DIR / "summary_by_game_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, summary_fields)
        writer.writeheader()
        writer.writerows(summary)


def make_elo_plot(summary: list[dict[str, Any]], key: str, ylabel: str, filename: str) -> Path | None:
    games = [game for game in ("game1", "game2", "game3") if any(row["game"] == game for row in summary)]
    if not games:
        return None
    fig, axes = plt.subplots(1, len(games), figsize=(6 * len(games), 4.8), squeeze=False)
    for ax, game in zip(axes[0], games):
        game_rows = [row for row in summary if row["game"] == game]
        xs = [float(row["elo"]) for row in game_rows]
        ys = [float(row[key]) for row in game_rows]
        ax.scatter(xs, ys, s=[45 + 8 * int(row["n"]) for row in game_rows], color="#2563eb")
        if len(xs) > 1:
            ax.plot(xs, ys, color="#2563eb", alpha=0.5)
        for row, x_val, y_val in zip(game_rows, xs, ys):
            ax.annotate(
                f"{row['short']}\nn={row['n']}",
                (x_val, y_val),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(GAME_LABELS[game])
        ax.set_xlabel("Adversary Elo")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Partial {ylabel} vs Elo", y=1.03)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def make_competition_plot(rows: list[dict[str, Any]]) -> Path | None:
    games = [game for game in ("game1", "game2", "game3") if any(row["game"] == game for row in rows)]
    if not games:
        return None
    fig, axes = plt.subplots(1, len(games), figsize=(6 * len(games), 4.8), squeeze=False)
    for ax, game in zip(axes[0], games):
        game_rows = [row for row in rows if row["game"] == game]
        ax.scatter(
            [float(row["comp"]) for row in game_rows],
            [float(row["advu"]) for row in game_rows],
            c=[float(row["elo"]) for row in game_rows],
            cmap="viridis",
            s=50,
        )
        ax.set_title(GAME_LABELS[game])
        ax.set_xlabel("competition_level" if game == "game1" else "competition_index")
        ax.set_ylabel("Adversary utility")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Partial adversary utility vs competition ({len(rows)} runs)", y=1.03)
    fig.tight_layout()
    path = OUTPUT_DIR / "partial_adversary_utility_vs_competition_by_game.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def scan_logs() -> dict[str, Any]:
    matches: list[str] = []
    provider_reports: list[str] = []
    for root in RUN_ROOTS.values():
        for path in sorted((root / "slurm_logs").glob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if LOG_ERROR_PATTERN.search(line):
                    matches.append(f"{path}:{idx}: {line}")
        provider_path = root / "monitoring/provider_failures.md"
        if provider_path.exists() and provider_path.stat().st_size > 0:
            provider_reports.append(provider_path.read_text(encoding="utf-8", errors="replace"))
    return {
        "log_error_match_count": len(matches),
        "log_error_sample": matches[:20],
        "provider_reports": provider_reports,
    }


def write_report(
    status: dict[str, Any],
    rows: list[dict[str, Any]],
    missing: Counter,
    parse_errors: list[str],
    summary: list[dict[str, Any]],
    plots: list[Path],
    log_info: dict[str, Any],
) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"Last updated: {now}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "Scheduler:",
        f"  total={status['total']}",
        f"  started={status['started']}",
        f"  finished={status['finished']}",
        f"  queued={status['queued']}",
        f"  in_progress={status['in_progress']}",
        f"  state_counts={status['state_counts']}",
        f"  per_game={status['per_game']}",
        "",
        f"Completed parsed runs: {len(rows)}",
    ]
    for game in ("game1", "game2", "game3"):
        game_rows = [row for row in rows if row["game"] == game]
        lines.append(
            f"{GAME_LABELS[game]} completed parsed runs: {len(game_rows)}; "
            f"missing result configs: {missing[game]}"
        )
        if not game_rows:
            continue
        final_rounds = [
            float(row["round"]) for row in game_rows if row["round"] is not None
        ]
        lines.append(
            "  "
            f"consensus_rate={mean([1.0 if row['cons'] else 0.0 for row in game_rows]):.3f}; "
            f"avg_final_round={mean(final_rounds):.2f}; "
            f"avg_messages={mean([float(row['messages']) for row in game_rows]):.1f}; "
            f"empty_messages={sum(int(row['empty_messages']) for row in game_rows)}; "
            f"duplicate_messages={sum(int(row['duplicate_messages']) for row in game_rows)}; "
            f"transcript_error_terms={sum(int(row['error_terms_in_transcript']) for row in game_rows)}"
        )
        game_summary = [row for row in summary if row["game"] == game]
        adv_corr = corr(
            [float(row["elo"]) for row in game_summary],
            [float(row["adv"]) for row in game_summary],
        )
        base_corr = corr(
            [float(row["elo"]) for row in game_summary],
            [float(row["base"]) for row in game_summary],
        )
        comp_adv_corr = corr(
            [float(row["comp"]) for row in game_rows],
            [float(row["advu"]) for row in game_rows],
        )
        comp_base_corr = corr(
            [float(row["comp"]) for row in game_rows],
            [float(row["baseu"]) for row in game_rows],
        )
        if math.isnan(adv_corr):
            lines.append("  Elo trend: not estimable yet.")
        else:
            lines.append(
                f"  Elo trend: corr(Elo, adversary utility)={adv_corr:.3f}; "
                f"corr(Elo, baseline utility)={base_corr:.3f}."
            )
        if math.isnan(comp_adv_corr):
            lines.append("  Competition trend: not estimable yet.")
        else:
            lines.append(
                f"  Competition trend: corr(competition, adversary utility)={comp_adv_corr:.3f}; "
                f"corr(competition, baseline utility)={comp_base_corr:.3f}."
            )
        for row in game_summary:
            lines.append(
                f"  {row['short']} Elo {row['elo']}: n={row['n']}, "
                f"advU={row['adv']:.2f}, baseU={row['base']:.2f}, "
                f"consensus={row['cons']:.2f}, round={row['round']:.2f}"
            )

    lines.extend(
        [
            "",
            f"Parse/utility mapping errors: {len(parse_errors)}",
            f"Log error/warning matches: {log_info['log_error_match_count']}",
            f"Provider failure report files: {len(log_info['provider_reports'])}",
        ]
    )
    if parse_errors:
        lines.extend(f"  {err}" for err in parse_errors[:10])
    if log_info["log_error_sample"]:
        lines.append("Log warning sample:")
        lines.extend(f"  {line}" for line in log_info["log_error_sample"][:10])
    if plots:
        lines.append("Plots:")
        lines.extend(f"  {path}" for path in plots)

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "partial_report.txt").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "last_updated.txt").write_text(now + "\n", encoding="utf-8")
    (OUTPUT_DIR / "scheduler_status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "log_health.json").write_text(
        json.dumps(log_info, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elo_by_model = parse_elo_markdown()
    manifest_rows = read_manifests()
    status = scheduler_status(manifest_rows)
    rows, missing, parse_errors = collect_result_rows(elo_by_model)
    summary = summarize_models(rows, elo_by_model)
    write_csvs(rows, summary)
    plots = [
        path
        for path in [
            make_elo_plot(summary, "adv", "Adversary utility", "partial_adversary_utility_vs_elo_by_game.png"),
            make_elo_plot(summary, "base", "Llama 3.3 baseline utility", "partial_baseline_utility_vs_elo_by_game.png"),
            make_competition_plot(rows),
        ]
        if path is not None
    ]
    log_info = scan_logs()
    print(write_report(status, rows, missing, parse_errors, summary, plots, log_info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
