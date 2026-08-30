#!/usr/bin/env python3
"""Test bilateral payoff trends against public capability metrics.

The script uses fixed local snapshots of public leaderboards. Model matches are
declared below so that aliases and configuration proxies remain reviewable.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis/capability_metric_robustness_20260816"
SOURCE_DIR = ANALYSIS_DIR / "sources"
PLOT_DIR = ANALYSIS_DIR / "plots"
PAPER_FIGURE_PATH = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/appendix/capability_benchmarks_over20.pdf"
)
BILATERAL_CSV = (
    ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-funding",
}
GAME_COLORS = {"game1": "#2b7bba", "game2": "#e63946", "game3": "#2ca02c"}

EXPECTED_ROSTER_SIZE = 30
EXPECTED_GAME_COUNTS = {"game1": 420, "game2": 540, "game3": 540}


# Values are source leaderboard model names. The optional third item identifies
# a nearby configuration rather than an exact configuration match.
MMLU_PRO_MATCHES = {
    "llama-3.2-1b-instruct": ("Llama-3.2-1B", "name_alias"),
    "llama-3.2-3b-instruct": ("Llama-3.2-3B", "name_alias"),
    "llama-3.1-8b-instruct": ("Llama-3.1-8B-Instruct", "exact"),
    "claude-3-haiku-20240307": ("Claude-3-Haiku-20240307", "exact"),
    "qwen2.5-72b-instruct": ("Qwen2.5-72B", "name_alias"),
    "gpt-4o-mini-2024-07-18": ("GPT-4o-mini", "exact"),
    "llama-3.3-70b-instruct": ("Llama-3.3-70B-Instruct", "exact"),
    "qwq-32b": ("QwQ-32B", "exact"),
    "gpt-4o-2024-05-13": ("GPT-4o (2024-05-13)", "exact"),
    "deepseek-v3": ("Deepseek-V3", "exact"),
    "gemma-3-27b-it": ("Gemma-3-27B-it", "exact"),
    "deepseek-r1": ("DeepSeek-R1", "exact"),
    "deepseek-r1-0528": ("DeepSeek-R1-0528", "exact"),
    "gemini-2.5-pro": ("Gemini-2.5-Pro", "exact"),
    "claude-opus-4-5-20251101-thinking-32k": (
        "Claude-4.5-Opus(Thinking)",
        "configuration_proxy",
    ),
    "gemini-3.1-pro": ("Gemini-3.1-Pro", "name_alias"),
    "claude-opus-4-6-thinking": ("Claude-4.6-Opus(Thinking)", "configuration_proxy"),
}

LIVEBENCH_MATCHES = {
    "llama-3.1-8b-instruct": ("meta-llama-3.1-8b-instruct-turbo", "provider_alias"),
    "amazon-nova-micro-v1.0": ("amazon.nova-micro-v1:0", "exact"),
    "command-r-plus-08-2024": ("command-r-plus-08-2024", "exact"),
    "amazon-nova-pro-v1.0": ("amazon.nova-pro-v1:0", "exact"),
    "qwen2.5-72b-instruct": ("qwen2.5-72b-instruct-turbo", "provider_alias"),
    "gpt-4o-mini-2024-07-18": ("gpt-4o-mini-2024-07-18", "exact"),
    "llama-3.3-70b-instruct": ("llama-3.3-70b-instruct-turbo", "provider_alias"),
    "qwq-32b": ("qwq-32b", "exact"),
    "deepseek-v3": ("deepseek-v3", "exact"),
    "deepseek-r1": ("deepseek-r1", "exact"),
    "gemma-3-27b-it": ("gemma-3-27b-it", "exact"),
}

HELM_MATCHES = {
    "llama-3.1-8b-instruct": ("Llama 3.1 Instruct Turbo (8B)", "provider_alias"),
    "amazon-nova-micro-v1.0": ("Amazon Nova Micro", "exact"),
    "claude-3-haiku-20240307": ("Claude 3 Haiku (20240307)", "exact"),
    "command-r-plus-08-2024": ("Command R Plus", "name_alias"),
    "amazon-nova-pro-v1.0": ("Amazon Nova Pro", "exact"),
    "qwen2.5-72b-instruct": ("Qwen2.5 Instruct Turbo (72B)", "provider_alias"),
    "gpt-4o-mini-2024-07-18": ("GPT-4o mini (2024-07-18)", "exact"),
    "llama-3.3-70b-instruct": ("Llama 3.3 Instruct Turbo (70B)", "provider_alias"),
    "gpt-4o-2024-05-13": ("GPT-4o (2024-05-13)", "exact"),
    "deepseek-v3": ("DeepSeek v3", "exact"),
}

AA_MATCHES = {
    "llama-3.2-1b-instruct": ("llama-3-2-instruct-1b", "exact"),
    "llama-3.2-3b-instruct": ("llama-3-2-instruct-3b", "exact"),
    "llama-3.1-8b-instruct": ("llama-3-1-instruct-8b", "exact"),
    "amazon-nova-micro-v1.0": ("nova-micro", "exact"),
    "claude-3-haiku-20240307": ("claude-3-haiku", "exact"),
    "amazon-nova-pro-v1.0": ("nova-pro", "exact"),
    "qwen2.5-72b-instruct": ("qwen2-5-72b-instruct", "exact"),
    "gpt-4o-mini-2024-07-18": ("gpt-4o-mini", "exact"),
    "llama-3.3-70b-instruct": ("llama-3-3-instruct-70b", "exact"),
    "gpt-4.1-nano-2025-04-14": ("gpt-4-1-nano", "exact"),
    "qwq-32b": ("qwq-32b", "exact"),
    "gpt-5-nano-high": ("gpt-5-nano", "exact"),
    "gpt-4o-2024-05-13": ("gpt-4o-2024-05-13", "exact"),
    "deepseek-v3": ("deepseek-v3", "exact"),
    "o3-mini-high": ("o3-mini", "configuration_proxy"),
    "gemma-3-27b-it": ("gemma-3-27b", "exact"),
    "claude-sonnet-4-20250514": ("claude-4-sonnet", "exact"),
    "deepseek-r1": ("deepseek-r1-0120", "exact"),
    "claude-haiku-4-5-20251001": ("claude-4-5-haiku", "exact"),
    "deepseek-r1-0528": ("deepseek-r1", "exact"),
    "qwen3-max-preview": ("qwen3-max-preview", "exact"),
    "gemini-2.5-pro": ("gemini-2-5-pro", "exact"),
    "claude-opus-4-5-20251101": ("claude-opus-4-5", "exact"),
    "claude-opus-4-5-20251101-thinking-32k": (
        "claude-opus-4-5-thinking",
        "configuration_proxy",
    ),
    "gpt-5.2-chat-latest-20260210": ("gpt-5-2-non-reasoning", "configuration_proxy"),
    "gpt-5.4-high": ("gpt-5-4", "configuration_proxy"),
    "gemini-3.1-pro": ("gemini-3-1-pro-preview", "name_alias"),
    "claude-opus-4-6": ("claude-opus-4-6", "exact"),
    "claude-opus-4-6-thinking": ("claude-opus-4-6-adaptive", "configuration_proxy"),
}

OPEN_LLM_MATCHES = {
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwq-32b": "Qwen/QwQ-32B",
}

HLE_MATCHES = {
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "claude-opus-4-5-20251101-thinking-32k": "claude-opus-4-5-20251101-thinking",
    "claude-opus-4-6": "claude-opus-4-6 (Non-Thinking)",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking-max",
    "gemini-3.1-pro": "gemini-3.1-pro-preview (thinking high)",
}


BENCHMARK_META = {
    "arena_elo": {
        "label": "LM Arena Elo",
        "x_label": "LM Arena Elo",
        "source_url": "https://lmarena.ai/leaderboard",
        "score_kind": "human-preference rating",
    },
    "mmlu_pro": {
        "label": "MMLU-Pro",
        "x_label": "MMLU-Pro accuracy (%)",
        "source_url": "https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro",
        "score_kind": "academic benchmark",
    },
    "livebench": {
        "label": "LiveBench overall",
        "x_label": "LiveBench overall score",
        "source_url": "https://github.com/LiveBench/livebench.github.io/blob/main/public/table_2024_11_25.csv",
        "score_kind": "academic benchmark suite",
    },
    "helm_lite": {
        "label": "HELM Lite mean win rate",
        "x_label": "HELM Lite mean win rate (%)",
        "source_url": "https://crfm.stanford.edu/helm/lite/latest/",
        "score_kind": "academic benchmark suite",
    },
    "mmlu_helm": {
        "label": "MMLU via HELM Lite",
        "x_label": "MMLU exact match (%)",
        "source_url": "https://crfm.stanford.edu/helm/lite/latest/",
        "score_kind": "academic benchmark",
    },
    "aa_index": {
        "label": "Artificial Analysis Intelligence Index v4.1.1",
        "x_label": "Artificial Analysis Intelligence Index",
        "source_url": "https://artificialanalysis.ai/leaderboards/models",
        "score_kind": "industry composite; includes estimated scores",
    },
}


AA_COMPONENT_META = {
    "aa_gpqa": {
        "field": "gpqa",
        "label": "GPQA Diamond (AA evaluation)",
        "x_label": "GPQA Diamond accuracy (%)",
        "scale": 100.0,
        "group": "reasoning and knowledge",
    },
    "aa_hle": {
        "field": "hle",
        "label": "Humanity's Last Exam (AA evaluation)",
        "x_label": "HLE accuracy (%)",
        "scale": 100.0,
        "group": "reasoning and knowledge",
    },
    "aa_scicode": {
        "field": "scicode",
        "label": "SciCode",
        "x_label": "SciCode score (%)",
        "scale": 100.0,
        "group": "scientific coding",
    },
    "aa_ifbench": {
        "field": "ifbench",
        "label": "IFBench (retired from AA Index)",
        "x_label": "IFBench score (%)",
        "scale": 100.0,
        "group": "instruction following",
    },
    "aa_tau2": {
        "field": "tau2",
        "label": "tau2-bench (superseded)",
        "x_label": "tau2-bench score (%)",
        "scale": 100.0,
        "group": "agentic tool use",
    },
    "aa_lcr": {
        "field": "lcr",
        "label": "AA-LCR",
        "x_label": "Long-context reasoning score (%)",
        "scale": 100.0,
        "group": "long-context reasoning",
    },
    "aa_terminalbench_hard": {
        "field": "terminalbenchHard",
        "label": "Terminal-Bench Hard (superseded)",
        "x_label": "Terminal-Bench Hard score (%)",
        "scale": 100.0,
        "group": "agentic terminal use",
    },
    "aa_omniscience": {
        "field": "omniscience",
        "label": "AA-Omniscience",
        "x_label": "AA-Omniscience score",
        "scale": 1.0,
        "group": "knowledge and hallucination",
    },
    "aa_critpt": {
        "field": "critpt",
        "label": "CritPt",
        "x_label": "CritPt score (%)",
        "scale": 100.0,
        "group": "physics reasoning",
    },
    "aa_mmmu_pro": {
        "field": "mmmuPro",
        "label": "MMMU-Pro",
        "x_label": "MMMU-Pro score (%)",
        "scale": 100.0,
        "group": "multimodal reasoning",
    },
    "aa_terminalbench_v21": {
        "field": "terminalbenchV21",
        "label": "Terminal-Bench 2.1",
        "x_label": "Terminal-Bench 2.1 score (%)",
        "scale": 100.0,
        "group": "agentic terminal use",
    },
    "aa_gdpval": {
        "field": "gdpvalNormalized",
        "label": "GDPval-AA v2",
        "x_label": "GDPval-AA normalized score (%)",
        "scale": 100.0,
        "group": "real-world agentic work",
    },
}

for _benchmark_id, _meta in AA_COMPONENT_META.items():
    BENCHMARK_META[_benchmark_id] = {
        "label": _meta["label"],
        "x_label": _meta["x_label"],
        "source_url": "https://artificialanalysis.ai/leaderboards/models",
        "score_kind": _meta["group"],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bilateral() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(BILATERAL_CSV)
    df = df[df["baseline_key"].eq("gpt5_nano")].copy()
    counts = df.groupby("game_id").size().to_dict()
    if counts != EXPECTED_GAME_COUNTS:
        raise RuntimeError(f"Unexpected bilateral game counts: {counts}")
    roster = (
        df[["adversary_model", "adversary_short", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")
    )
    if len(roster) != EXPECTED_ROSTER_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_ROSTER_SIZE} models, found {len(roster)}")
    means = (
        df.groupby(
            ["game_id", "adversary_model", "adversary_short", "adversary_elo"],
            as_index=False,
        )
        .agg(
            payoff=("adversary_utility", "mean"),
            payoff_sem=("adversary_utility", "sem"),
            runs=("adversary_utility", "size"),
        )
    )
    return roster, means


def rows_from_matches(
    benchmark_id: str,
    matches: dict[str, tuple[str, str]],
    source_scores: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    missing = []
    for model, (source_model, match_quality) in matches.items():
        if source_model not in source_scores:
            missing.append(source_model)
            continue
        source = source_scores[source_model]
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "benchmark_label": BENCHMARK_META[benchmark_id]["label"],
                "adversary_model": model,
                "source_model": source_model,
                "score": float(source["score"]),
                "estimated": bool(source.get("estimated", False)),
                "score_source": source.get("score_source", ""),
                "match_quality": match_quality,
                "source_url": BENCHMARK_META[benchmark_id]["source_url"],
            }
        )
    if missing:
        raise RuntimeError(f"Missing {benchmark_id} source models: {missing}")
    return rows


def load_mmlu_pro() -> list[dict[str, object]]:
    path = SOURCE_DIR / "mmlu_pro_space_config_20260816.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    components = [c for c in config["components"] if c.get("type") == "dataframe"]
    if len(components) != 1:
        raise RuntimeError(f"Expected one MMLU-Pro dataframe, found {len(components)}")
    data = components[0]["props"]["value"]["data"]
    source_scores = {
        str(row[0]): {"score": float(row[3]) * 100, "score_source": str(row[2])}
        for row in data
        if row[3] not in {None, "-"}
    }
    return rows_from_matches("mmlu_pro", MMLU_PRO_MATCHES, source_scores)


def load_livebench() -> list[dict[str, object]]:
    table = pd.read_csv(SOURCE_DIR / "livebench_table_2024_11_25.csv")
    categories = json.loads(
        (SOURCE_DIR / "livebench_categories_2024_11_25.json").read_text(encoding="utf-8")
    )
    category_scores = []
    for category, tasks in categories.items():
        missing = sorted(set(tasks) - set(table.columns))
        if missing:
            raise RuntimeError(f"LiveBench category {category} is missing tasks: {missing}")
        category_scores.append(table[tasks].apply(pd.to_numeric, errors="coerce").mean(axis=1))
    table["overall"] = pd.concat(category_scores, axis=1).mean(axis=1)
    source_scores = {
        str(row.model): {"score": float(row.overall), "score_source": "LiveBench"}
        for row in table.itertuples(index=False)
    }
    return rows_from_matches("livebench", LIVEBENCH_MATCHES, source_scores)


def load_helm() -> list[dict[str, object]]:
    data = json.loads(
        (SOURCE_DIR / "helm_lite_v1.13.0_core_scenarios.json").read_text(encoding="utf-8")
    )
    accuracy_tables = [table for table in data if table.get("title") == "Accuracy"]
    if len(accuracy_tables) != 1:
        raise RuntimeError(f"Expected one HELM accuracy table, found {len(accuracy_tables)}")
    table = accuracy_tables[0]
    headers = [str(item["value"]) for item in table["header"]]
    model_idx = headers.index("Model")
    mean_win_idx = headers.index("Mean win rate")
    mmlu_idx = headers.index("MMLU - EM")
    mean_win_scores = {}
    mmlu_scores = {}
    for row in table["rows"]:
        model = str(row[model_idx]["value"])
        mean_win_scores[model] = {
            "score": float(row[mean_win_idx]["value"]) * 100,
            "score_source": "HELM Lite v1.13.0",
        }
        mmlu_scores[model] = {
            "score": float(row[mmlu_idx]["value"]) * 100,
            "score_source": "HELM Lite v1.13.0",
        }
    rows = rows_from_matches("helm_lite", HELM_MATCHES, mean_win_scores)
    rows.extend(rows_from_matches("mmlu_helm", HELM_MATCHES, mmlu_scores))
    return rows


def load_artificial_analysis() -> list[dict[str, object]]:
    text = (SOURCE_DIR / "artificial_analysis_models_20260816.html").read_text(
        encoding="utf-8"
    )
    pattern = re.compile(
        r'\\"id\\":\\"[0-9a-f-]+\\",'
        r'\\"name\\":\\"([^\\]+)\\",'
        r'\\"shortName\\":\\"([^\\]+)\\",'
        r'\\"slug\\":\\"([^\\]+)\\"'
        r'(.{0,2500}?)'
        r'\\"intelligenceIndex\\":(-?[0-9.]+|null),'
        r'\\"intelligenceIndexIsEstimated\\":(true|false)',
        re.S,
    )
    source_scores = {}
    for match in pattern.finditer(text):
        name, _, slug, _, score, estimated = match.groups()
        if score == "null":
            continue
        source_scores[slug] = {
            "score": float(score),
            "estimated": estimated == "true",
            "score_source": name,
        }
    if len(source_scores) < 500:
        raise RuntimeError(f"Parsed only {len(source_scores)} Artificial Analysis models")
    return rows_from_matches("aa_index", AA_MATCHES, source_scores)


def load_artificial_analysis_components() -> list[dict[str, object]]:
    """Load individual benchmark results embedded in the AA model table."""
    text = (SOURCE_DIR / "artificial_analysis_models_20260816.html").read_text(
        encoding="utf-8"
    )
    rows = []
    for adversary_model, (slug, match_quality) in AA_MATCHES.items():
        needle = r'\"slug\":\"' + slug + r'\"'
        positions = [match.start() for match in re.finditer(re.escape(needle), text)]
        starts = [
            position
            for position in positions
            if 0 < text.find(r'\"intelligenceIndex\"', position) - position < 2000
        ]
        if len(starts) != 1:
            raise RuntimeError(
                f"Expected one detailed Artificial Analysis object for {slug}, found {starts}"
            )
        block = text[starts[0] : starts[0] + 4000]
        for benchmark_id, meta in AA_COMPONENT_META.items():
            value_match = re.search(
                re.escape(r'\"' + str(meta["field"]) + r'\":')
                + r'(-?[0-9.]+|null|\"\$undefined\")',
                block,
            )
            if value_match is None:
                raise RuntimeError(
                    f"Missing field {meta['field']} in Artificial Analysis object {slug}"
                )
            raw_value = value_match.group(1)
            if raw_value in {"null", r'\"$undefined\"'}:
                continue
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_label": BENCHMARK_META[benchmark_id]["label"],
                    "adversary_model": adversary_model,
                    "source_model": slug,
                    "score": float(raw_value) * float(meta["scale"]),
                    "estimated": False,
                    "score_source": "Artificial Analysis published evaluation result",
                    "match_quality": match_quality,
                    "source_url": BENCHMARK_META[benchmark_id]["source_url"],
                }
            )
    return rows


def load_arena_scores(roster: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "benchmark_id": "arena_elo",
            "benchmark_label": BENCHMARK_META["arena_elo"]["label"],
            "adversary_model": row.adversary_model,
            "source_model": row.adversary_model,
            "score": float(row.adversary_elo),
            "estimated": False,
            "score_source": "Paper roster Arena snapshot",
            "match_quality": "exact",
            "source_url": BENCHMARK_META["arena_elo"]["source_url"],
        }
        for row in roster.itertuples(index=False)
    ]


def ols_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = stats.linregress(x, y)
    dof = len(x) - 2
    critical = stats.t.ppf(0.975, dof)
    return {
        "slope": float(fit.slope),
        "slope_ci_low": float(fit.slope - critical * fit.stderr),
        "slope_ci_high": float(fit.slope + critical * fit.stderr),
        "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def calculate_trends(scores: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark_id, score_df in scores.groupby("benchmark_id", sort=False):
        score_df = score_df.copy()
        score_df["score_z"] = stats.zscore(score_df["score"], ddof=1)
        elo_rho, elo_p = stats.spearmanr(score_df["score"], score_df["adversary_elo"])
        for game_id in GAME_ORDER:
            merged = means[means["game_id"].eq(game_id)].merge(
                score_df[["adversary_model", "score", "score_z"]],
                on="adversary_model",
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(score_df):
                raise RuntimeError(
                    f"{benchmark_id}/{game_id}: matched {len(merged)} of {len(score_df)} models"
                )
            raw_fit = ols_summary(merged["score"].to_numpy(), merged["payoff"].to_numpy())
            z_fit = ols_summary(merged["score_z"].to_numpy(), merged["payoff"].to_numpy())
            elo_z = stats.zscore(merged["adversary_elo"], ddof=1)
            elo_fit = ols_summary(elo_z, merged["payoff"].to_numpy())
            payoff_rho, payoff_rho_p = stats.spearmanr(merged["score"], merged["payoff"])
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_label": BENCHMARK_META[benchmark_id]["label"],
                    "game_id": game_id,
                    "game_label": GAME_LABELS[game_id],
                    "models": len(merged),
                    "direct_scores": int((~score_df["estimated"]).sum()),
                    "estimated_scores": int(score_df["estimated"].sum()),
                    "score_vs_arena_spearman_rho": float(elo_rho),
                    "score_vs_arena_spearman_p": float(elo_p),
                    "payoff_vs_score_spearman_rho": float(payoff_rho),
                    "payoff_vs_score_spearman_p": float(payoff_rho_p),
                    "payoff_per_score_unit": raw_fit["slope"],
                    "payoff_per_score_unit_ci_low": raw_fit["slope_ci_low"],
                    "payoff_per_score_unit_ci_high": raw_fit["slope_ci_high"],
                    "payoff_per_score_sd": z_fit["slope"],
                    "payoff_per_score_sd_ci_low": z_fit["slope_ci_low"],
                    "payoff_per_score_sd_ci_high": z_fit["slope_ci_high"],
                    "payoff_vs_score_r_squared": z_fit["r_squared"],
                    "payoff_vs_score_p": z_fit["p_value"],
                    "same_subset_payoff_per_arena_sd": elo_fit["slope"],
                    "same_subset_arena_ci_low": elo_fit["slope_ci_low"],
                    "same_subset_arena_ci_high": elo_fit["slope_ci_high"],
                    "same_subset_arena_r_squared": elo_fit["r_squared"],
                }
            )
    out = pd.DataFrame(rows)
    out["payoff_vs_score_p_bh_12_academic"] = np.nan
    academic = out["benchmark_id"].ne("aa_index")
    out.loc[academic, "payoff_vs_score_p_bh_12_academic"] = benjamini_hochberg(
        out.loc[academic, "payoff_vs_score_p"]
    )
    return out


def plot_one(
    benchmark_id: str,
    score_df: pd.DataFrame,
    means: pd.DataFrame,
    out_path: Path,
    *,
    compact: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0 if not compact else 5.2, 4.35), dpi=300)
    for game_id in GAME_ORDER:
        merged = means[means["game_id"].eq(game_id)].merge(
            score_df,
            on="adversary_model",
            how="inner",
            validate="one_to_one",
        )
        color = GAME_COLORS[game_id]
        direct = merged[~merged["estimated"]]
        estimated = merged[merged["estimated"]]
        if not direct.empty:
            ax.errorbar(
                direct["score"],
                direct["payoff"],
                yerr=direct["payoff_sem"],
                fmt="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.8,
                alpha=0.72,
                zorder=3,
            )
        if not estimated.empty:
            ax.errorbar(
                estimated["score"],
                estimated["payoff"],
                yerr=estimated["payoff_sem"],
                fmt="x",
                markersize=4.5,
                markeredgewidth=1.0,
                color=color,
                ecolor=color,
                elinewidth=0.7,
                capsize=1.5,
                alpha=0.62,
                zorder=2,
            )
        fit = stats.linregress(merged["score"], merged["payoff"])
        xs = np.linspace(float(merged["score"].min()), float(merged["score"].max()), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, "--", color=color, linewidth=2.2)

    estimated_count = int(score_df["estimated"].sum())
    title = f"{BENCHMARK_META[benchmark_id]['label']} (n={len(score_df)})"
    if estimated_count:
        title += f"\n{estimated_count} x-axis values are marked as estimates by Artificial Analysis"
    ax.set_title(title, fontsize=12.5, pad=8)
    ax.set_xlabel(BENCHMARK_META[benchmark_id]["x_label"], fontsize=11.5)
    ax.set_ylabel("Adversary payoff", fontsize=11.5)
    ax.set_ylim(-10, 102)
    ax.grid(True, color="#d1d5db", alpha=0.52, linewidth=0.75)
    ax.tick_params(axis="both", labelsize=9.5)
    handles = [
        Line2D([0], [0], color=GAME_COLORS[g], linestyle="--", linewidth=2.2, label=GAME_LABELS[g])
        for g in GAME_ORDER
    ]
    if estimated_count:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    marker="o",
                    markerfacecolor="white",
                    linestyle="None",
                    label="Circle: direct index score",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    marker="x",
                    linestyle="None",
                    label="Cross: AA-estimated index score",
                ),
            ]
        )
    ax.legend(handles=handles, fontsize=8.0, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_combined(scores: pd.DataFrame, means: pd.DataFrame) -> None:
    order = ["mmlu_pro", "livebench", "helm_lite", "mmlu_helm"]
    fig = plt.figure(figsize=(12.0, 8.0), dpi=250)
    for idx, benchmark_id in enumerate(order):
        ax = fig.add_subplot(2, 2, idx + 1)
        score_df = scores[scores["benchmark_id"].eq(benchmark_id)]
        for game_id in GAME_ORDER:
            merged = means[means["game_id"].eq(game_id)].merge(
                score_df, on="adversary_model", how="inner", validate="one_to_one"
            )
            color = GAME_COLORS[game_id]
            direct = merged[~merged["estimated"]]
            if not direct.empty:
                ax.errorbar(
                    direct["score"], direct["payoff"], yerr=direct["payoff_sem"],
                    fmt="o", markersize=4.0, markerfacecolor="white",
                    markeredgewidth=0.9, color=color, ecolor=color,
                    elinewidth=0.7, capsize=1.5, alpha=0.72, zorder=3,
                )
            fit = stats.linregress(merged["score"], merged["payoff"])
            xs = np.linspace(float(merged["score"].min()), float(merged["score"].max()), 100)
            ax.plot(xs, fit.intercept + fit.slope * xs, "--", color=color, linewidth=1.8)
        title = f"{BENCHMARK_META[benchmark_id]['label']} (n={len(score_df)})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(BENCHMARK_META[benchmark_id]["x_label"], fontsize=9.5)
        ax.set_ylabel("Adversary payoff", fontsize=9.5)
        ax.set_ylim(-10, 102)
        ax.grid(True, color="#d1d5db", alpha=0.5, linewidth=0.65)
        ax.tick_params(labelsize=8.5)
    handles = [
        Line2D([0], [0], color=GAME_COLORS[g], linestyle="--", linewidth=2.2, label=GAME_LABELS[g])
        for g in GAME_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(PLOT_DIR / "bilateral_payoff_by_alternative_capability_metrics.png", dpi=300, facecolor="white")
    plt.close(fig)


BROAD_PLOT_ORDER = [
    "arena_elo",
    "mmlu_pro",
    "aa_index",
    "aa_gpqa",
    "aa_hle",
    "livebench",
    "helm_lite",
    "mmlu_helm",
    "aa_scicode",
    "aa_ifbench",
    "aa_tau2",
    "aa_lcr",
    "aa_terminalbench_hard",
    "aa_omniscience",
    "aa_critpt",
    "aa_mmmu_pro",
    "aa_terminalbench_v21",
    "aa_gdpval",
]

HIGH_COVERAGE_PLOT_ORDER = [
    "arena_elo",
    "aa_index",
    "aa_gpqa",
    "aa_hle",
    "aa_scicode",
    "aa_ifbench",
    "aa_tau2",
    "aa_lcr",
    "aa_terminalbench_hard",
    "aa_omniscience",
    "aa_critpt",
]


def plot_broad_combined(scores: pd.DataFrame, means: pd.DataFrame) -> None:
    eligible = scores.groupby("benchmark_id")["adversary_model"].nunique()
    missing = [
        benchmark_id
        for benchmark_id in HIGH_COVERAGE_PLOT_ORDER
        if eligible.get(benchmark_id, 0) <= 20
    ]
    if missing:
        raise RuntimeError(f"Filtered plot benchmarks at or below 20 models: {missing}")

    fig = plt.figure(figsize=(16.0, 12.5), dpi=220)
    for idx, benchmark_id in enumerate(HIGH_COVERAGE_PLOT_ORDER):
        ax = fig.add_subplot(3, 4, idx + 1)
        score_df = scores[scores["benchmark_id"].eq(benchmark_id)]
        for game_id in GAME_ORDER:
            merged = means[means["game_id"].eq(game_id)].merge(
                score_df,
                on="adversary_model",
                how="inner",
                validate="one_to_one",
            )
            color = GAME_COLORS[game_id]
            direct = merged[~merged["estimated"]]
            estimated = merged[merged["estimated"]]
            if not direct.empty:
                ax.errorbar(
                    direct["score"],
                    direct["payoff"],
                    yerr=direct["payoff_sem"],
                    fmt="o",
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.75,
                    color=color,
                    ecolor=color,
                    elinewidth=0.55,
                    capsize=1.1,
                    alpha=0.62,
                    zorder=3,
                )
            if not estimated.empty:
                ax.errorbar(
                    estimated["score"],
                    estimated["payoff"],
                    yerr=estimated["payoff_sem"],
                    fmt="x",
                    markersize=3.2,
                    markeredgewidth=0.75,
                    color=color,
                    ecolor=color,
                    elinewidth=0.5,
                    capsize=1.0,
                    alpha=0.54,
                    zorder=2,
                )
            fit = stats.linregress(merged["score"], merged["payoff"])
            xs = np.linspace(
                float(merged["score"].min()),
                float(merged["score"].max()),
                100,
            )
            ax.plot(
                xs,
                fit.intercept + fit.slope * xs,
                "--",
                color=color,
                linewidth=1.45,
            )
        estimated_count = int(score_df["estimated"].sum())
        title = f"{BENCHMARK_META[benchmark_id]['label']} (n={len(score_df)})"
        if estimated_count:
            title += f"\n{estimated_count} index estimates"
        ax.set_title(title, fontsize=9.4, pad=5)
        ax.set_xlabel(BENCHMARK_META[benchmark_id]["x_label"], fontsize=7.9)
        ax.set_ylabel("Adversary payoff", fontsize=7.9)
        ax.set_ylim(-10, 102)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7.2)

    legend_ax = fig.add_subplot(3, 4, 12)
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=GAME_COLORS[game_id],
            linestyle="--",
            linewidth=2.0,
            label=GAME_LABELS[game_id],
        )
        for game_id in GAME_ORDER
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="o",
                markerfacecolor="white",
                linestyle="None",
                label="Published score",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="x",
                linestyle="None",
                label="AA index estimate",
            ),
        ]
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=9.2,
    )
    legend_ax.text(
        0.02,
        0.42,
        "Each point is one model-level mean.\n"
        "Error bars are payoff SEM.\n"
        "Fits are unweighted model-level OLS.\n"
        "Only the AA composite has estimated x values.\n"
        "Configuration matches are recorded in the score table.",
        va="top",
        ha="left",
        fontsize=8.7,
        linespacing=1.4,
    )
    fig.suptitle(
        "Bilateral adversary payoff across LM Arena and Artificial Analysis capability measures",
        fontsize=17,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.991), h_pad=2.0, w_pad=1.5)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            PLOT_DIR / f"bilateral_payoff_capability_benchmarks_over20.{suffix}",
            dpi=300 if suffix == "png" else None,
            facecolor="white",
        )
    PAPER_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURE_PATH, facecolor="white")
    plt.close(fig)


def broad_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        scores.groupby(["benchmark_id", "benchmark_label"], as_index=False)
        .agg(
            matched_models=("adversary_model", "nunique"),
            estimated_scores=("estimated", "sum"),
            exact_name_or_alias=(
                "match_quality",
                lambda values: int((values != "configuration_proxy").sum()),
            ),
            configuration_proxies=(
                "match_quality",
                lambda values: int((values == "configuration_proxy").sum()),
            ),
        )
    )
    coverage["roster_models"] = EXPECTED_ROSTER_SIZE
    coverage["coverage_fraction"] = coverage["matched_models"] / EXPECTED_ROSTER_SIZE
    coverage["plotted"] = coverage["matched_models"].ge(10)
    order = {benchmark_id: idx for idx, benchmark_id in enumerate(BROAD_PLOT_ORDER)}
    coverage["plot_order"] = coverage["benchmark_id"].map(order)
    return coverage.sort_values(["plot_order", "benchmark_id"], na_position="last")


def candidate_coverage() -> pd.DataFrame:
    """Audit candidate metrics, including candidates below the plot threshold."""
    hle_text = (SOURCE_DIR / "hle_scale_leaderboard_20260816.html").read_text(
        encoding="utf-8"
    )
    missing_hle = [name for name in HLE_MATCHES.values() if name not in hle_text]
    if missing_hle:
        raise RuntimeError(f"Missing HLE source models: {missing_hle}")

    gpqa_rows = json.loads(
        (SOURCE_DIR / "gpqa_hf_leaderboard_20260816.json").read_text(encoding="utf-8")
    )
    gpqa_ids = {str(row["modelId"]) for row in gpqa_rows}
    gpqa_exact = {
        "meta-llama/Llama-3.2-1B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-0528",
    }
    if not gpqa_exact.issubset(gpqa_ids):
        raise RuntimeError(f"Missing GPQA source models: {sorted(gpqa_exact - gpqa_ids)}")

    open_llm = pd.read_parquet(
        SOURCE_DIR / "open_llm_leaderboard_contents_20260816.parquet"
    )
    open_llm_names = set(open_llm["fullname"].astype(str))
    missing_open_llm = [name for name in OPEN_LLM_MATCHES.values() if name not in open_llm_names]
    if missing_open_llm:
        raise RuntimeError(f"Missing Open LLM source models: {missing_open_llm}")

    rows = [
        ("arena_elo", "LM Arena Elo", 30, 30, 0, 30, 0, "Reference metric used in Figure 2"),
        ("aa_index", "Artificial Analysis Intelligence Index v4.1.1", 29, 7, 22, 24, 5, "Diagnostic only; most scores are estimated"),
        ("mmlu_pro", "MMLU-Pro", 17, 17, 0, 15, 2, "Plot"),
        ("livebench", "LiveBench overall", 11, 11, 0, 11, 0, "Plot"),
        ("helm_lite", "HELM Lite mean win rate", 10, 10, 0, 10, 0, "Plot"),
        ("mmlu_helm", "MMLU via HELM Lite", 10, 10, 0, 10, 0, "Plot"),
        ("hle", "Humanity's Last Exam", len(HLE_MATCHES), len(HLE_MATCHES), 0, 4, 3, "Below threshold"),
        ("bbh", "BIG-Bench Hard via Open LLM Leaderboard 2", len(OPEN_LLM_MATCHES), len(OPEN_LLM_MATCHES), 0, 6, 0, "Below threshold"),
        ("gpqa", "GPQA via Open LLM Leaderboard 2", len(OPEN_LLM_MATCHES), len(OPEN_LLM_MATCHES), 0, 6, 0, "Below threshold; HF benchmark table has 4 exact matches"),
        ("open_llm_v2", "Open LLM Leaderboard 2 composite", len(OPEN_LLM_MATCHES), len(OPEN_LLM_MATCHES), 0, 6, 0, "Below threshold"),
    ]
    out = pd.DataFrame(
        rows,
        columns=[
            "benchmark_id", "benchmark_label", "published_matches", "direct_scores",
            "estimated_scores", "exact_name_or_alias", "configuration_proxies",
            "plot_status",
        ],
    )
    out["roster_models"] = EXPECTED_ROSTER_SIZE
    out["direct_coverage_fraction"] = out["direct_scores"] / EXPECTED_ROSTER_SIZE
    return out


def mmlu_pro_sensitivity(scores: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    mmlu = scores[scores["benchmark_id"].eq("mmlu_pro")].copy()
    subsets = {
        "all_matches": mmlu,
        "exclude_configuration_proxies": mmlu[
            mmlu["match_quality"].ne("configuration_proxy")
        ],
        "tiger_lab_scores_only": mmlu[mmlu["score_source"].eq("TIGER-Lab")],
    }
    rows = []
    for subset_name, score_df in subsets.items():
        for game_id in GAME_ORDER:
            merged = means[means["game_id"].eq(game_id)].merge(
                score_df[["adversary_model", "score"]],
                on="adversary_model",
                how="inner",
                validate="one_to_one",
            )
            score_z = stats.zscore(merged["score"], ddof=1)
            fit = ols_summary(score_z, merged["payoff"].to_numpy())
            rho, rho_p = stats.spearmanr(merged["score"], merged["payoff"])
            rows.append(
                {
                    "subset": subset_name,
                    "game_id": game_id,
                    "models": len(merged),
                    "payoff_per_score_sd": fit["slope"],
                    "ci_low": fit["slope_ci_low"],
                    "ci_high": fit["slope_ci_high"],
                    "p_value": fit["p_value"],
                    "spearman_rho": float(rho),
                    "spearman_p": float(rho_p),
                }
            )
    return pd.DataFrame(rows)


def source_manifest() -> list[dict[str, object]]:
    urls = {
        "mmlu_pro_space_config_20260816.json": BENCHMARK_META["mmlu_pro"]["source_url"],
        "livebench_table_2024_11_25.csv": BENCHMARK_META["livebench"]["source_url"],
        "livebench_categories_2024_11_25.json": "https://github.com/LiveBench/livebench.github.io/blob/main/public/categories_2024_11_25.json",
        "helm_lite_v1.13.0_core_scenarios.json": "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json",
        "artificial_analysis_models_20260816.html": BENCHMARK_META["aa_index"]["source_url"],
        "hle_scale_leaderboard_20260816.html": "https://labs.scale.com/leaderboard/humanitys_last_exam",
        "gpqa_hf_leaderboard_20260816.json": "https://huggingface.co/api/datasets/Idavidrein/gpqa/leaderboard",
        "open_llm_leaderboard_contents_20260816.parquet": "https://huggingface.co/datasets/open-llm-leaderboard/contents",
    }
    return [
        {
            "file": str((SOURCE_DIR / name).relative_to(ROOT)),
            "source_url": url,
            "accessed": "2026-08-16",
            "sha256": sha256(SOURCE_DIR / name),
            "bytes": (SOURCE_DIR / name).stat().st_size,
        }
        for name, url in urls.items()
    ]


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    roster, means = load_bilateral()
    rows = []
    rows.extend(load_mmlu_pro())
    rows.extend(load_livebench())
    rows.extend(load_helm())
    rows.extend(load_artificial_analysis())
    scores = pd.DataFrame(rows).merge(
        roster[["adversary_model", "adversary_short", "adversary_elo"]],
        on="adversary_model",
        how="left",
        validate="many_to_one",
    )
    if scores[["adversary_short", "adversary_elo"]].isna().any().any():
        raise RuntimeError("A benchmark match is outside the bilateral roster")
    scores = scores.sort_values(["benchmark_id", "score", "adversary_model"])
    scores.to_csv(ANALYSIS_DIR / "benchmark_scores_matched.csv", index=False)

    trends = calculate_trends(scores, means)
    trends.to_csv(ANALYSIS_DIR / "trend_summary.csv", index=False)
    mmlu_pro_sensitivity(scores, means).to_csv(
        ANALYSIS_DIR / "mmlu_pro_sensitivity.csv", index=False
    )

    coverage = (
        scores.groupby(["benchmark_id", "benchmark_label"], as_index=False)
        .agg(
            matched_models=("adversary_model", "nunique"),
            direct_scores=("estimated", lambda x: int((~x).sum())),
            estimated_scores=("estimated", "sum"),
            exact_name_or_alias=("match_quality", lambda x: int((x != "configuration_proxy").sum())),
            configuration_proxies=("match_quality", lambda x: int((x == "configuration_proxy").sum())),
        )
    )
    coverage["roster_models"] = EXPECTED_ROSTER_SIZE
    coverage["coverage_fraction"] = coverage["matched_models"] / EXPECTED_ROSTER_SIZE
    coverage.to_csv(ANALYSIS_DIR / "plotted_benchmark_coverage.csv", index=False)
    candidate_coverage().to_csv(
        ANALYSIS_DIR / "candidate_benchmark_coverage.csv", index=False
    )

    for benchmark_id, score_df in scores.groupby("benchmark_id", sort=False):
        plot_one(
            benchmark_id,
            score_df,
            means,
            PLOT_DIR / f"bilateral_payoff_vs_{benchmark_id}.png",
        )
    plot_combined(scores, means)

    broad_rows = []
    broad_rows.extend(load_artificial_analysis_components())
    broad_rows.extend(load_arena_scores(roster))
    broad_additions = pd.DataFrame(broad_rows).merge(
        roster[["adversary_model", "adversary_short", "adversary_elo"]],
        on="adversary_model",
        how="left",
        validate="many_to_one",
    )
    broad_scores = pd.concat([scores, broad_additions], ignore_index=True)
    if broad_scores.duplicated(["benchmark_id", "adversary_model"]).any():
        duplicates = broad_scores[
            broad_scores.duplicated(["benchmark_id", "adversary_model"], keep=False)
        ]
        raise RuntimeError(
            "Duplicate broad benchmark matches:\n"
            + duplicates[["benchmark_id", "adversary_model"]].to_string(index=False)
        )
    broad_scores = broad_scores.sort_values(
        ["benchmark_id", "score", "adversary_model"]
    )
    broad_scores.to_csv(
        ANALYSIS_DIR / "broad_benchmark_scores_matched.csv", index=False
    )
    broad_coverage_df = broad_coverage(broad_scores)
    broad_coverage_df.to_csv(
        ANALYSIS_DIR / "broad_benchmark_coverage.csv", index=False
    )
    broad_trends = calculate_trends(broad_scores, means).drop(
        columns=["payoff_vs_score_p_bh_12_academic"]
    )
    broad_trends["payoff_vs_score_p_bh_all"] = benjamini_hochberg(
        broad_trends["payoff_vs_score_p"]
    )
    broad_trends.to_csv(
        ANALYSIS_DIR / "broad_benchmark_trend_summary.csv", index=False
    )
    high_coverage_df = broad_coverage_df[
        broad_coverage_df["benchmark_id"].isin(HIGH_COVERAGE_PLOT_ORDER)
    ].copy()
    if len(high_coverage_df) != len(HIGH_COVERAGE_PLOT_ORDER):
        raise RuntimeError("High-coverage table does not contain every filtered benchmark")
    high_coverage_df.to_csv(
        ANALYSIS_DIR / "benchmark_coverage_over20.csv", index=False
    )
    high_trends = broad_trends[
        broad_trends["benchmark_id"].isin(HIGH_COVERAGE_PLOT_ORDER)
    ].drop(columns=["payoff_vs_score_p_bh_all"])
    high_trends["payoff_vs_score_p_bh_33_over20"] = benjamini_hochberg(
        high_trends["payoff_vs_score_p"]
    )
    high_trends.to_csv(
        ANALYSIS_DIR / "benchmark_trends_over20.csv", index=False
    )
    plot_broad_combined(broad_scores, means)
    broad_individual_dir = PLOT_DIR / "broad_individual"
    broad_individual_dir.mkdir(parents=True, exist_ok=True)
    for benchmark_id in BROAD_PLOT_ORDER:
        plot_one(
            benchmark_id,
            broad_scores[broad_scores["benchmark_id"].eq(benchmark_id)],
            means,
            broad_individual_dir / f"bilateral_payoff_vs_{benchmark_id}.png",
        )

    provenance = {
        "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        "bilateral_input": str(BILATERAL_CSV.relative_to(ROOT)),
        "bilateral_input_sha256": sha256(BILATERAL_CSV),
        "baseline_key": "gpt5_nano",
        "roster_size": len(roster),
        "game_counts": EXPECTED_GAME_COUNTS,
        "source_snapshots": source_manifest(),
        "paper_figure": str(PAPER_FIGURE_PATH.relative_to(ROOT)),
        "paper_figure_sha256": sha256(PAPER_FIGURE_PATH),
        "notes": [
            "All payoff fits use unweighted model-level mean payoffs, as in Figure 2.",
            "LiveBench overall is the mean of category means for release 2024-11-25.",
            "Artificial Analysis estimated scores remain marked and are not direct v4.1.1 evaluations.",
            "Individual Artificial Analysis evaluation fields are plotted when a numeric result is published; the source does not mark those component values as estimates.",
            "Configuration proxies are identified in benchmark_scores_matched.csv.",
            "The broad subplot figure includes every audited score set with at least 10 matched roster models.",
            "The paper subplot figure includes the 11 score sets with more than 20 matched roster models.",
        ],
    }
    (ANALYSIS_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(coverage.to_string(index=False))
    print("\nBroad benchmark coverage")
    print(broad_coverage_df.to_string(index=False))
    print(f"\nWrote outputs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
