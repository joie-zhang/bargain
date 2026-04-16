#!/usr/bin/env python3
"""Replot Game 1 (n=2) figures from CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def short_label(name: str) -> str:
    value = str(name)
    replacements = {
        "claude-opus-4-6-thinking": "Opus 4.6 Think",
        "claude-opus-4-6": "Opus 4.6",
        "claude-sonnet-4-20250514": "Sonnet 4",
        "claude-haiku-4-5-20251001": "Haiku 4.5",
        "gpt-5.4-high": "GPT-5.4",
        "gpt-5.2-chat-latest-20260210": "GPT-5.2",
        "gpt-5-nano-high": "GPT-5-nano",
        "gemini-3-pro": "Gemini 3 Pro",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "deepseek-r1-0528": "DeepSeek R1-0528",
        "deepseek-r1": "DeepSeek R1",
        "deepseek-v3": "DeepSeek V3",
        "qwen3-max-preview": "Qwen3 Max",
        "qwen2.5-72b-instruct": "Qwen2.5 72B",
        "llama-3.3-70b-instruct": "Llama 3.3 70B",
        "llama-3.1-8b-instruct": "Llama 3.1 8B",
        "llama-3.2-3b-instruct": "Llama 3.2 3B",
        "llama-3.2-1b-instruct": "Llama 3.2 1B",
        "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
        "gpt-4o-mini-2024-07-18": "GPT-4o mini",
        "gpt-4o-2024-05-13": "GPT-4o",
        "amazon-nova-micro-v1.0": "Nova Micro",
        "amazon-nova-pro-v1.0": "Nova Pro",
    }
    return replacements.get(value, value)


def plot_overall(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "average_utility_vs_elo.csv").sort_values("elo")
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in df.iterrows():
        x = float(row["elo"])
        y = float(row["avg_utility"])
        ax.scatter(x, y, s=110, color="#2563eb", alpha=0.9)
        ax.annotate(short_label(row["model"]), (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title("Game 1: Mean Adversary Utility vs Chatbot Arena Elo\nAverages include competition level, speaking order, and discussion-turn setting.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "average_utility_vs_elo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_comp(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "average_utility_vs_elo_by_competition_level.csv")
    df = df.sort_values(["competition_level", "elo"])
    comps = sorted(df["competition_level"].unique().tolist())
    colors = plt.cm.viridis([i / max(len(comps) - 1, 1) for i in range(len(comps))])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for color, comp in zip(colors, comps):
        subset = df[df["competition_level"] == comp].sort_values("elo")
        ax.plot(
            subset["elo"],
            subset["avg_utility"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=color,
            alpha=0.96,
            label=f"competition={comp:g}",
        )
    ax.set_title("Game 1: Mean Adversary Utility vs Chatbot Arena Elo\nEach curve fixes one competition level; averages include speaking order and discussion-turn setting.")
    ax.set_xlabel("Chatbot Arena Elo")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Competition")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "average_utility_vs_elo_by_competition_level.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rounds(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "average_rounds_to_consensus_vs_elo.csv").sort_values("elo")
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in df.iterrows():
        x = float(row["elo"])
        y = float(row["avg_rounds_to_consensus"])
        ax.scatter(x, y, s=110, color="#2563eb", alpha=0.9)
        ax.annotate(short_label(row["model"]), (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title("Game 1: Mean Rounds to Consensus vs Chatbot Arena Elo\nAverages include competition level, speaking order, and discussion-turn setting.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Rounds to Consensus")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "average_rounds_to_consensus_vs_elo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "average_utility_vs_competition_level_aggregated_over_models.csv").sort_values("competition_level")
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(df["competition_level"], df["avg_utility"], marker="o", markersize=8, linewidth=2.4, color="#2563eb")
    for _, row in df.iterrows():
        ax.annotate(f"{float(row['avg_utility']):.2f}", (float(row["competition_level"]), float(row["avg_utility"])), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("Game 1: Mean Adversary Utility vs Competition Level\nAggregated over all active adversary models, speaking orders, and discussion-turn settings.")
    ax.set_xlabel("Competition Level")
    ax.set_ylabel("Mean Adversary Utility")
    ax.set_xticks(df["competition_level"].tolist())
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "average_utility_vs_competition_level_aggregated_over_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(args.input_dir, args.output_dir)
    plot_by_comp(args.input_dir, args.output_dir)
    plot_rounds(args.input_dir, args.output_dir)
    plot_aggregate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
