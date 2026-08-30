#!/usr/bin/env python3
"""Compose the N=2 appendix panels from canonical analyzer outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "experiments/results/n2_baseline_comparison_analysis_20260505"
OUTPUT_DIR = ROOT / "analysis/recreated_figures"


def stack_images(inputs: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in inputs]
    try:
        width = max(image.width for image in images)
        resized = [
            image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
            if image.width != width
            else image.copy()
            for image in images
        ]
        canvas = Image.new("RGB", (width, sum(image.height for image in resized)), "white")
        y = 0
        for image in resized:
            canvas.paste(image, (0, y))
            y += image.height
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, optimize=True)
    finally:
        for image in images:
            image.close()


def main() -> None:
    payoff_output = OUTPUT_DIR / "01_02_adversary_payoff_combined.png"
    stack_images(
        [
            ANALYSIS_DIR / "gpt5_nano/01_adversary_payoff_overall.png",
            ANALYSIS_DIR / "gpt5_nano/02_adversary_payoff_by_competition.png",
        ],
        payoff_output,
    )

    order_output = OUTPUT_DIR / "bilateral_order_diagnostics_4x3.png"
    stack_images(
        [
            ANALYSIS_DIR / "gpt5_nano/05_adversary_payoff_by_order.png",
            ANALYSIS_DIR / "llama33/05_adversary_payoff_by_order.png",
            ANALYSIS_DIR / "gpt5_nano/06_adversary_payoff_by_order_and_competition.png",
            ANALYSIS_DIR / "llama33/06_adversary_payoff_by_order_and_competition.png",
        ],
        order_output,
    )

    print(payoff_output)
    print(order_output)


if __name__ == "__main__":
    main()
