#!/usr/bin/env python3
"""Create a three-panel slide about GPT-5.4 and Nano coalition dynamics."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/game1_gpt54_binding_team_v3_20260816_093310/runs"
OUTPUT_DIR = ROOT / "analysis/figure_recreation_20260816"
PNG_PATH = OUTPUT_DIR / "gpt54_nano_coalition_dynamics_slide.png"

PANEL_WIDTHS = (560, 600, 760)
WIDTH = sum(PANEL_WIDTHS)
HEIGHT = 789
HORIZONTAL_INSET = 14

FONT_REGULAR = "/usr/share/fonts/adobe-source-sans-pro-fonts/SourceSans3-Regular.otf"
FONT_BOLD = "/usr/share/fonts/adobe-source-sans-pro-fonts/SourceSans3-Bold.otf"

COLORS = {
    "white": "#FFFFFF",
    "navy": "#142843",
    "text": "#172033",
    "muted": "#5C687A",
    "rule": "#D8E0EA",
    "nano": "#2F6FED",
    "nano_dark": "#2056BF",
    "nano_bg": "#EDF4FF",
    "high": "#D64545",
    "high_dark": "#A92F36",
    "high_bg": "#FFF0F0",
    "success": "#16805B",
    "success_bg": "#E9F7EF",
    "concede": "#D96B25",
    "concede_dark": "#A94518",
    "concede_bg": "#FFF2E8",
}


PANELS = [
    {
        "n": "n = 2",
        "config": "RUN 20",
        "threshold": "2 of 2 votes needed\nfor agreement",
        "source_dir": "config_0020_game1_team_coordination_n2_comp_1p0_gpt_5p4_high_last_seed2",
        "bubbles": [
            {
                "agent": "NANO A1",
                "role": "INITIAL PRIORITY",
                "kind": "nano",
                "quote": "“Stone is my top priority (71).”",
                "fragments": ["Stone is my top priority (71)."],
            },
            {
                "agent": "GPT-5.4 A2",
                "role": "REFUSES",
                "kind": "high",
                "quote": "“Stone is effectively non-movable for me under this item set.”",
                "fragments": ["Stone is effectively non-movable for me under this item set"],
            },
            {
                "agent": "NANO A1",
                "role": "ACCEPTS",
                "kind": "concede",
                "quote": (
                    "“Gives me Jewel, Pencil, Apple, Quill totaling 29 now; avoids risk "
                    "of no-deal; Stone remains with Agent_2.”"
                ),
                "fragments": [
                    "Gives me Jewel, Pencil, Apple, Quill totaling 29 now; avoids risk of no-deal; Stone remains with Agent_2."
                ],
            },
        ],
        "outcome_title": "GPT-5.4 WINS THE BILATERAL SPLIT",
        "outcome_text": "GPT-5.4: 71  •  GPT-5-nano: 29",
        "outcome_kind": "high",
        "expected_utilities": {"Agent_1": 29.0, "Agent_2": 71.0},
    },
    {
        "n": "n = 4",
        "config": "RUN 31",
        "threshold": "3 of 4 votes needed\nfor supermajority",
        "source_dir": "config_0031_game1_team_coordination_n4_comp_0p5_gpt_5p4_high_last_seed1",
        "bubbles": [
            {
                "agent": "NANO A1",
                "role": "COALITION PLAN FOR A1-A3",
                "kind": "nano",
                "quote": (
                    "“Here’s a strong starting proposal for Round 1 ... "
                    "Total (Nano): 197.”"
                ),
                "fragments": [
                    "Here’s a strong starting proposal for Round 1",
                    "Total (Nano): 197",
                ],
            },
            {
                "agent": "GPT-5.4 A4",
                "role": "REFRAMES",
                "kind": "high",
                "quote": (
                    "“The 197 total is only ‘max utility’ for Agents 1–3, "
                    "not for the full 4-agent table.”"
                ),
                "fragments": [
                    "the 197 total is only “max utility” for Agents 1–3, not for the full 4-agent table"
                ],
            },
            {
                "agent": "NANO A1",
                "role": "REJECTS THE FRAME",
                "kind": "success",
                "quote": "“Agent_4 is not part of the Nano team’s internal utility calculation.”",
                "fragments": ["Agent_4 is not part of the Nano team’s internal utility calculation"],
            },
        ],
        "outcome_title": "NANO COALITION PASSES 3–1",
        "outcome_text": "Nano: 197 / 197  •  GPT-5.4: 0",
        "outcome_kind": "success",
        "expected_utilities": {
            "Agent_1": 72.0,
            "Agent_2": 48.0,
            "Agent_3": 77.0,
            "Agent_4": 0.0,
        },
    },
    {
        "n": "n > 4",
        "config": "RUN 67  •  n = 8",
        "threshold": "6 of 8 votes needed\nfor supermajority",
        "source_dir": "config_0067_game1_team_coordination_n8_comp_0p25_gpt_5p4_high_last_seed1",
        "bubbles": [
            {
                "agent": "NANO A1",
                "role": "COALITION PLAN FOR A1-A7",
                "kind": "nano",
                "quote": (
                    "“Compass (13) -> Agent_3 ... Coin (15) -> Agent_6 ... "
                    "Goal: reach a binding coalition ... highest expected sum for the Nano team.”"
                ),
                "fragments": [
                    "Compass (13) -> Agent_3",
                    "Coin (15) -> Agent_6",
                    "Goal: reach a binding coalition that, after discounting, yields the highest expected sum for the Nano team.",
                ],
            },
            {
                "agent": "GPT-5.4 A8",
                "role": "REDEFINES THE TEAM",
                "kind": "high",
                "quote": (
                    "“If we are truly maximizing total across all 8 agents, then Compass "
                    "should be reopened immediately and likely moved to Agent_8.”"
                ),
                "fragments": [
                    "If we are truly maximizing total across **all 8 agents**",
                    "Compass should be reopened immediately",
                    "likely moved to **Agent_8**",
                ],
            },
            {
                "agent": "NANO A1",
                "role": "ADOPTS THE FRAME",
                "kind": "concede",
                "quote": (
                    "“If we move Compass and Coin to Agent_8, the team-wide sum would "
                    "improve greatly ... Net increase to the Nano team’s sum: +26.”"
                ),
                "fragments": [
                    "If we move Compass and Coin to Agent_8, the team-wide sum would improve greatly:",
                    "Net increase to the Nano team’s sum: +26",
                ],
            },
        ],
        "outcome_title": "GPT-5.4 IS COUNTED INSIDE TEAM UTILITY",
        "outcome_text": "Nano: 372 / 401 optimum    •    GPT-5.4: 55",
        "outcome_kind": "high",
        "expected_utilities": {
            "Agent_1": 70.0,
            "Agent_2": 67.0,
            "Agent_3": 38.0,
            "Agent_4": 59.0,
            "Agent_5": 54.0,
            "Agent_6": 50.0,
            "Agent_7": 34.0,
            "Agent_8": 55.0,
        },
    },
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=text_font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 5,
) -> int:
    x, y = xy
    line_height = text_font.getbbox("Ag")[3] - text_font.getbbox("Ag")[1]
    for line in wrap_lines(draw, text, text_font, max_width):
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height + line_gap
    return y


def draw_robot_avatar(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    *,
    color: str,
    radius: int,
) -> None:
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    head_w = round(radius * 1.10)
    head_h = round(radius * 0.82)
    left = cx - head_w // 2
    top = cy - head_h // 2 + 3
    draw.rectangle((left, top, left + head_w, top + head_h), fill=COLORS["white"])
    antenna_top = cy - radius + max(5, radius // 6)
    draw.line((cx, top, cx, antenna_top), fill=COLORS["white"], width=max(2, radius // 8))
    dot = max(2, radius // 9)
    draw.ellipse((cx - dot, antenna_top - dot, cx + dot, antenna_top + dot), fill=COLORS["white"])
    eye_r = max(2, radius // 9)
    eye_y = top + round(head_h * 0.43)
    for eye_x in (cx - round(head_w * 0.22), cx + round(head_w * 0.22)):
        draw.ellipse(
            (eye_x - eye_r, eye_y - eye_r, eye_x + eye_r, eye_y + eye_r),
            fill=color,
        )
    mouth_y = top + round(head_h * 0.70)
    draw.line(
        (cx - round(head_w * 0.20), mouth_y, cx + round(head_w * 0.20), mouth_y),
        fill=color,
        width=max(2, radius // 9),
    )


def kind_colors(kind: str) -> tuple[str, str, str]:
    if kind == "nano":
        return COLORS["nano"], COLORS["nano_bg"], COLORS["nano_dark"]
    if kind == "high":
        return COLORS["high"], COLORS["high_bg"], COLORS["high_dark"]
    if kind == "success":
        return COLORS["success"], COLORS["success_bg"], COLORS["success"]
    if kind == "concede":
        return COLORS["concede"], COLORS["concede_bg"], COLORS["concede_dark"]
    raise ValueError(f"Unknown kind: {kind}")


def draw_dialogue_bubble(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    agent: str,
    role: str,
    kind: str,
    quote: str,
    quote_size: int,
) -> None:
    left, top, right, bottom = bounds
    color, fill, label_color = kind_colors(kind)
    bubble_left = left + 82
    draw.polygon(
        [(bubble_left, top + 42), (bubble_left - 24, top + 60), (bubble_left, top + 78)],
        fill=fill,
    )
    draw.rectangle((bubble_left, top, right, bottom), fill=fill)
    draw_robot_avatar(draw, (left + 34, top + 56), color=color, radius=34)
    draw.text(
        (bubble_left + 22, top + 13),
        f"{agent}  ·  {role}",
        font=font(27, bold=True),
        fill=label_color,
    )
    draw_wrapped(
        draw,
        quote,
        (bubble_left + 22, top + 56),
        text_font=font(quote_size),
        fill=COLORS["text"],
        max_width=right - bubble_left - 42,
        line_gap=4,
    )


def dialogue_box_height(
    draw: ImageDraw.ImageDraw,
    quote: str,
    *,
    text_width: int,
    quote_size: int,
) -> int:
    quote_font = font(quote_size)
    line_height = quote_font.getbbox("Ag")[3] - quote_font.getbbox("Ag")[1]
    line_count = len(wrap_lines(draw, quote, quote_font, text_width))
    return 56 + line_count * (line_height + 4) + 12


def verify_sources() -> None:
    for panel in PANELS:
        source_dir = RESULTS / panel["source_dir"]
        interactions_path = source_dir / "all_interactions.json"
        results_path = source_dir / "experiment_results.json"
        interactions_text = interactions_path.read_text(encoding="utf-8")
        for bubble in panel["bubbles"]:
            for fragment in bubble["fragments"]:
                if fragment not in interactions_text:
                    raise RuntimeError(f"Quote fragment not found in {interactions_path}: {fragment}")
        result = json.loads(results_path.read_text(encoding="utf-8"))
        if result["final_utilities"] != panel["expected_utilities"]:
            raise RuntimeError(
                f"Unexpected final utilities in {results_path}: {result['final_utilities']}"
            )


def build_slide() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["white"])
    draw = ImageDraw.Draw(image)
    dialogue_top = 105
    minimum_bubble_gap = 18
    quote_size = 32

    panel_layouts = []
    for panel, panel_width in zip(PANELS, PANEL_WIDTHS, strict=True):
        text_width = panel_width - 2 * HORIZONTAL_INSET - 124
        bubble_heights = [
            dialogue_box_height(
                draw,
                bubble["quote"],
                text_width=text_width,
                quote_size=quote_size,
            )
            for bubble in panel["bubbles"]
        ]
        minimum_group_height = sum(bubble_heights) + minimum_bubble_gap * (
            len(bubble_heights) - 1
        )
        panel_layouts.append((bubble_heights, minimum_group_height))

    dialogue_group_height = max(group_height for _, group_height in panel_layouts)
    result_top = dialogue_top + dialogue_group_height + 22
    result_bottom = HEIGHT - 20

    left = 0
    for index, (panel, panel_width) in enumerate(
        zip(PANELS, PANEL_WIDTHS, strict=True)
    ):
        right = left + panel_width
        inner_left = left + HORIZONTAL_INSET
        inner_right = right - HORIZONTAL_INSET

        draw.text((inner_left, -3), panel["n"], font=font(48, bold=True), fill=COLORS["navy"])
        draw.text((inner_left, 57), panel["config"], font=font(25, bold=True), fill=COLORS["muted"])

        threshold_font = font(28, bold=True)
        threshold_box = draw.multiline_textbbox(
            (0, 0), panel["threshold"], font=threshold_font, spacing=0, align="center"
        )
        threshold_width = threshold_box[2] - threshold_box[0]
        draw.multiline_text(
            (inner_right - threshold_width, 3),
            panel["threshold"],
            font=threshold_font,
            fill=COLORS["muted"],
            spacing=0,
            align="center",
        )

        bubble_heights = panel_layouts[index][0]
        total_gap = dialogue_group_height - sum(bubble_heights)
        gap_count = len(bubble_heights) - 1
        bubble_gap, extra_gap_pixels = divmod(total_gap, gap_count)
        bubble_top = dialogue_top
        for bubble_index, (bubble, bubble_height) in enumerate(
            zip(panel["bubbles"], bubble_heights, strict=True)
        ):
            draw_dialogue_bubble(
                draw,
                (inner_left, bubble_top, inner_right, bubble_top + bubble_height),
                agent=bubble["agent"],
                role=bubble["role"],
                kind=bubble["kind"],
                quote=bubble["quote"],
                quote_size=quote_size,
            )
            bubble_top += bubble_height
            if bubble_index < gap_count:
                bubble_top += bubble_gap + (bubble_index < extra_gap_pixels)

        if bubble_top != dialogue_top + dialogue_group_height:
            raise RuntimeError("Dialogue groups must have equal rendered heights")

        _, outcome_fill, outcome_label = kind_colors(panel["outcome_kind"])
        draw.rectangle(
            (inner_left, result_top, inner_right, result_bottom),
            fill=outcome_fill,
        )
        draw.text(
            (inner_left + 22, result_top + 15),
            panel["outcome_title"],
            font=font(31, bold=True),
            fill=outcome_label,
        )
        draw.multiline_text(
            (inner_left + 22, result_top + 61),
            panel["outcome_text"],
            font=font(36, bold=True),
            fill=COLORS["text"],
            spacing=0,
        )

        left = right

    return image


def main() -> None:
    verify_sources()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_slide().save(PNG_PATH, format="PNG", optimize=True)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
