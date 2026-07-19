#!/usr/bin/env python3
"""Apply Figure 3 label edits while preserving the original plotted content.

The direct source script for the current combined Figure 3 was not checked in.
This script therefore starts from the byte-for-byte recreated original PNG and
edits only the requested title, axis-label, and legend-label regions.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PNG = Path(__file__).resolve().parent / "assets" / "fairshare_residual_combined_base.png"
OUT_ICML = (
    PROJECT_ROOT
    / "overleaf"
    / "icml_aiwild_template"
    / "graphics"
    / "n2_gpt5_nano"
    / "fairshare_residual_combined.png"
)
OUT_NEURIPS = (
    PROJECT_ROOT
    / "overleaf"
    / "neurips"
    / "graphics"
    / "n2_gpt5_nano"
    / "fairshare_residual_combined.png"
)
FONT_PATH = Path(font_manager.findfont("DejaVu Sans"))
RIGHT_PANEL_SHIFT = 16
RIGHT_PANEL_SOURCE_LEFT = 1210


def draw_centered(draw: ImageDraw.ImageDraw, text: str, x_center: int, y_top: int, font: ImageFont.FreeTypeFont) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    draw.text((x_center - width / 2, y_top - top), text, fill=(0, 0, 0), font=font)


def paste_rotated_label(
    image: Image.Image,
    text: str,
    x_left: int,
    y_center: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    tmp = Image.new("RGBA", (1200, 90), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tmp)
    draw.text((0, 0), text, fill=(0, 0, 0, 255), font=font)
    bbox = tmp.getbbox()
    if bbox is None:
        return
    cropped = tmp.crop(bbox)
    rotated = cropped.rotate(90, expand=True)
    image.alpha_composite(rotated, (x_left, int(y_center - rotated.height / 2)))


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int = 32,
    gap: int = 16,
) -> None:
    x0, y0, x1, y1 = xy
    x = x0
    while x < x1:
        draw.line((x, y0, min(x + dash, x1), y1), fill=fill, width=width)
        x += dash + gap


def main() -> None:
    if not BASE_PNG.exists():
        raise FileNotFoundError(
            f"Missing original base PNG: {BASE_PNG}. Recreate it from the saved original before rerunning."
        )

    image = Image.open(BASE_PNG).convert("RGBA")
    if image.size != (2293, 1047):
        raise ValueError(f"Expected 2293x1047 base image, got {image.size}")

    right_panel = image.crop((RIGHT_PANEL_SOURCE_LEFT, 0, 2293 - RIGHT_PANEL_SHIFT, 1047))
    draw = ImageDraw.Draw(image)
    draw.rectangle((RIGHT_PANEL_SOURCE_LEFT, 0, 2293, 1047), fill=(255, 255, 255, 255))
    image.alpha_composite(right_panel, (RIGHT_PANEL_SOURCE_LEFT + RIGHT_PANEL_SHIFT, 0))

    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(FONT_PATH), 48)
    axis_font = ImageFont.truetype(str(FONT_PATH), 48)
    legend_font = ImageFont.truetype(str(FONT_PATH), 32)

    white = (255, 255, 255, 255)

    # Titles.
    draw.rectangle((0, 0, 2293, 70), fill=white)
    draw_centered(draw, "Bilateral (N=2)", 650, 14, title_font)
    draw_centered(draw, "Multi-agent (N≥2)", 1783 + RIGHT_PANEL_SHIFT, 14, title_font)

    # Left y-axis label.
    draw.rectangle((0, 105, 86, 935), fill=white)
    paste_rotated_label(image, "Adversary utility above fair share", 8, 520, axis_font)

    # Right y-axis label, moved rightward to open space between the two panels.
    draw.rectangle((1120, 105, 1216, 890), fill=white)
    paste_rotated_label(image, "Utility above fair share", 1178, 520, axis_font)

    # Right x-axis label.
    draw.rectangle((1535 + RIGHT_PANEL_SHIFT, 963, 2025 + RIGHT_PANEL_SHIFT, 1047), fill=white)
    draw_centered(draw, "Elo", 1783 + RIGHT_PANEL_SHIFT, 968, axis_font)

    # Right legend text labels. Keep the existing markers/lines and legend box.
    draw.rectangle((1826 + RIGHT_PANEL_SHIFT, 696, 2253 + RIGHT_PANEL_SHIFT, 884), fill=white)
    legend_x = 1845 + RIGHT_PANEL_SHIFT
    for y, label in [
        (712, "Heterogenous"),
        (760, "Homogenous: adversary"),
        (808, "Homogenous: baseline"),
        (856, "Homogenous Control"),
    ]:
        draw.text((legend_x, y), label, fill=(0, 0, 0), font=legend_font)

    # Left legend redraw. Keep all labels in the same font as the right legend.
    draw.rectangle((1094, 718, 1120, 893), fill=white)
    draw.rounded_rectangle(
        (500, 718, 1094, 893),
        radius=8,
        fill=(255, 255, 255, 255),
        outline=(209, 213, 219, 255),
        width=2,
    )
    for y, color, label in [
        (762, (124, 58, 237, 255), "Game 1: Item allocation"),
        (814, (15, 118, 110, 255), "Game 2: Diplomatic Treaty"),
        (866, (249, 115, 22, 255), "Game 3: Co-funding"),
    ]:
        draw_dashed_line(draw, (522, y, 630, y), color, width=8)
        draw.text((652, y - 19), label, fill=(0, 0, 0), font=legend_font)
    draw.line((1116, 78, 1116, 910), fill=(0, 0, 0, 255), width=2)
    draw.line((1094, 910, 1116, 910), fill=(0, 0, 0, 255), width=2)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", "Matplotlib version3.10.7, https://matplotlib.org/")
    for out_path in (OUT_ICML, OUT_NEURIPS):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, pnginfo=pnginfo, dpi=(200, 200))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
