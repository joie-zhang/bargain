#!/usr/bin/env python3
"""Validate the active paper figure manifest."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs/reproducibility"
PAPER_ROOTS = {
    "nextgame": {
        "root": PROJECT_ROOT / "overleaf/NExT_Game_2026_style_new",
        "manifest": DOCS_ROOT / "paper_figure_manifest.csv",
        "tex": ("4_analysis.tex", "appendix.tex"),
        "detailed": True,
    },
    "icml": {
        "root": PROJECT_ROOT / "overleaf/icml_aiwild_template",
        "manifest": DOCS_ROOT / "icml_figure_manifest.csv",
        "tex": ("1_intro.tex", "4_analysis.tex", "appendix.tex"),
        "detailed": False,
    },
    "neurips": {
        "root": PROJECT_ROOT / "overleaf/neurips",
        "manifest": DOCS_ROOT / "neurips_figure_manifest.csv",
        "tex": ("1_intro.tex", "4_analysis.tex", "appendix.tex"),
        "detailed": False,
    },
}
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def active_graphics(root: Path, tex_names: tuple[str, ...]) -> set[str]:
    graphics: set[str] = set()
    for tex_name in tex_names:
        tex_path = root / tex_name
        graphics.update(INCLUDE_RE.findall(tex_path.read_text(encoding="utf-8")))
    return graphics


def split_paths(value: str) -> list[str]:
    if not value or value == "UNRESOLVED":
        return []
    return [item for item in value.split(";") if item]


def validate_coverage(
    name: str,
    rows: list[dict[str, str]],
    active: set[str],
    errors: list[str],
) -> None:
    manifest_graphics = {row["graphic_path"] for row in rows}
    for path in sorted(active - manifest_graphics):
        errors.append(f"{name}: active graphic is missing from manifest: {path}")
    for path in sorted(manifest_graphics - active):
        errors.append(f"{name}: manifest graphic is not active in paper: {path}")


def validate_detailed_rows(
    rows: list[dict[str, str]],
    root: Path,
    errors: list[str],
    warnings: list[str],
) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        figure_id = row["figure_id"]
        if figure_id in ids:
            errors.append(f"nextgame: duplicate figure_id: {figure_id}")
        ids.add(figure_id)

        graphic = root / row["graphic_path"]
        if not graphic.is_file():
            errors.append(f"{figure_id}: graphic does not exist: {row['graphic_path']}")
        elif file_hash(graphic) != row["sha256"]:
            errors.append(f"{figure_id}: graphic hash changed: {row['graphic_path']}")

        for producer in split_paths(row["producer_script"]):
            if not (PROJECT_ROOT / producer).is_file():
                errors.append(f"{figure_id}: producer does not exist: {producer}")

        for source in split_paths(row["source_inputs"]):
            source_path = Path(source) if source.startswith("/") else PROJECT_ROOT / source
            if not source_path.exists():
                message = f"{figure_id}: source does not exist: {source}"
                if row["provenance_status"] == "nonportable":
                    warnings.append(message)
                else:
                    errors.append(message)

        if row["provenance_status"] in {"unresolved", "manual-composite", "nonportable"}:
            warnings.append(
                f"{figure_id}: provenance requires work ({row['provenance_status']})"
            )
    return ids


def validate_reference_rows(
    name: str,
    rows: list[dict[str, str]],
    root: Path,
    detailed_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    ids: set[str] = set()
    for row in rows:
        figure_id = row["figure_id"]
        if figure_id in ids:
            errors.append(f"{name}: duplicate figure_id: {figure_id}")
        ids.add(figure_id)

        graphic = root / row["graphic_path"]
        expected_missing = row["resolution_status"] == "missing"
        if expected_missing:
            if graphic.exists():
                errors.append(f"{figure_id}: marked missing but now exists: {row['graphic_path']}")
            warnings.append(f"{figure_id}: active TeX graphic path is missing")
        elif not graphic.is_file():
            errors.append(f"{figure_id}: graphic does not exist: {row['graphic_path']}")
        elif file_hash(graphic) != row["sha256"]:
            errors.append(f"{figure_id}: graphic hash changed: {row['graphic_path']}")

        for reference in split_paths(row["provenance_ref"]):
            if reference in detailed_ids:
                continue
            if not (PROJECT_ROOT / reference).is_file():
                errors.append(f"{figure_id}: provenance reference does not exist: {reference}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    configs = {name: dict(config) for name, config in PAPER_ROOTS.items()}
    rows_by_name = {
        name: read_manifest(config["manifest"])
        for name, config in configs.items()
    }
    active_by_name = {
        name: active_graphics(config["root"], config["tex"])
        for name, config in configs.items()
    }
    for name in configs:
        validate_coverage(name, rows_by_name[name], active_by_name[name], errors)

    detailed_ids = validate_detailed_rows(
        rows_by_name["nextgame"],
        configs["nextgame"]["root"],
        errors,
        warnings,
    )
    for name in ("icml", "neurips"):
        validate_reference_rows(
            name,
            rows_by_name[name],
            configs[name]["root"],
            detailed_ids,
            errors,
            warnings,
        )

    for name in configs:
        print(
            f"{name}: manifest rows={len(rows_by_name[name])} "
            f"active graphics={len(active_by_name[name])}"
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
