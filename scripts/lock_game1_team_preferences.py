#!/usr/bin/env python3
"""Lock an existing team-treatment batch to literal historical preferences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    config_paths = sorted((root / "configs").glob("config_*.json"))
    if len(config_paths) != 100:
        raise ValueError(f"Expected 100 configs, found {len(config_paths)}")

    locked_hashes: dict[int, str] = {}
    for config_path in config_paths:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        control_path = Path(config["control_result_path"])
        control = json.loads(control_path.read_text(encoding="utf-8"))
        preferences = control.get("agent_preferences")
        if not isinstance(preferences, dict):
            raise ValueError(f"Missing control preferences: {control_path}")
        if set(preferences) != set(config["agent_role_map"]):
            raise ValueError(f"Agent IDs differ for config {config['config_id']}")
        if any(
            not isinstance(values, list) or len(values) != int(config["num_items"])
            for values in preferences.values()
        ):
            raise ValueError(f"Preference dimensions differ for config {config['config_id']}")
        digest = canonical_json_sha256(preferences)
        config["fixed_agent_preferences"] = preferences
        config["fixed_agent_preferences_sha256"] = digest
        write_json_atomic(config_path, config)
        locked_hashes[int(config["config_id"])] = digest

    lineage_path = root / "control_lineage.csv"
    with lineage_path.open(newline="", encoding="utf-8") as handle:
        lineage = list(csv.DictReader(handle))
    for row in lineage:
        row["fixed_agent_preferences_sha256"] = locked_hashes[int(row["config_id"])]
    fieldnames = list(lineage[0])
    if "fixed_agent_preferences_sha256" not in fieldnames:
        fieldnames.append("fixed_agent_preferences_sha256")
    temporary_lineage = lineage_path.with_suffix(".csv.tmp")
    with temporary_lineage.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lineage)
    os.replace(temporary_lineage, lineage_path)

    print(json.dumps({
        "results_root": str(root),
        "locked_configs": len(locked_hashes),
        "unique_preference_hashes": len(set(locked_hashes.values())),
    }, indent=2))


if __name__ == "__main__":
    main()
