from __future__ import annotations

import gzip
import json
from pathlib import Path

from ui.random_monoculture_sample_viewer import BatchIndex, read_prompt_text


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_batch_index_reads_binding_team_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs" / "config_0001_example"
    config = {
        "config_id": 1,
        "game_label": "game1",
        "game_type": "item_allocation",
        "n_agents": 4,
        "competition_id": "comp_0p25",
        "competition_level": 0.25,
        "adversary_position": "last",
        "seed_replicate": 2,
        "adversary_model": "gpt-5.4-high",
        "agent_role_map": {
            "Agent_1": "baseline",
            "Agent_2": "baseline",
            "Agent_3": "baseline",
            "Agent_4": "adversary",
        },
        "team_coordination": {
            "captain_id": "Agent_2",
            "member_ids": ["Agent_1", "Agent_2", "Agent_3"],
        },
        "output_dir": str(output_dir),
    }
    write_json(tmp_path / "configs" / "config_0001.json", config)
    write_json(
        output_dir / "experiment_results.json",
        {
            "config": config,
            "consensus_reached": True,
            "final_round": 1,
            "final_utilities": {
                "Agent_1": 20,
                "Agent_2": 30,
                "Agent_3": 40,
                "Agent_4": 10,
            },
            "conversation_logs": [],
            "vote_integrity": {},
        },
    )
    write_json(output_dir / "all_interactions.json", [])

    index = BatchIndex(tmp_path)

    assert index.summary()["total"] == 1
    row = index.records["config_0001"].row
    assert row["state"] == "SUCCESS"
    assert row["captain_id"] == "Agent_2"
    assert row["team_size"] == 3
    assert row["team_utility"] == 90
    assert row["adversary_utility"] == 10


def test_read_prompt_text_decompresses_externalized_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "run" / "externalized_prompts" / "prompt.txt.gz"
    prompt_path.parent.mkdir(parents=True)
    with gzip.open(prompt_path, "wt", encoding="utf-8") as handle:
        handle.write("full prompt")

    interaction = {"prompt_storage_path": "externalized_prompts/prompt.txt.gz"}
    assert read_prompt_text(tmp_path / "run", interaction) == "full prompt"


def test_read_prompt_text_rejects_path_escape(tmp_path: Path) -> None:
    interaction = {"prompt_storage_path": "../outside.txt"}
    try:
        read_prompt_text(tmp_path / "run", interaction)
    except ValueError as exc:
        assert "leaves the run directory" in str(exc)
    else:
        raise AssertionError("Expected a path traversal error")
