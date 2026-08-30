"""Tests for the paired reviewer context-compaction pilot grid."""

from pathlib import Path

from scripts import context_compaction_pilot as pilot


def test_pilot_grid_has_25_exact_pairs(tmp_path: Path) -> None:
    configs = pilot.build_configs(tmp_path / "pilot")
    assert pilot.validate_configs(configs) == []
    assert len(configs) == 50
    assert len({config["pair_id"] for config in configs}) == 25


def test_pair_differs_only_in_treatment_metadata_and_output(tmp_path: Path) -> None:
    configs = pilot.build_configs(tmp_path / "pilot")
    first_pair = [config for config in configs if config["pair_id"] == "n02_seed01"]
    assert len(first_pair) == 2

    by_arm = {config["treatment_arm"]: config for config in first_pair}
    on = by_arm["on"]
    off = by_arm["off"]

    paired_fields = (
        "n_agents",
        "models",
        "random_seed",
        "m_projects",
        "alpha",
        "sigma",
        "max_rounds",
        "discussion_turns",
        "parallel_phases",
    )
    for field in paired_fields:
        assert on[field] == off[field]

    assert on["compaction_enabled"] is True
    assert off["compaction_enabled"] is False
    assert on["output_dir"] != off["output_dir"]


def test_five_seeded_pairs_at_every_n(tmp_path: Path) -> None:
    configs = pilot.build_configs(tmp_path / "pilot")
    for n_agents in pilot.N_VALUES:
        subset = [config for config in configs if config["n_agents"] == n_agents]
        assert len(subset) == 10
        assert len({config["random_seed"] for config in subset}) == 5
        for environment_seed in {config["random_seed"] for config in subset}:
            arms = {
                config["treatment_arm"]
                for config in subset
                if config["random_seed"] == environment_seed
            }
            assert arms == {"on", "off"}


def test_followup_grid_uses_fresh_replicate_labels_and_seeds(tmp_path: Path) -> None:
    pilot_configs = pilot.build_configs(tmp_path / "pilot", replicate_start=1)
    followup_configs = pilot.build_configs(tmp_path / "followup", replicate_start=6)

    assert {config["seed_replicate"] for config in followup_configs} == set(
        range(6, 11)
    )
    assert not (
        {config["random_seed"] for config in pilot_configs}
        & {config["random_seed"] for config in followup_configs}
    )
    assert pilot.validate_configs(followup_configs) == []


def test_n_selection_keeps_randomized_arms_adjacent(tmp_path: Path) -> None:
    configs = pilot.build_configs(tmp_path / "pilot")
    by_id = {config["config_id"]: config for config in configs}
    selected = pilot.paired_randomized_ids(configs, 8)
    assert len(selected) == 10

    for index in range(0, len(selected), 2):
        left = by_id[selected[index]]
        right = by_id[selected[index + 1]]
        assert left["pair_id"] == right["pair_id"]
        assert {left["treatment_arm"], right["treatment_arm"]} == {"on", "off"}


def test_paired_task_selection_can_skip_one_completed_arm(tmp_path: Path) -> None:
    configs = pilot.build_configs(tmp_path / "pilot")
    selected = pilot.paired_randomized_ids(configs, 4)
    source_order = {config_id: index for index, config_id in enumerate(selected)}
    unfinished = selected[1:]
    by_id = {config["config_id"]: config for config in configs}
    grouped: dict[str, list[int]] = {}
    for config_id in unfinished:
        grouped.setdefault(by_id[config_id]["pair_id"], []).append(config_id)
    rows = [
        sorted(ids, key=source_order.__getitem__)
        for _, ids in sorted(
            grouped.items(),
            key=lambda item: min(source_order[config_id] for config_id in item[1]),
        )
    ]
    assert len(rows) == 5
    assert len(rows[0]) == 1
    assert sum(map(len, rows)) == 9
