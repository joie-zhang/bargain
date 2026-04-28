from __future__ import annotations

import csv
from collections import Counter
import json

from scripts.full_games123_multiagent_batch import (
    BASELINE_MODEL,
    HETEROGENEOUS_EXCLUDED_MODELS,
    build_command,
    build_configs,
    filtered_heterogeneous_pool,
    rho_lower_bound,
    validate_configs,
    write_generated_files,
)


def test_full_games123_generation_counts_and_core_invariants(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    validation = validate_configs(configs)
    assert validation["valid"], validation["errors"][:10]
    assert len(configs) == 2730

    game_family_counts = Counter(
        (cfg["game_label"], cfg["experiment_family"])
        for cfg in configs
    )
    assert game_family_counts[("game1", "homogeneous_control")] == 50
    assert game_family_counts[("game1", "homogeneous_adversary")] == 500
    assert game_family_counts[("game1", "heterogeneous_random")] == 500
    assert game_family_counts[("game2", "homogeneous_control")] == 40
    assert game_family_counts[("game2", "homogeneous_adversary")] == 400
    assert game_family_counts[("game2", "heterogeneous_random")] == 400
    assert game_family_counts[("game3", "homogeneous_control")] == 40
    assert game_family_counts[("game3", "homogeneous_adversary")] == 400
    assert game_family_counts[("game3", "heterogeneous_random")] == 400

    for cfg in configs:
        assert cfg["parallel_phases"] is True
        assert cfg["max_rounds"] == 10
        assert cfg["discussion_turns"] == 2
        assert cfg.get("max_tokens_voting") is None
        assert "qwq" not in json.dumps(cfg, sort_keys=True).lower()
        assert len(cfg["models"]) == cfg["n_agents"]


def test_full_games123_model_order_and_heterogeneous_pool(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    pool = filtered_heterogeneous_pool()
    assert len(pool) == 24
    assert "qwq-32b" not in pool
    assert HETEROGENEOUS_EXCLUDED_MODELS == {"qwq-32b"}

    adversary_first = next(
        cfg for cfg in configs
        if cfg["experiment_family"] == "homogeneous_adversary"
        and cfg["adversary_position"] == "first"
    )
    assert adversary_first["models"][0] == adversary_first["adversary_model"]
    assert all(model == BASELINE_MODEL for model in adversary_first["models"][1:])

    adversary_last = next(
        cfg for cfg in configs
        if cfg["experiment_family"] == "homogeneous_adversary"
        and cfg["adversary_position"] == "last"
    )
    assert adversary_last["models"][-1] == adversary_last["adversary_model"]
    assert all(model == BASELINE_MODEL for model in adversary_last["models"][:-1])

    hetero = [
        cfg for cfg in configs
        if cfg["experiment_family"] == "heterogeneous_random"
    ]
    assert hetero
    for cfg in hetero:
        assert len(set(cfg["models"])) == cfg["n_agents"]
        assert "qwq-32b" not in cfg["models"]
        assert cfg["model_pool_size"] == 24


def test_full_games123_game_specific_parameters(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    for cfg in configs:
        n = cfg["n_agents"]
        if cfg["game_label"] == "game1":
            assert cfg["num_items"] == int(2.5 * n)
            assert cfg["competition_level"] in {0.0, 0.25, 0.5, 0.75, 1.0}
        elif cfg["game_label"] == "game2":
            assert cfg["n_issues"] == 10
            assert cfg["theta"] in {0.2, 0.8}
            assert cfg["rho"] == 0.9 or abs(cfg["rho"] - rho_lower_bound(n)) < 1e-12
        elif cfg["game_label"] == "game3":
            assert cfg["m_projects"] == int(2.5 * n)
            assert cfg["sigma"] in {0.2, 0.5}
            assert cfg["alpha"] in {0.2, 0.8}


def test_full_games123_command_omits_voting_cap_by_default(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    command = build_command(configs[0])
    assert "--max-tokens-voting" not in command

    capped = dict(configs[0])
    capped["max_tokens_voting"] = 8192
    capped_command = build_command(capped)
    assert capped_command[capped_command.index("--max-tokens-voting") + 1] == "8192"


def test_full_games123_experiment_index_is_replicable_audit_table(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    write_generated_files(
        tmp_path,
        {"slurm_time": "08:00:00", "max_concurrent": 50},
        configs,
    )

    with (tmp_path / "configs" / "experiment_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2730
    assert rows[0]["agent_1_model"] == "gpt-5-nano"
    assert rows[0]["agent_2_model"] == "gpt-5-nano"
    assert rows[0]["agent_1_role"] == "baseline_control"
    assert rows[0]["agent_3_model"] == ""
    assert rows[0]["model_pool"] == ""
    assert rows[0]["heterogeneous_sampling"] == ""

    hetero = next(row for row in rows if row["experiment_family"] == "heterogeneous_random")
    hetero_config = configs[int(hetero["config_id"]) - 1]
    n_agents = int(hetero["n_agents"])
    ordered_agent_models = [
        hetero[f"agent_{idx}_model"]
        for idx in range(1, n_agents + 1)
    ]
    assert ordered_agent_models == hetero_config["models"]
    assert hetero["models"] == "+".join(hetero_config["models"])
    assert hetero["heterogeneous_draw_seed"] == str(hetero_config["heterogeneous_draw_seed"])
    assert hetero["random_seed"] == str(hetero_config["random_seed"])
    assert hetero["seed"] == str(hetero_config["seed"])
    assert hetero["model_pool_size"] == "24"
    assert hetero["model_pool"] == "+".join(hetero_config["model_pool"])
    assert json.loads(hetero["heterogeneous_sampling"]) == hetero_config["heterogeneous_sampling"]
    assert hetero["elo_bucket_method"] == hetero_config["elo_bucket_method"]
    assert hetero["agent_1_elo"] == str(hetero_config["agent_elo_map"]["Agent_1"])
    assert hetero["agent_1_elo_bucket"] == hetero_config["agent_elo_bucket_map"]["Agent_1"]
    assert hetero["agent_1_role"] == "heterogeneous_random_agent"
