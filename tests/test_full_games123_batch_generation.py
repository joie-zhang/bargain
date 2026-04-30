from __future__ import annotations

from argparse import Namespace
import csv
from collections import Counter
import json

import scripts.full_games123_multiagent_batch as batch
from strong_models_experiment.configs import STRONG_MODELS_CONFIG
from scripts.full_games123_multiagent_batch import (
    ADVERSARY_MODELS,
    BASELINE_MODEL,
    GEMINI_MODELS,
    HETEROGENEOUS_EXCLUDED_MODELS,
    HETEROGENEOUS_RUNS_PER_STRATUM,
    HETEROGENEOUS_SAMPLING_STRATEGY,
    HETEROGENEOUS_STRATA_COUNT,
    HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH,
    HETEROGENEOUS_STRATEGY_PURE_RANDOM,
    build_command,
    build_configs,
    cmd_submit_selection,
    config_output_dir,
    read_config_id_file,
    filtered_heterogeneous_pool,
    rho_lower_bound,
    select_config_ids,
    selection_ids_path,
    selection_preset_criteria,
    squeue_array_states,
    validate_configs,
    write_generated_files,
    write_selection_files,
    write_slurm_file,
)


def write_valid_result(config):
    result_path = config_output_dir(config) / "experiment_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "config_id": config["config_id"],
                    "game_type": config["game_type"],
                    "random_seed": config["random_seed"],
                },
                "consensus_reached": False,
                "final_round": config["max_rounds"],
                "final_utilities": {
                    f"Agent_{idx}": 0.0 for idx in range(1, int(config["n_agents"]) + 1)
                },
            }
        ),
        encoding="utf-8",
    )
    return result_path


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
        payload_without_exclusion_log = {
            key: value
            for key, value in cfg.items()
            if key != "heterogeneous_excluded_models"
        }
        assert "qwq-32b" not in json.dumps(payload_without_exclusion_log, sort_keys=True).lower()
        assert "gemini-3-pro" not in json.dumps(payload_without_exclusion_log, sort_keys=True).lower()
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
    assert "gemini-3-pro" not in pool
    assert "gemini-3.1-pro" in pool
    assert HETEROGENEOUS_EXCLUDED_MODELS == {"qwq-32b", "gemini-3-pro"}
    assert batch.elo_for_model("gemini-3.1-pro") == 1494
    assert batch.elo_for_model("gemini-3-pro") == 1494
    assert STRONG_MODELS_CONFIG["gemini-3.1-pro"]["api_type"] == "openrouter"
    assert STRONG_MODELS_CONFIG["gemini-3.1-pro"]["model_id"] == "google/gemini-3.1-pro-preview"
    assert STRONG_MODELS_CONFIG["gemini-2.5-pro"]["api_type"] == "openrouter"
    assert STRONG_MODELS_CONFIG["gemini-2.5-pro"]["model_id"] == "google/gemini-2.5-pro"
    assert STRONG_MODELS_CONFIG["gpt-5.4-high"]["api_type"] == "openrouter"
    assert STRONG_MODELS_CONFIG["gpt-5.4-high"]["model_id"] == "openai/gpt-5.4"
    assert STRONG_MODELS_CONFIG["gpt-5.4-high"]["custom_parameters"]["reasoning"]["effort"] == "high"
    assert HETEROGENEOUS_SAMPLING_STRATEGY == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH
    assert batch.elo_for_model(BASELINE_MODEL) == batch.elo_for_model("gpt-5-nano-high") == 1337
    assert ADVERSARY_MODELS == [
        "amazon-nova-micro-v1.0",
        "gpt-4o-mini-2024-07-18",
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro",
        "gpt-5.4-high",
    ]

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

    homogeneous_adversary = [
        cfg for cfg in configs
        if cfg["experiment_family"] == "homogeneous_adversary"
    ]
    adversary_counts = Counter(cfg["adversary_model"] for cfg in homogeneous_adversary)
    assert set(adversary_counts) == set(ADVERSARY_MODELS)
    assert adversary_counts["gpt-5.4-high"] == 260
    assert "claude-opus-4-6-thinking" not in adversary_counts
    for cfg in homogeneous_adversary:
        if cfg["adversary_model"] != "gpt-5.4-high":
            continue
        adversary_agent = next(
            agent_id for agent_id, role in cfg["agent_role_map"].items()
            if role == "adversary"
        )
        assert cfg["agent_elo_map"][adversary_agent] == 1484

    hetero = [
        cfg for cfg in configs
        if cfg["experiment_family"] == "heterogeneous_random"
    ]
    assert hetero
    for cfg in hetero:
        assert len(set(cfg["models"])) == cfg["n_agents"]
        assert "qwq-32b" not in cfg["models"]
        assert "gemini-3-pro" not in cfg["models"]
        assert cfg["model_pool_size"] == 24
        assert cfg["heterogeneous_sampling_strategy"] == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH
        assert cfg["stratum_method"] == "equal_width_stddev"
        assert cfg["sampling_with_replacement"] is True
        assert cfg["planned_roster_replication"] is False
        assert cfg["stratum_stddev_min"] <= cfg["elo_stddev"] <= cfg["stratum_stddev_max"]
        assert all(cfg["agent_elo_map"].values())

    for key, count in Counter(
        (cfg["game_label"], cfg["n_agents"], cfg["competition_id"], cfg["stratum_index"])
        for cfg in hetero
    ).items():
        assert count == HETEROGENEOUS_RUNS_PER_STRATUM, key
    assert set(cfg["stratum_index"] for cfg in hetero) == set(range(HETEROGENEOUS_STRATA_COUNT))

    control = next(cfg for cfg in configs if cfg["experiment_family"] == "homogeneous_control")
    assert set(control["agent_elo_map"].values()) == {1337}


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
    assert hetero["heterogeneous_sampling_strategy"] == hetero_config["heterogeneous_sampling_strategy"]
    assert hetero["stratum_method"] == "equal_width_stddev"
    assert hetero["stratum_index"] == str(hetero_config["stratum_index"])
    assert hetero["subset_key"] == hetero_config["subset_key"]
    assert hetero["subset_model_ids_unordered"] == "+".join(
        hetero_config["subset_model_ids_unordered"]
    )
    assert hetero["subset_model_elos_unordered"] == "+".join(
        str(elo) for elo in hetero_config["subset_model_elos_unordered"]
    )
    assert hetero["elo_bucket_method"] == hetero_config["elo_bucket_method"]
    assert hetero["agent_1_elo"] == str(hetero_config["agent_elo_map"]["Agent_1"])
    assert hetero["agent_1_elo_bucket"] == hetero_config["agent_elo_bucket_map"]["Agent_1"]
    assert hetero["agent_1_role"] == "heterogeneous_random_agent"


def test_extended_derisk_selection_is_n10_amazon_and_gpt4o_only(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    config_by_id = {cfg["config_id"]: cfg for cfg in configs}
    ids = select_config_ids(
        configs,
        selection_preset_criteria("extended_derisk_n10_amazon_gpt4o"),
    )
    selected = [config_by_id[config_id] for config_id in ids]

    assert len(ids) == 52
    assert {cfg["experiment_family"] for cfg in selected} == {"homogeneous_adversary"}
    assert {cfg["n_agents"] for cfg in selected} == {10}
    assert {cfg["seed_replicate"] for cfg in selected} == {1}
    assert {cfg["adversary_position"] for cfg in selected} == {"first", "last"}
    assert {cfg["adversary_model"] for cfg in selected} == {
        "amazon-nova-micro-v1.0",
        "gpt-4o-mini-2024-07-18",
    }
    assert Counter(cfg["game_label"] for cfg in selected) == {
        "game1": 20,
        "game2": 16,
        "game3": 16,
    }
    assert Counter(cfg["adversary_model"] for cfg in selected) == {
        "amazon-nova-micro-v1.0": 26,
        "gpt-4o-mini-2024-07-18": 26,
    }


def test_gemini_and_non_gemini_swaths_partition_homogeneous_and_heterogeneous(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
    )
    config_by_id = {cfg["config_id"]: cfg for cfg in configs}

    non_gemini_homogeneous = select_config_ids(
        configs,
        selection_preset_criteria("non_gemini_homogeneous"),
    )
    gemini_homogeneous = select_config_ids(
        configs,
        selection_preset_criteria("gemini_homogeneous"),
    )
    assert len(non_gemini_homogeneous) == 1170
    assert len(gemini_homogeneous) == 260
    assert set(non_gemini_homogeneous).isdisjoint(gemini_homogeneous)
    for config_id in non_gemini_homogeneous:
        cfg = config_by_id[config_id]
        assert cfg["experiment_family"] in {"homogeneous_control", "homogeneous_adversary"}
        assert not (GEMINI_MODELS & set(cfg["models"]))
    for config_id in gemini_homogeneous:
        cfg = config_by_id[config_id]
        assert cfg["experiment_family"] == "homogeneous_adversary"
        assert GEMINI_MODELS & set(cfg["models"])

    non_gemini_heterogeneous = select_config_ids(
        configs,
        selection_preset_criteria("non_gemini_heterogeneous"),
    )
    gemini_heterogeneous = select_config_ids(
        configs,
        selection_preset_criteria("gemini_heterogeneous"),
    )
    assert non_gemini_heterogeneous
    assert gemini_heterogeneous
    assert set(non_gemini_heterogeneous).isdisjoint(gemini_heterogeneous)
    assert len(non_gemini_heterogeneous) == 757
    assert len(gemini_heterogeneous) == 543
    assert len(non_gemini_heterogeneous) + len(gemini_heterogeneous) == 1300
    for config_id in gemini_heterogeneous:
        cfg = config_by_id[config_id]
        assert cfg["experiment_family"] == "heterogeneous_random"
        assert GEMINI_MODELS & set(cfg["models"])


def test_selection_files_freeze_sparse_ids_and_heterogeneous_order(tmp_path):
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
    hetero_ids = select_config_ids(
        configs,
        selection_preset_criteria("gemini_heterogeneous"),
    )[:3]
    manifest = write_selection_files(
        results_root=tmp_path,
        configs=configs,
        selection_name="gemini_heterogeneous_hold",
        criteria=selection_preset_criteria("gemini_heterogeneous"),
        config_ids=hetero_ids,
    )

    assert read_config_id_file(selection_ids_path(tmp_path, "gemini_heterogeneous_hold")) == hetero_ids
    assert manifest["count"] == 3
    for config_id in hetero_ids:
        original = configs[config_id - 1]
        reloaded = json.loads(
            (tmp_path / "configs" / f"config_{config_id:04d}.json").read_text(encoding="utf-8")
        )
        assert reloaded["models"] == original["models"]
        assert list(reloaded["agent_model_map"].values()) == original["models"]
        assert reloaded["model_order"] == "sampled_random_order"


def test_full_games123_pure_random_heterogeneous_strategy_still_validates(tmp_path):
    configs = build_configs(
        results_root=tmp_path,
        master_seed=20260427,
        max_rounds=10,
        discussion_turns=2,
        max_tokens_voting=None,
        heterogeneous_sampling_strategy=HETEROGENEOUS_STRATEGY_PURE_RANDOM,
    )
    validation = validate_configs(configs)
    assert validation["valid"], validation["errors"][:10]
    hetero = [
        cfg for cfg in configs
        if cfg["experiment_family"] == "heterogeneous_random"
    ]
    assert {cfg["heterogeneous_sampling_strategy"] for cfg in hetero} == {
        HETEROGENEOUS_STRATEGY_PURE_RANDOM
    }
    assert all("stratum_index" not in cfg for cfg in hetero)


def test_selection_slurm_uses_selected_config_id_file_and_queue_mapping(tmp_path, monkeypatch):
    sbatch_path = write_slurm_file(
        tmp_path,
        {"slurm_time": "08:00:00", "max_concurrent": 50},
    )
    sbatch_text = sbatch_path.read_text(encoding="utf-8")
    assert "SELECTED_CONFIG_IDS_FILE" in sbatch_text
    assert 'sed -n "${SLURM_ARRAY_TASK_ID}p"' in sbatch_text

    def fake_check_output(*args, **kwargs):
        return "\n".join([
            "123_1|PENDING|0:00|Priority|fg123",
            "123_2|RUNNING|0:10|della-h01|fg123",
        ])

    monkeypatch.setattr(batch.subprocess, "check_output", fake_check_output)
    states = squeue_array_states([
        {"job_id": "123", "config_ids": [77, 1042]},
    ])
    assert states[77]["state"] == "PENDING"
    assert states[1042]["state"] == "RUNNING"
    assert states[1042]["task_id"] == "2"


def test_submit_selection_dry_run_skips_successful_configs_by_default(tmp_path):
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
    ids = select_config_ids(
        configs,
        selection_preset_criteria("extended_derisk_n10_amazon_gpt4o"),
    )[:2]
    write_selection_files(
        results_root=tmp_path,
        configs=configs,
        selection_name="skip_success_check",
        criteria=selection_preset_criteria("extended_derisk_n10_amazon_gpt4o"),
        config_ids=ids,
    )
    write_valid_result(configs[ids[0] - 1])

    cmd_submit_selection(
        Namespace(
            results_root=str(tmp_path),
            selection_name="skip_success_check",
            selection_file=None,
            max_concurrent=50,
            rerun_existing=False,
            dry_run=True,
        )
    )

    submission_files = list((tmp_path / "submissions").glob("skip_success_check_*_submission.json"))
    assert len(submission_files) == 1
    payload = json.loads(submission_files[0].read_text(encoding="utf-8"))
    assert payload["selected_count"] == 2
    assert payload["skipped_existing_count"] == 1
    assert payload["submitted_count"] == 1
    assert payload["skipped_existing_config_ids"] == [ids[0]]
    assert payload["submitted_config_ids"] == [ids[1]]


def test_submit_selection_dry_run_does_not_skip_incomplete_result_file(tmp_path):
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
    ids = select_config_ids(
        configs,
        selection_preset_criteria("extended_derisk_n10_amazon_gpt4o"),
    )[:2]
    write_selection_files(
        results_root=tmp_path,
        configs=configs,
        selection_name="invalid_success_check",
        criteria=selection_preset_criteria("extended_derisk_n10_amazon_gpt4o"),
        config_ids=ids,
    )
    completed_output = config_output_dir(configs[ids[0] - 1])
    completed_output.mkdir(parents=True, exist_ok=True)
    (completed_output / "experiment_results.json").write_text(
        json.dumps({"config": {"config_id": ids[0]}, "final_utilities": {}}),
        encoding="utf-8",
    )

    assert batch.validate_result_file(
        configs[ids[0] - 1],
        completed_output / "experiment_results.json",
    ).startswith("missing final_utilities")

    cmd_submit_selection(
        Namespace(
            results_root=str(tmp_path),
            selection_name="invalid_success_check",
            selection_file=None,
            max_concurrent=50,
            rerun_existing=False,
            dry_run=True,
        )
    )

    submission_files = list((tmp_path / "submissions").glob("invalid_success_check_*_submission.json"))
    assert len(submission_files) == 1
    payload = json.loads(submission_files[0].read_text(encoding="utf-8"))
    assert payload["selected_count"] == 2
    assert payload["skipped_existing_count"] == 0
    assert payload["submitted_count"] == 2
    assert payload["skipped_existing_config_ids"] == []
    assert payload["submitted_config_ids"] == ids
