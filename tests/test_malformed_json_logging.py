#!/usr/bin/env python3
"""Tests for centralized malformed JSON diagnostics."""

import json

from scripts.materialize_malformed_json_examples import materialize_jsonl
from strong_models_experiment.experiment import StrongModelsExperiment


def test_malformed_json_diagnostics_written_per_run_and_batch(tmp_path):
    """Invalid parse diagnostics should be mirrored to run JSON and batch JSONL."""
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "config_0001_game3"
    experiment = StrongModelsExperiment(output_dir=run_dir)
    experiment.current_experiment_id = "exp-test"
    experiment.current_config = {
        "config_id": 1,
        "game_label": "game3",
        "game_type": "co_funding",
        "experiment_family": "heterogeneous_random",
        "competition_id": "sigma0_5_alpha0_8",
        "n_agents": 10,
        "sigma": 0.5,
        "alpha": 0.8,
        "random_seed": 123,
    }

    raw_response = '{"contributions": [1, 2, 3], "reasoning": "unterminated'
    diagnostic = {
        "proposed_by": "Agent_1",
        "round": 2,
        "game_type": "co_funding",
        "proposal_repair_attempt": 1,
        "raw_response": raw_response,
        "raw_proposal": raw_response,
        "parse_error": {
            "type": "JSONDecodeError",
            "message": "Unterminated string",
        },
        "error_summary": "parse error",
        "will_retry": False,
        "hard_failed": True,
    }

    experiment._save_interaction(
        "Agent_1",
        "proposal_round_2_invalid_attempt_1",
        "repair prompt",
        json.dumps(diagnostic),
        round_num=2,
        model_name="gemini-2.5-pro",
    )

    run_json = run_dir / "monitoring" / "malformed_json_examples.json"
    batch_jsonl = batch_root / "monitoring" / "malformed_json_examples.jsonl"
    assert run_json.exists()
    assert batch_jsonl.exists()

    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert run_payload["count"] == 1
    example = run_payload["examples"][0]
    assert example["full_experiment_results_filepath"] == str(run_dir / "experiment_results.json")
    assert example["game"] == "game3"
    assert example["game_type"] == "co_funding"
    assert example["N"] == 10
    assert example["sigma"] == 0.5
    assert example["alpha"] == 0.8
    assert example["model"] == "gemini-2.5-pro"
    assert example["round"] == 2
    assert example["phase"] == "proposal"
    assert example["interaction_phase"] == "proposal_round_2_invalid_attempt_1"
    assert example["turn"] == 1
    assert example["raw_malformed_json"] == raw_response
    assert example["parse_error"]["type"] == "JSONDecodeError"

    jsonl_examples = [
        json.loads(line)
        for line in batch_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_examples) == 1
    assert jsonl_examples[0]["raw_malformed_json"] == raw_response

    materialized_path, materialized = materialize_jsonl(batch_root)
    assert materialized_path == batch_root / "monitoring" / "malformed_json_examples.json"
    assert materialized["count"] == 1
    assert materialized["examples"][0]["raw_malformed_json"] == raw_response
