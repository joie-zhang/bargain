from pathlib import Path

import pandas as pd
import pytest

from scripts.recover_ttc_reasoning_tokens import (
    ProxyRecord,
    aggregate_recovery,
    count_cache_key,
    match_proxy_group,
    proxy_direct_validation_records,
    saved_text_fidelity,
    sha256_text,
)


def _interaction(
    *,
    timestamp: float,
    response: str,
    index: int,
) -> dict:
    return {
        "interaction_timestamp": timestamp,
        "interaction_response_sha256": sha256_text(response),
        "run_path": f"/tmp/run-{index}.json",
        "interaction_index": index,
    }


def _proxy(
    *,
    suffix: str,
    timestamp: float,
    response: str,
) -> ProxyRecord:
    return ProxyRecord(
        suffix=suffix,
        request_timestamp=timestamp,
        effort="high",
        prompt_sha256="prompt",
        completion_tokens=100,
        direct_reasoning_tokens=None,
        response_sha256=sha256_text(response),
        response_text=response,
        request_path=Path(f"/tmp/request_{suffix}.json"),
        response_path=Path(f"/tmp/response_{suffix}.json"),
    )


def test_saved_text_fidelity_marks_structured_phases() -> None:
    assert (
        saved_text_fidelity("private_thinking_agent_0")
        == "parsed_or_canonicalized_interaction_text"
    )
    assert (
        saved_text_fidelity("proposal_agent_1")
        == "parsed_or_canonicalized_interaction_text"
    )
    assert (
        saved_text_fidelity("voting_agent_2")
        == "parsed_or_canonicalized_interaction_text"
    )
    assert saved_text_fidelity("discussion_agent_0") == "likely_raw_interaction_text"


def test_proxy_match_prefers_exact_response_hash() -> None:
    interaction = _interaction(timestamp=105.0, response="raw response", index=0)
    wrong = _proxy(suffix="wrong", timestamp=104.0, response="other response")
    exact = _proxy(suffix="exact", timestamp=100.0, response="raw response")

    assignments = match_proxy_group([interaction], [wrong, exact])

    assert len(assignments) == 1
    assert assignments[0][1].suffix == "exact"
    assert assignments[0][2] == "exact_response_hash"


def test_proxy_match_uses_global_one_to_one_timestamp_assignment() -> None:
    interactions = [
        _interaction(timestamp=101.0, response="parsed-a", index=0),
        _interaction(timestamp=199.0, response="parsed-b", index=1),
    ]
    candidates = [
        _proxy(suffix="early", timestamp=100.0, response="raw-a"),
        _proxy(suffix="late", timestamp=200.0, response="raw-b"),
    ]

    assignments = match_proxy_group(interactions, candidates)
    by_interaction = {
        interaction["interaction_index"]: proxy.suffix
        for interaction, proxy, _, _ in assignments
    }

    assert by_interaction == {0: "early", 1: "late"}
    assert len({proxy.suffix for _, proxy, _, _ in assignments}) == 2


def test_proxy_match_rejects_candidate_shortage() -> None:
    interactions = [
        _interaction(timestamp=100.0, response="parsed-a", index=0),
        _interaction(timestamp=101.0, response="parsed-b", index=1),
    ]
    candidates = [_proxy(suffix="only", timestamp=100.0, response="raw")]

    with pytest.raises(RuntimeError, match="Too few proxy candidates"):
        match_proxy_group(interactions, candidates)


def test_proxy_direct_validation_requires_a_direct_reasoning_field() -> None:
    missing = _proxy(suffix="missing", timestamp=100.0, response="raw")
    direct = ProxyRecord(
        **{
            **missing.__dict__,
            "suffix": "direct",
            "direct_reasoning_tokens": 80,
        }
    )

    records = proxy_direct_validation_records([missing, direct])

    assert len(records) == 1
    assert records[0]["direct_reasoning_tokens"] == 80
    assert records[0]["provider_output_tokens_inclusive"] == 100
    assert records[0]["visible_text_fidelity"] == "exact_raw_provider_output"


def test_aggregate_preserves_signed_negative_residuals() -> None:
    frame = pd.DataFrame(
        [
            {
                "seed": 42,
                "family": "gemini-3-flash",
                "reasoning_effort": "minimal",
                "provider_output_tokens_inclusive": 10,
                "estimated_visible_output_tokens": 12,
                "reconstructed_hidden_tokens": -2,
                "negative_hidden_residual": True,
                "visible_text_fidelity": "likely_raw_interaction_text",
            },
            {
                "seed": 42,
                "family": "gemini-3-flash",
                "reasoning_effort": "minimal",
                "provider_output_tokens_inclusive": 20,
                "estimated_visible_output_tokens": 15,
                "reconstructed_hidden_tokens": 5,
                "negative_hidden_residual": False,
                "visible_text_fidelity": "exact_raw_provider_output",
            },
        ]
    )

    summary = aggregate_recovery(
        frame, ["seed", "family", "reasoning_effort"]
    ).iloc[0]

    assert summary["reconstructed_hidden_tokens_sum"] == 3
    assert summary["reconstructed_hidden_tokens_per_call"] == 1.5
    assert summary["negative_residual_calls"] == 1


def test_count_cache_key_includes_provider_model_and_role() -> None:
    base = count_cache_key("google", "model-a", "model", "same text")

    assert base != count_cache_key("anthropic", "model-a", "model", "same text")
    assert base != count_cache_key("google", "model-b", "model", "same text")
    assert base != count_cache_key("google", "model-a", "user", "same text")
