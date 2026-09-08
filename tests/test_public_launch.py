"""Public launch tests use doubles only at the paid provider boundary."""

import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from strong_models_experiment.cli import main
from strong_models_experiment.launch.plan import build_plan, validate_plan, catalog, resolve_model
from strong_models_experiment.launch.providers import (RequestJournal, UnknownRequestOutcome, build_payload,
                                                       doctor, load_credentials, parse_response, redact)
from strong_models_experiment.launch.schema import LaunchRequest, strict_json
from strong_models_experiment.launch.store import (atomic_json, read_json, new_attempt, create_root,
                                                  status_report, validate_complete, lock)


def request(preset="two-player", **kwargs):
    if preset in ("two-player", "two-player-llama", "homogeneous-adversary"):
        kwargs.setdefault("adversary", "gpt-4o-mini-2024-07-18")
    if preset == "homogeneous":
        kwargs.setdefault("model", "gpt-5-nano")
    if preset == "ttc":
        kwargs.setdefault("family", "gpt5")
    return LaunchRequest(preset, **kwargs)


@pytest.mark.parametrize("preset", ["two-player", "two-player-llama", "homogeneous", "heterogeneous", "homogeneous-adversary", "ttc", "team"])
@pytest.mark.parametrize("game", ["game1", "game2", "game3"])
def test_all_family_plans(preset, game):
    if preset == "team" and game != "game1":
        with pytest.raises(ValueError, match="Game 1"):
            build_plan(request(preset, game=game))
        return
    plan = build_plan(request(preset, game=game, seed=0))
    assert validate_plan(plan) == plan
    assert len(plan["runs"]) == (4 if preset == "ttc" else 1)
    for run in plan["runs"]:
        assert run["engine"]["random_seed"] == 0
        assert run["engine"]["parallel_phases"] is False
        assert len(run["seats"]) == run["engine"]["n_agents"]
        assert run["failure_policy"]["provider_attempts"] == 1
        assert run["failure_policy"]["action_repairs"] == 0
    if preset == "heterogeneous":
        assert len(set(s["name"] for s in plan["runs"][0]["seats"])) == 4
        assert len(plan["runs"][0]["sampling"]["pool"]) == 24


@pytest.mark.parametrize("family,provider,efforts", [
    ("gpt5", "openai", ["minimal", "low", "medium", "high"]),
    ("claude", "anthropic", ["low", "medium", "high", "max"]),
    ("gemini", "openrouter", ["minimal", "low", "medium", "high"]),
])
def test_ttc_routes_and_efforts(family, provider, efforts):
    plan = build_plan(request("ttc", family=family))
    assert [r["effort"] for r in plan["runs"]] == efforts
    assert all(r["seats"][0]["provider"] == provider for r in plan["runs"])
    assert len({json.dumps(r["engine"], sort_keys=True) for r in plan["runs"]}) == 1


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": True}, {"rounds": 0}, {"discount": float("nan")},
    {"rho": 0.5}, {"model": "gpt-5-nano"}, {"family": "gpt5"}, {"agents": 4},
    {"competition": float("inf")}, {"max_tokens": 0}, {"stratum": 2}, {"adversary": "not-a-model"},
])
def test_invalid_input_rejected(kwargs):
    with pytest.raises(ValueError):
        build_plan(request(**kwargs))


def test_rho_feasibility():
    with pytest.raises(ValueError, match="infeasible"):
        build_plan(request("homogeneous", game="game2", agents=4, rho=-1))


def test_plan_tampering_rejected():
    plan = build_plan(request())
    plan["runs"][0]["seats"][0]["endpoint"] = "https://untrusted.invalid"
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_strict_json():
    for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
        with pytest.raises(ValueError):
            strict_json(text)


def test_offline_commands_do_not_import_engine():
    code = "from strong_models_experiment.cli import main; import sys; main(['plan','two-player','--adversary','gpt-4o-mini-2024-07-18']); assert 'strong_models_experiment.experiment' not in sys.modules; assert 'aiohttp' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_credentials_file_is_explicit_private_and_not_executed(tmp_path):
    path = tmp_path / "private.env"
    path.write_text("OPENAI_API_KEY='test-key'\n")
    path.chmod(0o600)
    assert load_credentials(path, {"ANTHROPIC_API_KEY": "shell-key"}) == {"OPENAI_API_KEY": "test-key"}
    path.write_text("OPENAI_API_KEY=x\nNEGOTIATION_DISABLE_CONTEXT_COMPACTION=1\n")
    with pytest.raises(ValueError, match="unsupported"):
        load_credentials(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        load_credentials(path)


def test_missing_keys_are_not_authentication_success():
    report = doctor(build_plan(request("two-player-llama")), {})
    assert report["missing_keys"] == ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    assert not report["authentication_tested"]


def test_environment_does_not_override_plan(monkeypatch):
    from strong_models_experiment.launch.execute import worker_environment
    monkeypatch.setenv("NEGOTIATION_MAX_HISTORY_MESSAGES", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid")
    monkeypatch.setenv("PYTHONPATH", "/wrong")
    env = worker_environment({"OPENAI_API_KEY": "private"})
    assert "NEGOTIATION_MAX_HISTORY_MESSAGES" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "PYTHONPATH" not in env
    assert env["OPENROUTER_PROVIDER_FALLBACK"] == "0"


def test_payloads_keep_native_controls():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    nano = build_payload(resolve_model("gpt-5-nano"), messages, 1000)
    assert nano["reasoning_effort"] == "low" and nano["max_completion_tokens"] == 1000
    assert "temperature" not in nano
    claude = build_payload(resolve_model("claude-sonnet-4-6-effort-max"), messages, 16384)
    assert claude["output_config"] == {"effort": "max"} and claude["thinking"] == {"type": "adaptive"}
    assert claude["system"] == "rules" and "extra_body" not in claude
    gemini = build_payload(resolve_model("gemini-3-flash-thinking-high"), messages, 16384)
    assert gemini["reasoning"]["effort"] == "high"


@pytest.mark.parametrize("content,finish", [(None, "stop"), ("", "stop"), ("partial", "length")])
def test_reasoning_is_never_an_answer(content, finish):
    with pytest.raises(Exception, match="final answer|incomplete"):
        parse_response("openrouter", {"choices": [{"message": {"content": content, "reasoning": "private analysis"}, "finish_reason": finish}]})


def test_secret_redaction():
    assert redact({"error": "failed key exact-secret"}, {"OPENAI_API_KEY": "exact-secret"}) == {"error": "failed key [REDACTED]"}


def test_captain_requires_explicit_choice():
    from strong_models_experiment.launch.runtime import validate_ballot
    validate_ballot('{"selected_proposal_number": null}', [1, 2], captain=True)
    for raw in ('{}', '{"selected_proposal_number": true}', '{"selected_proposal_number": 1.5}'):
        with pytest.raises(Exception):
            validate_ballot(raw, [1, 2], captain=True)


def test_lock_excludes_another_process(tmp_path):
    with lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with lock(tmp_path):
                pass


def mock_provider(monkeypatch, *, reject=False, failure=None):
    calls = []

    async def post(self, seat, payload, context_record):
        # Unit-test substitute for the nondeterministic, billed HTTP boundary.
        assert (self.attempt / "initial_state.json").exists()
        if self.failure is not None:
            raise self.failure
        calls.append(context_record)
        if failure is not None:
            self.stop(failure)
            raise failure
        phase = context_record["phase"]
        state = self.game_state
        n = len(self.game_state.get("agent_preferences", self.game_state.get("agent_positions", self.game_state.get("agent_valuations", {}))))
        ids = [f"Agent_{i + 1}" for i in range(n)]
        captain = "You are the captain. Resolve" in json.dumps(payload)
        if phase == "proposal" or captain:
            if state["game_type"] == "item_allocation":
                content = {"allocation": {aid: list(range(len(state["items"]))) if i == 0 else [] for i, aid in enumerate(ids)}}
            elif state["game_type"] == "diplomatic_treaty":
                content = {"agreement": [50] * state["n_issues"]}
            else:
                content = {"contributions": [0] * state["m_projects"]}
        elif phase == "binding_ballot":
            content = {"selected_proposal_number": None if reject else 1}
        elif phase in ("private_voting", "voting"):
            # The JSON example covers only two proposals, but the prompt lists
            # every proposal. Respond to the complete list, including n > 2.
            import re
            prompt = payload["messages"][-1]["content"]
            numbers = sorted({int(v) for v in re.findall(r'PROPOSAL #(\d+)', prompt)} |
                             {int(v) for v in re.findall(r'"proposal_number"\s*:\s*(\d+)', prompt)})
            content = {"votes": [{"proposal_number": number, "vote": "reject" if reject else "accept"} for number in numbers]}
            if state["game_type"] == "co_funding":
                content = {"vote": "reject" if reject else "accept"}
        else:
            content = {"reasoning": "unit-test boundary response"}
        response = {"model": seat["model_id"], "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}], "usage": None}
        if seat["provider"] == "anthropic":
            response = {"model": seat["model_id"], "content": [{"type": "text", "text": json.dumps(content)}], "stop_reason": "end_turn"}
        return response, "requests/unit-test-boundary"

    monkeypatch.setattr(RequestJournal, "post", post)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-credential")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-credential")
    monkeypatch.setenv("NEGOTIATION_DISABLE_CONTEXT_COMPACTION", "1")
    return calls


def prepare_attempt(tmp_path, req):
    from strong_models_experiment.launch.execute import provenance
    plan = build_plan(req)
    root = create_root(tmp_path / "output", plan)
    attempt = new_attempt(root, plan["runs"][0], provenance())
    return root, plan, attempt


@pytest.mark.parametrize("preset,game", [("two-player", "game1"), ("two-player", "game2"),
                                         ("two-player", "game3"), ("team", "game1"),
                                         ("two-player-llama", "game1"), ("homogeneous", "game1"),
                                         ("heterogeneous", "game1"), ("homogeneous-adversary", "game1")])
@pytest.mark.parametrize("reject", [False, True])
def test_real_engine_with_only_provider_boundary_replaced(tmp_path, monkeypatch, preset, game, reject):
    from strong_models_experiment.launch.execute import run_worker
    calls = mock_provider(monkeypatch, reject=reject)
    root, plan, attempt = prepare_attempt(tmp_path, request(preset, game=game, rounds=1))
    rc = asyncio.run(run_worker(attempt))
    assert rc == 0, read_json(attempt / "status.json")
    result = read_json(attempt / "result.json")
    assert result["consensus_reached"] is (not reject)
    validate_complete(attempt, plan["runs"][0])
    assert status_report(root, plan)["complete"] == 1
    assert calls


def test_provider_failure_is_not_a_zero_payoff_result(tmp_path, monkeypatch):
    from strong_models_experiment.launch.execute import run_worker
    calls = mock_provider(monkeypatch, failure=UnknownRequestOutcome("unit-test connection interrupted"))
    root, plan, attempt = prepare_attempt(tmp_path, request(rounds=1))
    assert asyncio.run(run_worker(attempt)) == 1
    assert len(calls) == 1
    assert read_json(attempt / "status.json")["state"] == "unknown_request_outcome"
    assert not (attempt / "result.json").exists()
    assert status_report(root, plan)["complete"] == 0


def test_result_tampering_is_not_reused(tmp_path, monkeypatch):
    from strong_models_experiment.launch.execute import run_worker
    mock_provider(monkeypatch)
    _, plan, attempt = prepare_attempt(tmp_path, request(rounds=1))
    assert asyncio.run(run_worker(attempt)) == 0
    atomic_json(attempt / "result.json", {"final_utilities": {}})
    with pytest.raises(ValueError, match="checksum"):
        validate_complete(attempt, plan["runs"][0])


@pytest.mark.parametrize("game", ["game1", "game2", "game3"])
def test_ttc_keeps_same_initial_state(tmp_path, monkeypatch, game):
    from strong_models_experiment.launch.execute import provenance, run_worker
    mock_provider(monkeypatch)
    plan = build_plan(request("ttc", game=game, rounds=1, seed=0))
    root = create_root(tmp_path / "output", plan)
    states = []
    for run in plan["runs"]:
        attempt = new_attempt(root, run, provenance())
        assert asyncio.run(run_worker(attempt)) == 0, read_json(attempt / "status.json")
        states.append(read_json(attempt / "initial_state.json"))
    assert all(state == states[0] for state in states)


@pytest.mark.parametrize("game,key,values", [
    ("game1", "allocation", {"Agent_1": [0, 1, 2, 3, 4.9], "Agent_2": []}),
    ("game1", "allocation", {"Agent_1": [0, 1, 2, 3, 4, 999], "Agent_2": []}),
    ("game2", "agreement", [50, 50]),
    ("game2", "agreement", [50, 50, 50, 50, 101]),
    ("game3", "contributions", [0, 0]),
    ("game3", "contributions", [-1, 0, 0, 0, 0]),
])
def test_raw_actions_fail_before_legacy_normalization(tmp_path, monkeypatch, game, key, values):
    from strong_models_experiment.launch.execute import run_worker
    from strong_models_experiment.launch.runtime import HostedAgent
    calls = mock_provider(monkeypatch)
    original = HostedAgent._call_llm_api

    async def invalid_response(self, messages, **kwargs):
        response = await original(self, messages, **kwargs)
        if self.call_context["phase"] == "proposal":
            response.content = json.dumps({key: values})
        return response

    # This replaces only the external response, not production parsing or rules.
    monkeypatch.setattr(HostedAgent, "_call_llm_api", invalid_response)
    _, _, attempt = prepare_attempt(tmp_path, request(game=game, rounds=1))
    assert asyncio.run(run_worker(attempt)) == 1
    assert read_json(attempt / "status.json")["state"] == "failed"
    assert sum(c["phase"] == "proposal" for c in calls) == 1
    assert not (attempt / "result.json").exists()


@pytest.mark.parametrize("status", [200, 401, 503])
def test_http_journal_records_before_dispatch_without_retry(tmp_path, status):
    # A local HTTP fixture tests transport and persistence, not live authentication.
    from aiohttp import web

    async def scenario():
        journal = RequestJournal(tmp_path, {"OPENAI_API_KEY": "local-http-test-key"}, 1)
        seen = []

        async def handler(request):
            records = list((tmp_path / "requests").glob("*.json"))
            assert len(records) == 1
            assert read_json(records[0])["state"] == "dispatching"
            assert request.headers["Authorization"] == "Bearer local-http-test-key"
            seen.append(await request.json())
            return web.json_response({"model": "test-boundary", "choices": []}, status=status)

        app = web.Application()
        app.router.add_post("/request", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        seat = {**resolve_model("gpt-5-nano"), "endpoint": f"http://127.0.0.1:{port}/request"}
        try:
            if status == 200:
                await journal.post(seat, {"messages": []}, {})
            else:
                with pytest.raises(Exception):
                    await journal.post(seat, {"messages": []}, {})
                with pytest.raises(Exception):
                    await journal.post(seat, {"messages": []}, {})
            assert len(seen) == 1
            record = list((tmp_path / "requests").glob("*.json"))[0].read_text()
            assert "local-http-test-key" not in record
            if status == 503:
                assert isinstance(journal.failure, UnknownRequestOutcome)
        finally:
            await journal.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_resume_requires_explicit_unknown_outcome_decision(tmp_path, monkeypatch):
    from strong_models_experiment.launch.execute import execute
    from strong_models_experiment.launch.store import finish
    root, plan, attempt = prepare_attempt(tmp_path, request())
    finish(attempt, plan["runs"][0], "unknown_request_outcome", "test interruption")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="retry-failed"):
        execute(plan, root, {"OPENAI_API_KEY": "test-only"}, resume=True)
    with pytest.raises(ValueError, match="unknown"):
        execute(plan, root, {"OPENAI_API_KEY": "test-only"}, resume=True, retry_failed=True)


def test_resume_complete_never_starts_a_worker(tmp_path, monkeypatch):
    from strong_models_experiment.launch.execute import execute, run_worker
    mock_provider(monkeypatch)
    root, plan, attempt = prepare_attempt(tmp_path, request(rounds=1))
    assert asyncio.run(run_worker(attempt)) == 0

    def forbidden(*args, **kwargs):
        pytest.fail("A verified complete attempt must not start a worker")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Avoid replacing git's own diagnostic subprocess with the worker assertion.
    from strong_models_experiment.launch import execute as module
    recorded_provenance = read_json(attempt / "provenance.json")
    monkeypatch.setattr(module, "provenance", lambda: recorded_provenance)
    assert execute(plan, root, {"OPENAI_API_KEY": "test-only"}, resume=True) == 0


def test_worker_bootstrap_outside_checkout_without_credentials(tmp_path):
    from strong_models_experiment.launch.execute import worker_environment
    _, _, attempt = prepare_attempt(tmp_path, request(rounds=1))
    # The parent doctor normally rejects missing credentials. Invoking the worker
    # directly here tests real subprocess startup and output relocation without
    # permitting any request to reach a provider.
    result = subprocess.run(
        [sys.executable, "-m", "strong_models_experiment.launch.execute", str(attempt)],
        cwd=tmp_path, env=worker_environment({}), capture_output=True, text=True, timeout=30,
    )
    status = read_json(attempt / "status.json")
    assert result.returncode == 1
    assert status["state"] == "failed", status
    assert "OPENAI_API_KEY" in status["error"]
    assert (attempt / "initial_state.json").exists()
    assert list((attempt / "requests").glob("*.json")) == []
    assert not (attempt / "result.json").exists()
