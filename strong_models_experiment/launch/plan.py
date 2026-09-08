"""Resolve small experiment families without provider calls or output writes."""

import copy
from functools import lru_cache
import hashlib
from importlib.resources import files
from itertools import combinations, islice
import math
import random

from .schema import GAMES, PHASES, PROTOCOL, TEAM_PROTOCOL, LaunchRequest, digest, strict_json

ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}
KEYS = {provider: provider.upper() + "_API_KEY" for provider in ENDPOINTS}


def catalog():
    return strict_json(files("strong_models_experiment").joinpath("resources/launch_models_v1.json").read_text())


def resource_hashes():
    root = files("strong_models_experiment").joinpath("resources")
    return {name: hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest()
            for name in ("launch_models_v1.json", "model_context_2026_03_31.md")}


@lru_cache(maxsize=32)
def sample_heterogeneous(n, seed, stratum):
    """Sample uniformly within one equal-width population-Elo-SD stratum.

    Enumerate only the requested group size in bounded-memory chunks.
    This is a new seeded draw, not the historical paper's full-grid RNG sequence.
    """
    import numpy as np

    data = catalog()
    pool = data["heterogeneous_pool"]
    elos = np.array([data["elos"][name] for name in pool], dtype=float)

    def chunks():
        iterator = combinations(range(len(pool)), n)
        while chunk := list(islice(iterator, 8192)):
            indices = np.asarray(chunk)
            yield indices, elos[indices].std(axis=1, ddof=0)

    low, high = math.inf, -math.inf
    for _, deviations in chunks():
        low, high = min(low, float(deviations.min())), max(high, float(deviations.max()))
    width = (high - low) / 5
    if width <= 0:
        raise ValueError("The heterogeneous pool has no Elo-SD spread")
    rng = random.Random(seed)
    selected, count = None, 0
    for indices, deviations in chunks():
        bins = np.minimum(4, ((deviations - low) / width).astype(int))
        eligible = indices[bins == stratum]
        # Reservoir sampling by chunk preserves equal probability per subset.
        old_count = count
        count += len(eligible)
        if len(eligible) and rng.randrange(count) >= old_count:
            selected = eligible[rng.randrange(len(eligible))].tolist()
    if selected is None:
        raise ValueError("Requested heterogeneous stratum is empty")
    names = [pool[index] for index in selected]
    random.Random(seed ^ 0x5A17).shuffle(names)
    return tuple(names), (low, high, count)


def resolve_model(name, max_tokens=None, team=False):
    data = catalog()
    if not isinstance(name, str) or name not in data["models"]:
        raise ValueError(f"Unknown public model {name!r}; use bargain models")
    raw = copy.deepcopy(data["models"][name])
    provider = raw["api_type"]
    extras = raw.get("custom_parameters", {})
    extras.pop("phase_token_cap_policy", None)
    extras.update(extras.pop("extra_body", {}))
    if "thinking_budget_tokens" in extras:
        extras["thinking"] = {"type": "enabled", "budget_tokens": extras.pop("thinking_budget_tokens")}
    if provider == "openai":
        if any(token in raw["model_id"] for token in ("gpt-5", "o3", "o1")):
            # Match the current native client's low default explicitly; do not
            # infer high effort from an analysis alias such as o3-mini-high.
            extras["reasoning_effort"] = raw.get("reasoning_effort", "low")
    if team and name == "gpt-5.4-high":
        provider = "openai"
        raw["model_id"] = "gpt-5.4"
        extras = {"reasoning_effort": "high"}
    if provider not in ENDPOINTS:
        raise ValueError(f"Unsupported public provider: {provider}")
    # These are explicit launch caps, not claims about provider availability.
    declared_cap = raw.get("max_tokens_default", 16384)
    budget = extras.get("thinking", {}).get("budget_tokens")
    if budget is not None:
        declared_cap = max(declared_cap, budget + 1024)
    cap = declared_cap if max_tokens is None else max_tokens
    if max_tokens is not None and cap > declared_cap:
        raise ValueError(f"{name} has a declared launch output cap of {declared_cap}")
    if budget is not None and cap <= budget:
        raise ValueError(f"{name} needs an output cap larger than its {budget}-token thinking budget")
    reasoning = provider == "openai" and "reasoning_effort" in extras
    temperature = None if reasoning or "thinking" in extras else raw["temperature"]
    return {
        "name": name, "model_id": raw["model_id"], "provider": provider,
        "endpoint": ENDPOINTS[provider], "credential_env": KEYS[provider],
        "temperature": temperature, "parameters": extras,
        "system_prompt": raw["system_prompt"],
        "phase_caps": {phase: min(cap, 32768) if phase == "voting" else cap for phase in PHASES},
        "input_limit_tokens": 32768,
    }


def build_plan(request: LaunchRequest):
    request.validate()
    preset = request.preset
    n = request.agents if request.agents is not None else (2 if preset in ("two-player", "two-player-llama", "ttc") else 4)
    sampling = None
    if preset == "homogeneous":
        groups = [[request.model] * n]
    elif preset == "heterogeneous":
        stratum = request.stratum if request.stratum is not None else 2
        names, (low, high, count) = sample_heterogeneous(n, request.seed, stratum)
        groups = [list(names)]
        sampling = {"method": "equal-width-elo-sd-v1", "stratum": stratum,
                    "strata": 5, "subset_count": count, "min_sd": low, "max_sd": high,
                    "draw_seed": request.seed, "order_seed": request.seed ^ 0x5A17,
                    "pool": catalog()["heterogeneous_pool"]}
    elif preset == "ttc":
        efforts = ("low", "medium", "high", "max") if request.family == "claude" else ("minimal", "low", "medium", "high")
        templates = {"gpt5": "gpt-5-{}-effort", "claude": "claude-sonnet-4-6-effort-{}",
                     "gemini": "gemini-3-flash-thinking-{}"}
        groups = [[templates[request.family].format(effort), "gpt-5-nano"] for effort in efforts]
    else:
        adversary = "gpt-5.4-high" if preset == "team" else request.adversary
        baseline = "llama-3.3-70b-instruct" if preset == "two-player-llama" else "gpt-5-nano"
        if adversary == baseline:
            raise ValueError("The adversary must differ from the baseline; use homogeneous for identical models")
        groups = [[adversary] + [baseline] * (n - 1)]
    if request.position == "last":
        groups = [group[1:] + group[:1] for group in groups]

    # The old Game 1 option is a requested cosine, despite its competition name.
    rho = 0.0 if request.rho is None else request.rho
    if request.game == "game2" and 2 * math.sin(math.pi * rho / 6) < -1 / (n - 1) - 1e-12:
        raise ValueError("rho is infeasible for this number of agents under the Gaussian-copula generator")
    config = {
        "game_type": GAMES[request.game], "n_agents": n,
        "m_items": 5 * n // 2, "n_issues": 5 if preset in ("two-player", "two-player-llama", "ttc") else 10,
        "m_projects": 5 * n // 2,
        "competition_level": 0.5 if request.competition is None else request.competition,
        "rho": rho, "theta": 1.0 if request.theta is None else request.theta,
        "alpha": 0.5 if request.alpha is None else request.alpha,
        "sigma": 0.6 if request.sigma is None else request.sigma,
        "c_min": 10.0, "c_max": 30.0, "pledge_mode": "individual",
        "t_rounds": request.rounds, "discussion_turns": request.discussion_turns,
        "gamma_discount": request.discount, "random_seed": request.seed,
        "model_order": "specified", "parallel_phases": False,
        "disable_discussion": False, "disable_thinking": False, "disable_reflection": False,
        "reasoning_config": {"budget": None, "phases": []}, "reasoning_token_budget": None,
        "access_config": {"k": 1, "phases": [], "agent_ids": [], "agent_index": 0},
        "cofunding_discussion_transparency": "own", "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True, "cofunding_time_discount": request.discount,
        "max_tokens_per_phase": 16384, "team_coordination": None, "fixed_agent_preferences": None,
    }
    config.update({f"max_tokens_{phase}": None for phase in PHASES})
    runs = []
    for index, names in enumerate(groups):
        engine = copy.deepcopy(config)
        seats = [{"agent_id": f"Agent_{i+1}",
                  "role": "participant" if preset in ("homogeneous", "heterogeneous") else
                          ("adversary" if i == (0 if request.position == "first" else n - 1) else "baseline"),
                  **resolve_model(name, request.max_tokens, team=preset == "team")}
                 for i, name in enumerate(names)]
        if preset == "team":
            members = [s["agent_id"] for s in seats if s["role"] == "baseline"]
            engine["team_coordination"] = {
                "enabled": n > 2, "protocol_version": TEAM_PROTOCOL, "member_ids": members,
                "captain_id": members[0], "rotate_captain_each_round": True,
                "share_full_team_preferences": True, "share_private_thinking": False,
                "share_private_voting": False, "share_reflection": False, "share_team_planning": True,
                "planning_turns": 3, "planning_max_tokens": min(request.max_tokens or 8192, 8192),
                "ballot_max_tokens": min(request.max_tokens or 4096, 4096),
                "max_action_repairs": 0, "hard_fail_on_action_error": True,
                "synthetic_actions_allowed": False, "formal_coalition_proposal": True,
                "binding_ballot": True, "singleton_policy": "no-team-treatment",
                "joint_objective": "maximize_expected_discounted_sum_nano_utility",
                "utility_evaluator": "environment", "utility_reporting_required": False,
                "tie_break": "seeded_existing_tabulation_rule",
            }
        body = {
            "protocol": PROTOCOL, "preset": preset, "engine": engine, "seats": seats,
            "resources": resource_hashes(), "sampling": sampling,
            "effort": efforts[index] if preset == "ttc" else None,
            "context_policy": {"version": "full-history-hard-limit-v1", "estimator": "chars-v1",
                               "chars_per_token": 3.0, "compaction": False},
            "failure_policy": {"provider_attempts": 1, "action_repairs": 0,
                               "provider_substitution": False, "synthetic_actions": False,
                               "unknown_request_outcome": "stop-and-require-explicit-restart"},
            "preference_policy": "existing-generators-v1-no-new-cosine-acceptance-test",
            "transport": "direct", "timeout_seconds": request.timeout,
        }
        runs.append({"run_id": digest(body), **body})
    body = {"schema_version": 1, "request": request.to_dict(), "runs": runs}
    return {"plan_id": digest(body), **body}


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "request", "runs", "plan_id"}:
        raise ValueError("Invalid plan fields")
    try:
        expected = build_plan(LaunchRequest(**plan["request"]))
    except TypeError as exc:
        raise ValueError("Invalid or unknown request fields") from exc
    if plan != expected:
        raise ValueError("Plan differs from its resolved inputs or current resource version")
    return plan
