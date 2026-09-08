"""Strict hosted agents around the existing prompts, phases, and game rules."""

import copy
import math
import time

from negotiation.llm_agents import AgentResponse, BaseLLMAgent, LLMConfig, NonRetryableLLMError
from strong_models_experiment.phases.phase_handlers import PhaseHandler

from .providers import build_payload, parse_response
from .schema import TEAM_PROTOCOL, finite, strict_json


class InvalidAction(NonRetryableLLMError):
    pass


def json_object(content):
    try:
        value = strict_json(content)
    except (ValueError, TypeError) as exc:
        raise InvalidAction("Action must be one valid JSON object") from exc
    if not isinstance(value, dict):
        raise InvalidAction("Action must be a JSON object")
    return value


def validate_proposal_content(content, agent_id, state, environment, agent_ids):
    payload = json_object(content)
    game = state["game_type"]
    if game == "item_allocation":
        allocation = payload.get("allocation")
        if not isinstance(allocation, dict) or set(allocation) != set(agent_ids):
            raise InvalidAction("Allocation must name exactly all seats")
        indices = []
        for values in allocation.values():
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise InvalidAction("Allocation indices must be integers")
            indices.extend(values)
        if sorted(indices) != list(range(len(state["items"]))):
            raise InvalidAction("Each item must be allocated exactly once")
    elif game == "diplomatic_treaty":
        values = payload.get("agreement")
        # Current prompts use percentages; the existing parser also accepts [0,1].
        if not isinstance(values, list) or len(values) != state["n_issues"]:
            raise InvalidAction("Agreement has the wrong number of issues")
        if not all(finite(v) and 0 <= v <= 100 for v in values):
            raise InvalidAction("Agreement values must be finite and in [0,100]")
    elif game == "co_funding":
        values = payload.get("contributions")
        if not isinstance(values, list) or len(values) != state["m_projects"]:
            raise InvalidAction("Contributions have the wrong number of projects")
        if not all(finite(v) and v >= 0 for v in values):
            raise InvalidAction("Contributions must be finite and nonnegative")
        if sum(values) > state["agent_budgets"][agent_id] + 1e-9:
            raise InvalidAction("Contributions exceed the agent's budget")
    else:
        raise InvalidAction("Unsupported game for proposal validation")
    candidate = environment.parse_proposal(content, agent_id, copy.deepcopy(state), agent_ids)
    if "parse_error" in candidate or "validation_error" in candidate or not environment.validate_proposal(candidate, state):
        raise InvalidAction("The game environment rejected the proposal")


def validate_ballot(content, numbers, captain=False):
    payload = json_object(content)
    if captain:
        if "selected_proposal_number" not in payload:
            raise InvalidAction("Captain ballot must include selected_proposal_number, including for an explicit null decision")
        selected = payload["selected_proposal_number"]
        if selected is not None and (type(selected) is not int or selected not in numbers):
            raise InvalidAction("Invalid captain choice")
        return
    votes = payload.get("votes")
    if not isinstance(votes, list) or len(votes) != len(numbers):
        raise InvalidAction("Ballot must include exactly one vote per proposal")
    seen = set()
    for vote in votes:
        if not isinstance(vote, dict):
            raise InvalidAction("Vote must be an object")
        number = vote.get("proposal_number")
        if type(number) is not int or number not in numbers or number in seen:
            raise InvalidAction("Unknown or duplicate proposal number")
        if vote.get("vote") not in ("accept", "reject"):
            raise InvalidAction("Vote must be exactly accept or reject")
        seen.add(number)


class StrictPhaseHandler(PhaseHandler):
    BINDING_TEAM_PROTOCOL_VERSION = TEAM_PROTOCOL

    async def _run_agent_tasks_in_order(self, agents, task_fn):
        results = []
        for index, agent in enumerate(agents):
            result = await task_fn(index, agent)
            if agent.journal.failure is not None:
                raise NonRetryableLLMError(str(agent.journal.failure))
            if isinstance(result, dict) and result.get("error") is not None:
                raise result["error"]
            results.append(result)
        return results

    @staticmethod
    def neutral(text):
        return (text.replace("GPT-5.4 adversary", "non-team participant")
                .replace("non-team adversary", "non-team participant")
                .replace("adversary", "non-team participant")
                .replace("GPT-5.4", "the non-team participant")
                .replace("Nano", "coordinated"))

    def build_team_system_objective(self, agent_id):
        return self.neutral(super().build_team_system_objective(agent_id))

    def build_team_coordination_briefing(self, *args, **kwargs):
        return self.neutral(super().build_team_coordination_briefing(*args, **kwargs))

    async def _generate_response_with_access(self, agent, context, prompt, *, phase):
        if self.is_binding_team_protocol():
            prompt = self.neutral(prompt)
        return await super()._generate_response_with_access(agent, context, prompt, phase=phase)

    def _build_synthetic_default_proposal(self, **kwargs):
        raise InvalidAction("The public protocol does not permit substitute proposals")


class HostedAgent(BaseLLMAgent):
    def __init__(self, seat, journal, run):
        config = LLMConfig(
            model_type=seat["model_id"], temperature=seat["temperature"] if seat["temperature"] is not None else 1.0,
            max_tokens=seat["phase_caps"]["default"], max_retries=1,
            timeout=run["timeout_seconds"], system_prompt=seat["system_prompt"],
            custom_parameters={"phase_token_caps": seat["phase_caps"]},
        )
        config._actual_model_id = seat["model_id"]
        super().__init__(seat["agent_id"], config)
        self.seat = seat
        self.journal = journal
        self.run = run
        self.model_name = seat["model_id"]
        self.call_context = {}

    def get_model_info(self):
        return {"model_name": self.seat["model_id"], "provider": self.seat["provider"],
                "model_type": self.seat["name"], "agent_id": self.agent_id}

    def _maybe_compact_context_messages(self, *, messages, **kwargs):
        # Full context is retained until the saved hard input limit. This policy
        # is independent of optional tokenizer installs and missing TTC aliases.
        return messages, None

    async def _call_llm_api(self, messages, **kwargs):
        if self.journal.failure is not None:
            raise NonRetryableLLMError("Attempt already failed; no further provider calls are permitted")
        started = time.time()
        try:
            estimated = sum(math.ceil(len(m["content"]) / 3) + 4 for m in messages)
            if estimated > self.seat["input_limit_tokens"]:
                raise NonRetryableLLMError("Prompt exceeds the saved input limit; no provider call was made")
            payload = build_payload(self.seat, messages, self.config.max_tokens)
            data, reference = await self.journal.post(self.seat, payload, self.call_context)
            content, metadata = parse_response(self.seat["provider"], data)
            metadata.update(request_record=reference, requested_model=self.seat["model_id"],
                            provider=self.seat["provider"], transport="direct")
            usage = metadata["usage"]
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if self.seat["provider"] == "anthropic" and isinstance(usage, dict):
                if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
                    total = usage["input_tokens"] + usage["output_tokens"]
            return AgentResponse(content=content, model_used=self.seat["model_id"],
                                 response_time=time.time() - started, tokens_used=total, metadata=metadata)
        except Exception as exc:
            self.journal.stop(exc)
            # All legacy phase handlers recognize this as a hard failure.
            raise NonRetryableLLMError(str(exc)) from None

    def validate_action(self, content, context):
        state = self.journal.game_state
        if state is None:
            raise InvalidAction("Initial game state was not recorded")
        phase = context.turn_type
        team = self.run["engine"]["team_coordination"]
        captain = None
        if team and team["enabled"]:
            members = team["member_ids"]
            captain = members[(members.index(team["captain_id"]) + context.current_round - 1) % len(members)]
        if phase == "proposal" or (phase == "team_planning_turn_3" and self.agent_id == captain):
            validate_proposal_content(content, self.agent_id, state, self.journal.game_environment, context.agents)
        elif phase in ("private_voting", "binding_ballot", "voting"):
            if state["game_type"] == "co_funding":
                payload = json_object(content)
                if payload.get("vote") not in ("accept", "reject"):
                    raise InvalidAction("Vote must be exactly accept or reject")
                return
            numbers = list(range(1, (1 if state["game_type"] == "co_funding" else len(context.current_proposals)) + 1))
            if not numbers:
                raise InvalidAction("Voting context is missing proposals")
            validate_ballot(content, numbers, captain=phase == "binding_ballot")

    async def generate_response(self, context, prompt):
        self.call_context = {"phase": context.turn_type, "round": context.current_round, "agent_id": self.agent_id}
        try:
            response = await super().generate_response(context, prompt)
            # Validate the original answer before permissive legacy parsers can
            # pad, truncate, clamp, drop entries, or supply default decisions.
            self.validate_action(response.content, context)
            return response
        except Exception as exc:
            self.journal.stop(exc)
            raise NonRetryableLLMError(str(exc)) from None

    async def think_strategy(self, prompt, context):
        response = await self.generate_response(context, prompt)
        # Keep free-form private analysis verbatim, without inventing a strategy
        # when a model does not return the legacy thinking JSON schema.
        return {"reasoning": response.content, "strategy": "", "key_priorities": [],
                "potential_concessions": [], "_token_usage": self._extract_token_usage_from_response(response)}


class HostedAgentFactory:
    def __init__(self, run, journal):
        self.run = run
        self.journal = journal

    async def create_agents(self, models, config):
        if models != [s["name"] for s in self.run["seats"]] or config["n_agents"] != len(models):
            raise ValueError("Requested seats differ from the resolved plan")
        return [HostedAgent(seat, self.journal, self.run) for seat in self.run["seats"]]
