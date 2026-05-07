"""
Item Allocation game environment implementation.

This module implements the classic item allocation negotiation game where
agents compete to allocate discrete items based on private preference vectors.
"""

import json
from typing import Any, Dict, List, Optional

from .base import GameEnvironment, GameType, ItemAllocationConfig


class ItemAllocationGame(GameEnvironment):
    """
    Item Allocation negotiation game.

    Game mechanics:
    - N agents negotiate over M discrete items
    - Each agent has a secret preference vector (value for each item)
    - Proposals assign items to agents (each item to exactly one agent)
    - Two-thirds supermajority voting required for acceptance
    - Utility = sum of preference values for received items × gamma^(round-1)
    """

    ITEM_NAMES = [
        "Apple", "Jewel", "Stone", "Quill", "Pencil",
        "Book", "Hat", "Camera", "Ring", "Clock",
        "Key", "Map", "Lantern", "Compass", "Vase",
        "Coin", "Shell", "Brush", "Scarf", "Cup",
        "Bottle", "Globe", "Medal", "Ticket", "Tablet",
    ]

    def __init__(self, config: ItemAllocationConfig):
        """
        Initialize Item Allocation game.

        Args:
            config: ItemAllocationConfig with m_items, competition_level, etc.
        """
        super().__init__(config)
        self.config: ItemAllocationConfig = config

    def create_game_state(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Create items and generate competitive preferences.

        Args:
            agents: List of agent objects with agent_id attributes

        Returns:
            Game state with items, preferences, and metadata
        """
        # Import here to avoid circular dependency
        from negotiation import create_competitive_preferences

        # Create items
        items = [
            {"name": self.ITEM_NAMES[i] if i < len(self.ITEM_NAMES) else f"Item_{i}"}
            for i in range(self.config.m_items)
        ]

        # Generate competitive preferences
        preference_manager = create_competitive_preferences(
            n_agents=self.config.n_agents,
            m_items=self.config.m_items,
            cosine_similarity=self.config.competition_level,
            random_seed=self.config.random_seed
        )
        preferences_data = preference_manager.generate_preferences()

        # Map preferences to agent IDs
        agent_preferences = {}
        for i, agent in enumerate(agents):
            agent_id = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
            agent_preferences[agent_id] = preferences_data["agent_preferences"][f"agent_{i}"]

        return {
            "items": items,
            "agent_preferences": agent_preferences,
            "cosine_similarities": preferences_data.get("cosine_similarities", {}),
            "game_type": "item_allocation"
        }

    def _get_agent_phrase(self) -> str:
        """Create clearer phrasing for 2-agent negotiations."""
        if self.config.n_agents == 2:
            return "another agent"
        return f"{self.config.n_agents - 1} other agents"

    @staticmethod
    def _format_display_number(value: float) -> str:
        """Render integer-valued floats without trailing .00."""
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.2f}"

    def _get_rules_block(self, game_state: Dict[str, Any]) -> str:
        """Build the shared rules/setup block for item allocation."""
        items = game_state["items"]
        items_text = "\n".join([f"  {i}: {item['name']}" for i, item in enumerate(items)])

        agent_phrase = self._get_agent_phrase()
        threshold = self.supermajority_threshold(self.config.n_agents)

        return f"""Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with {agent_phrase} over {len(items)} valuable items. Here is your full setup information:

**ITEMS BEING NEGOTIATED:**
{items_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} agents participating (including you)
- The negotiation will last up to {self.config.t_rounds} rounds
- This message is the one-time setup phase
- After setup, each round follows: Discussion -> Private Thinking -> Proposal -> Voting -> Reflection

**PRIVATE INFORMATION:**
- You have been assigned private item preferences
- These preferences are SECRET and specific to you

**VOTING RULES:**
- All proposals submitted in a round are shown together during voting
- You vote "accept" or "reject" on each proposal independently
- A proposal passes if it receives at least {threshold} accept votes out of {self.config.n_agents} agents (a two-thirds supermajority, rounded up)
- If multiple proposals pass, the proposal with the most accept votes is selected; exact top-count ties are broken randomly
- If no proposal gets supermajority support, we continue to the next round
- If no agreement is reached by the final round, then all agents walk away with zero utility.

**REWARD DISCOUNTING:**
- Rewards are discounted by a factor of {self.config.gamma_discount} per round
- Round 1 rewards: 100% of utility
- Round 2 rewards: {self.config.gamma_discount * 100:.0f}% of utility
- Round 3 rewards: {self.config.gamma_discount ** 2 * 100:.0f}% of utility
- The longer negotiations take, the less valuable the final allocation becomes

**WINNING CONDITIONS:**
- The goal is to maximize your utility, which is the sum of the utility from each of the objects that you receive.
- Your goal is to maximize your total utility (after discounting)
- No deal means everyone gets zero utility
- Consider both immediate gains and the likelihood of proposals being accepted
- Earlier agreements are worth more due to discounting"""

    def _get_private_preferences_block(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Build the per-agent private preference block."""
        items = game_state["items"]
        agent_prefs = game_state["agent_preferences"][agent_id]

        pref_lines = []
        for i, (item, value) in enumerate(zip(items, agent_prefs)):
            value_text = self._format_display_number(value)
            pref_lines.append(f"  {i}: {item['name']} -> {value_text}")

        max_utility = sum(agent_prefs)
        max_utility_text = self._format_display_number(max_utility)

        return f"""LOCKED PRIVATE PREFERENCES

{agent_id}, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
{chr(10).join(pref_lines)}

**STRATEGIC ANALYSIS:**
- Your theoretical maximum utility: {max_utility_text} points (if you received ALL items — unrealistic in negotiation; use this only as an upper bound)

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need at least {self.supermajority_threshold(self.config.n_agents)} out of {self.config.n_agents} agents to accept a proposal"""

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate item allocation game rules explanation."""
        rules_block = self._get_rules_block(game_state)
        return f"""{rules_block}

Please acknowledge that you understand these rules and are ready to participate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate preference assignment prompt for an agent."""
        preferences_block = self._get_private_preferences_block(agent_id, game_state)
        return f"""{preferences_block}

Please acknowledge that you understand your private preferences."""

    def uses_combined_setup_phase(self) -> bool:
        """Item Allocation merges private preference assignment into setup."""
        return True

    def get_combined_setup_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate the one-time setup prompt with rules and private preferences."""
        rules_block = self._get_rules_block(game_state)
        preferences_block = self._get_private_preferences_block(agent_id, game_state)

        return f"""{rules_block}

{preferences_block}

Please do not initiate the discussion or proposal phase yet.
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private preferences that were assigned to you."""

    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate proposal prompt for item allocation."""
        items = game_state["items"]
        item_names = [item['name'] for item in items]

        # Use a complete, structurally valid example so formatting guidance
        # remains correct for N > 2 and M > N.
        example_alloc = {aid: [] for aid in agents}
        for item_index in range(len(items)):
            example_alloc[agents[item_index % len(agents)]].append(item_index)
        alternate_alloc = {aid: [] for aid in agents}
        for item_index in range(len(items)):
            alternate_alloc[agents[(item_index + 1) % len(agents)]].append(item_index)
        example_payload = json.dumps(
            {
                "allocation": example_alloc,
                "reasoning": "Brief explanation of your proposed allocation",
            },
            indent=4,
        )
        alternate_payload = json.dumps(
            {
                "allocation": alternate_alloc,
                "reasoning": "Another brief allocation rationale using the same schema",
            },
            indent=4,
        )

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Please propose an allocation of items among all agents.

**Current Context:**
- Items: {item_names} (indices 0-{len(items)-1})
- Agents: {agents}
- Round: {round_num}/{self.config.t_rounds}{reasoning_instruction}

**Complete-Ownership Invariant:**
- Your proposal must account for every available item index exactly once.
- The union of all agent arrays must be exactly {list(range(len(items)))}.
- Do not omit any item index.
- Do not duplicate an item index within one agent's list or across multiple agents.
- Do not assign any item index outside 0-{len(items)-1}.

**Instructions:**
Respond with ONLY a JSON object in this exact format:
{example_payload}

**Rules:**
- Use item INDICES (0-{len(items)-1}), not names
- Each item must be assigned to exactly one agent
- All items must be allocated
- An agent can receive zero or multiple items

Additional valid JSON example using the same schema:
{alternate_payload}

{self.json_format_requirements()}"""

    @staticmethod
    def _coerce_item_index(value: Any) -> Optional[int]:
        """Return an integer item index for clean integer-like values."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("-"):
                digits = stripped[1:]
            else:
                digits = stripped
            return int(stripped) if digits.isdigit() else None
        return None

    def allocation_validation_diagnostics(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return deterministic diagnostics for item-allocation proposal validity."""
        items = game_state["items"]
        n_items = len(items)
        expected_items = set(range(n_items))
        allocation = proposal.get("allocation")
        diagnostics: Dict[str, Any] = {
            "expected_items": list(range(n_items)),
            "missing_items": [],
            "duplicate_items": {},
            "out_of_range_items": [],
            "non_integer_items": [],
            "unknown_agents": [],
            "non_list_agent_values": [],
            "schema_errors": [],
        }

        if not isinstance(allocation, dict):
            diagnostics["schema_errors"].append("allocation must be an object")
            diagnostics["missing_items"] = list(range(n_items))
            return diagnostics

        known_agents = list((game_state.get("agent_preferences") or {}).keys())
        known_agent_set = set(known_agents)
        if known_agent_set:
            diagnostics["unknown_agents"] = sorted(
                agent_id for agent_id in allocation if agent_id not in known_agent_set
            )

        item_locations: Dict[int, List[str]] = {}
        for agent_id, agent_items in allocation.items():
            if not isinstance(agent_items, list):
                diagnostics["non_list_agent_values"].append(str(agent_id))
                continue
            for raw_item in agent_items:
                item_index = self._coerce_item_index(raw_item)
                if item_index is None:
                    diagnostics["non_integer_items"].append(
                        {"agent": str(agent_id), "item": repr(raw_item)}
                    )
                    continue
                if item_index not in expected_items:
                    diagnostics["out_of_range_items"].append(
                        {"agent": str(agent_id), "item": item_index}
                    )
                    continue
                item_locations.setdefault(item_index, []).append(str(agent_id))

        assigned_items = set(item_locations)
        diagnostics["missing_items"] = sorted(expected_items - assigned_items)
        diagnostics["duplicate_items"] = {
            item_index: locations
            for item_index, locations in sorted(item_locations.items())
            if len(locations) > 1
        }
        return diagnostics

    def proposal_validation_error(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
    ) -> Optional[str]:
        """Describe why an item-allocation proposal violates strict validity."""
        diagnostics = self.allocation_validation_diagnostics(proposal, game_state)
        parts: List[str] = []

        if diagnostics["schema_errors"]:
            parts.extend(diagnostics["schema_errors"])
        if diagnostics["unknown_agents"]:
            parts.append(f"unknown agent IDs {diagnostics['unknown_agents']}")
        if diagnostics["non_list_agent_values"]:
            parts.append(
                f"agent allocation values must be arrays for {diagnostics['non_list_agent_values']}"
            )
        if diagnostics["non_integer_items"]:
            parts.append(f"non-integer item entries {diagnostics['non_integer_items']}")
        if diagnostics["out_of_range_items"]:
            parts.append(f"out-of-range item indices {diagnostics['out_of_range_items']}")
        if diagnostics["duplicate_items"]:
            duplicate_text = ", ".join(
                f"{item} assigned to {locations}"
                for item, locations in diagnostics["duplicate_items"].items()
            )
            parts.append(f"duplicate item indices ({duplicate_text})")
        if diagnostics["missing_items"]:
            parts.append(f"missing item indices {diagnostics['missing_items']}")

        if not parts:
            return None

        expected = diagnostics["expected_items"]
        return (
            "; ".join(parts)
            + f"; expected each item index in {expected} to appear exactly once"
        )

    def parse_proposal(
        self,
        response: str,
        agent_id: str,
        game_state: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """Parse proposal from agent response."""
        items = game_state["items"]
        n_items = len(items)

        try:
            proposal = self._parse_json_object(response, "proposal response")

            # Ensure allocation exists and is valid
            if "allocation" not in proposal:
                if isinstance(proposal.get("agreement"), dict):
                    proposal["allocation"] = proposal["agreement"]
                    proposal["recovered_from_legacy_agreement_key"] = True
                else:
                    raise ValueError("No allocation in proposal")

            # Validate and clean allocation
            allocation = proposal["allocation"]
            cleaned_allocation = {}

            for aid in agents:
                if aid in allocation:
                    # Ensure items are integers
                    items_list = allocation[aid]
                    if isinstance(items_list, list):
                        cleaned_allocation[aid] = [int(i) for i in items_list if 0 <= int(i) < n_items]
                    else:
                        cleaned_allocation[aid] = []
                else:
                    cleaned_allocation[aid] = []

            proposal["allocation"] = cleaned_allocation
            proposal["proposed_by"] = agent_id

            return proposal

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            # Preserve diagnostics without manufacturing a valid proposal. The
            # phase handler can retry or fail the run; it should not silently
            # enter a synthetic proposer-gets-all allocation into voting.
            fallback_allocation = {aid: [] for aid in agents}

            return {
                "allocation": fallback_allocation,
                "reasoning": "Failed to parse response; proposal is invalid",
                "proposed_by": agent_id,
                "raw_response": response,
                "parse_error": self.parse_error_payload(exc),
            }

    def validate_proposal(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> bool:
        """Validate item allocation proposal."""
        allocation = proposal.get("allocation", {})
        items = game_state["items"]
        n_items = len(items)

        # Collect all allocated items
        allocated_items = []
        for agent_items in allocation.values():
            if isinstance(agent_items, list):
                allocated_items.extend(agent_items)

        # Check all items allocated exactly once
        if sorted(allocated_items) != list(range(n_items)):
            return False

        return True

    def calculate_utility(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> float:
        """Calculate utility for item allocation."""
        allocation = proposal.get("allocation", {})
        agent_prefs = game_state["agent_preferences"][agent_id]
        agent_items = allocation.get(agent_id, [])

        # Sum preferences for received items
        raw_utility = sum(
            agent_prefs[i] for i in agent_items
            if isinstance(i, int) and 0 <= i < len(agent_prefs)
        )

        # Apply discount
        discounted_utility = raw_utility * (self.config.gamma_discount ** (round_num - 1))

        return discounted_utility

    def get_discussion_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate discussion prompt with conversation history.

        Args:
            agent_id: ID of the agent receiving this prompt
            game_state: Current game state
            round_num: Current negotiation round
            max_rounds: Maximum number of rounds
            discussion_history: List of previous discussion messages (strings)
            reasoning_token_budget: Optional target reasoning tokens (prompt instruction only)
        """
        items = game_state["items"]
        items_text = "\n".join([f"  {i}: {item['name']}" for i, item in enumerate(items)])

        # Build conversation history section
        history_section = ""
        if discussion_history:
            history_section = "\n**CONVERSATION SO FAR:**\n"
            for msg in discussion_history:
                history_section += f"{msg}\n\n"
            history_section += "---\n"

        urgency = ""
        if round_num >= max_rounds - 1:
            urgency = "\n⏰ **URGENT**: This is one of the final rounds!"

        if round_num == 1 and not discussion_history:
            # First speaker in first round
            context = """**DISCUSSION OBJECTIVES:**
- Share strategic information about your preferences
- Learn about other agents' priorities
- Explore potential coalition opportunities
- Identify mutually beneficial trade possibilities

You are the first to speak. Please share your thoughts on the items and any initial ideas for how a deal might be reached."""
        elif discussion_history:
            # Responding after other agents have already spoken this round
            if round_num == 1:
                context = """**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Keep the conversation flowing naturally."""
            else:
                context = f"""**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Since this is not the first round, also draw on what you learned from earlier discussion, proposals, and votes.
Use lessons from failed proposals to decide what to emphasize, clarify, or revise in your public response.
You do not need to reveal your full private strategy.{urgency}

Keep the conversation flowing naturally."""
        else:
            # First speaker in a later round
            context = f"""Previous proposals didn't reach supermajority support. Use what you learned from earlier discussion, proposals, and votes to guide what you say in this round.{urgency}

**DISCUSSION FOCUS:**
- Refer back to what earlier rounds revealed about agents' priorities and sticking points
- Use lessons from failed proposals to shape what you emphasize, clarify, or revise
- Highlight possible compromises, trade-offs, or coalition opportunities that could move the group closer to supermajority support

You are speaking first this round. Open the discussion in a way that reflects what you learned in earlier rounds. You do not need to reveal your full private strategy."""

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

This is the open discussion phase where agents can discuss and strategize publicly.

**ITEMS AVAILABLE:**
{items_text}

{history_section}{context}{reasoning_instruction}"""

    def get_voting_prompt(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int,
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate a voting prompt for a single proposal."""
        single_proposal = proposal.copy()
        single_proposal.setdefault("proposal_number", 1)
        return self.get_batch_voting_prompt(
            agent_id=agent_id,
            proposals=[single_proposal],
            game_state=game_state,
            round_num=round_num,
            reasoning_token_budget=reasoning_token_budget
        )

    def get_batch_voting_prompt(
        self,
        agent_id: str,
        proposals: List[Dict[str, Any]],
        game_state: Dict[str, Any],
        round_num: int,
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate a batch voting prompt covering all proposals in the round."""
        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        proposal_blocks = []
        for fallback_num, proposal in enumerate(proposals, start=1):
            proposal_number = proposal.get("proposal_number", fallback_num)
            proposal_blocks.append(
                f"PROPOSAL #{proposal_number}:\n"
                f"ALLOCATION: {json.dumps(proposal.get('allocation', {}), indent=2)}\n"
                f"PROPOSED BY: {proposal.get('proposed_by', 'Unknown')}"
            )

        example_entries = []
        for example_index, proposal in enumerate(proposals[: max(1, min(2, len(proposals)))], start=1):
            proposal_number = proposal.get("proposal_number", example_index)
            example_vote = "accept" if example_index == 1 else "reject"
            example_entries.append(
                "        {\n"
                f'            "proposal_number": {proposal_number},\n'
                f'            "vote": "{example_vote}",\n'
                f'            "reasoning": "Brief explanation of your vote on Proposal #{proposal_number}"\n'
                "        }"
            )

        proposals_text = "\n\n".join(proposal_blocks)
        votes_example = ",\n".join(example_entries)
        threshold = self.supermajority_threshold(self.config.n_agents)

        return f"""The following proposals have been made for item allocation this round:

{proposals_text}

**REMINDER — YOUR UTILITY:**
- Your utility = sum of preference values for items you receive, multiplied by the round discount
- Round 1: 100% | Round 2: {self.config.gamma_discount * 100:.0f}% | Round 3: {self.config.gamma_discount**2 * 100:.0f}% (\u03b3={self.config.gamma_discount} per round)
- If no deal is reached by the final round, your utility is 0

Vote on EACH proposal independently. Consider:
- How each allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting or rejecting each proposal
- A proposal passes with at least {threshold} accept votes out of {self.config.n_agents} agents (a two-thirds supermajority, rounded up); if several pass, the most-supported proposal is selected
- You may accept zero, one, or multiple proposals
- You may reject zero, one, or multiple proposals
- Seeing all proposals together does not eliminate any proposal before you vote{reasoning_instruction}

Respond with ONLY a JSON object in this exact format:
{{
    "votes": [
{votes_example}
    ]
}}

Include exactly one vote entry for each proposal shown above.
Each vote must be either "accept" or "reject".

{self.json_format_requirements()}"""

    def parse_batch_voting_response(
        self,
        response: str,
        proposal_numbers: List[int],
        agent_id: str,
        round_num: int
    ) -> List[Dict[str, Any]]:
        """Parse a batch voting response into one vote per proposal."""
        parse_error = None
        try:
            payload = self._parse_json_object(response, "batch vote response")

            raw_votes = payload.get("votes", [])
            if not isinstance(raw_votes, list):
                raise ValueError("Batch vote response did not contain a votes list")
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            raw_votes = []
            parse_error = self.parse_error_payload(exc)

        proposal_number_set = set(proposal_numbers)
        parsed_votes = {}

        for raw_vote in raw_votes:
            if not isinstance(raw_vote, dict):
                continue

            try:
                proposal_number = int(raw_vote.get("proposal_number"))
            except (TypeError, ValueError):
                continue

            if proposal_number not in proposal_number_set:
                continue

            raw_vote_value = raw_vote.get("vote")
            vote_parse_error = None
            if raw_vote_value is None:
                vote_value = "reject"
                vote_parse_error = {
                    "type": "MissingVoteField",
                    "message": "Missing vote field; defaulted to reject",
                }
            else:
                vote_value = str(raw_vote_value).strip().lower()
            if vote_value not in ("accept", "reject"):
                vote_parse_error = {
                    "type": "InvalidVoteValue",
                    "message": f"Invalid vote value {raw_vote_value!r}; defaulted to reject",
                }
                vote_value = "reject"

            vote_entry = {
                "proposal_number": proposal_number,
                "vote": vote_value,
                "reasoning": raw_vote.get("reasoning", ""),
                "voter": agent_id,
                "round": round_num,
            }
            if vote_parse_error is not None:
                vote_entry["synthetic_vote"] = True
                vote_entry["fallback_policy_version"] = "invalid-output-default-v1"
                vote_entry["raw_response"] = response
                vote_entry["parse_error"] = vote_parse_error

            parsed_votes[proposal_number] = vote_entry

        vote_results = []
        for proposal_number in proposal_numbers:
            default_vote = {
                "proposal_number": proposal_number,
                "vote": "reject",
                "reasoning": "Missing or invalid vote entry",
                "voter": agent_id,
                "round": round_num,
            }
            if parse_error is not None:
                default_vote["raw_response"] = response
                default_vote["parse_error"] = parse_error

            vote_results.append(parsed_votes.get(proposal_number, default_vote))

        return vote_results

    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate private thinking prompt.

        Args:
            agent_id: ID of the thinking agent
            game_state: Current game state
            round_num: Current round number
            max_rounds: Total rounds
            discussion_history: Previous discussion messages
            reasoning_token_budget: Optional target reasoning tokens (prompt instruction only)
        """
        items = game_state["items"]
        agent_prefs = game_state["agent_preferences"][agent_id]
        items_text = "\n".join([f"  {i}: {item['name']}" for i, item in enumerate(items)])
        preference_lines = [
            f"  {i}: {items[i]['name']} -> {self._format_display_number(agent_prefs[i])}"
            for i in range(len(items))
        ]

        urgency = ""
        if round_num >= max_rounds - 1:
            urgency = "\n⚠️ **CRITICAL**: This is one of your final opportunities!"

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"""

**REASONING DEPTH:**
Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."""

        return f"""🧠 PRIVATE STRATEGIC ANALYSIS - Round {round_num}/{max_rounds}

This is your private strategic planning time.

**ITEMS AVAILABLE:**
{items_text}{urgency}

**YOUR FULL PREFERENCE REMINDER:**
{chr(10).join(preference_lines)}

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other agents' priorities from the discussion so far?
2. Which items are your highest priorities to secure, and which lower-value items could you concede on?
3. What allocation would maximize your utility while still having a realistic path to supermajority acceptance?
4. Where are the likely sticking points, and how should you adapt if other agents push for items you value highly?{reasoning_instruction}

**OUTPUT REQUIRED:**
Respond with a JSON object:
{{
    "reasoning": "Your analysis of the item-allocation situation",
    "strategy": "Your negotiation strategy for this round",
    "key_priorities": ["0: Apple (value=9.20)", "..."],
    "potential_concessions": ["4: Pencil (value=4.10)", "..."]
}}

{self.json_format_requirements()}

Remember: This analysis is completely private."""

    def format_proposal_display(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        """Format proposal for display."""
        items = game_state["items"]
        allocation = proposal.get("allocation", {})

        lines = [f"PROPOSAL (by {proposal.get('proposed_by', 'Unknown')}):"]
        for agent_id, item_indices in allocation.items():
            if item_indices:
                item_names = [
                    f"{idx}:{items[idx]['name']}"
                    for idx in item_indices
                    if isinstance(idx, int) and 0 <= idx < len(items)
                ]
                lines.append(f"  {agent_id}: {', '.join(item_names)}")
            else:
                lines.append(f"  {agent_id}: (no items)")

        return "\n".join(lines)

    def get_game_type(self) -> GameType:
        """Return game type identifier."""
        return GameType.ITEM_ALLOCATION

    def get_agent_preferences_summary(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get preferences for logging."""
        return {
            "preferences": game_state["agent_preferences"][agent_id],
            "items": [item["name"] for item in game_state["items"]],
            "competition_level": self.config.competition_level
        }
