"""
Item Allocation game environment implementation.

This module implements the classic item allocation negotiation game where
agents compete to allocate discrete items based on private preference vectors.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base import GameEnvironment, GameType, ItemAllocationConfig


class ItemAllocationGame(GameEnvironment):
    """
    Item Allocation negotiation game.

    Game mechanics:
    - N agents negotiate over M discrete items
    - Each agent has a secret preference vector (value for each item)
    - Proposals assign items to agents (each item to exactly one agent)
    - Unanimous voting required for acceptance
    - Utility = sum of preference values for received items × gamma^(round-1)
    """

    ITEM_NAMES = [
        "Apple", "Jewel", "Stone", "Quill", "Pencil",
        "Book", "Hat", "Camera", "Ring", "Clock"
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
            cosine_similarity=self.config.competition_level
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

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate item allocation game rules explanation."""
        items = game_state["items"]
        items_text = "\n".join([f"  {i}: {item['name']}" for i, item in enumerate(items)])

        # Create clearer phrasing for 2-agent negotiations
        if self.config.n_agents == 2:
            agent_phrase = "another agent"
        else:
            agent_phrase = f"{self.config.n_agents - 1} other agents"

        return f"""Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with {agent_phrase} over {len(items)} valuable items. Here are the complete rules:

**ITEMS BEING NEGOTIATED:**
{items_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} agents participating (including you)
- The negotiation will last up to {self.config.t_rounds} rounds
- Each round follows a structured sequence of phases

**YOUR PRIVATE PREFERENCES:**
You have been assigned private preferences for each item. These preferences are SECRET.

**VOTING RULES:**
- You vote "accept" or "reject" on each proposal
- A proposal needs UNANIMOUS acceptance to pass
- If no proposal gets unanimous support, we continue to the next round

**REWARD DISCOUNTING:**
- Rewards are discounted by a factor of {self.config.gamma_discount} per round
- Round 1 rewards: 100% of utility
- Round 2 rewards: {self.config.gamma_discount * 100:.0f}% of utility
- Round 3 rewards: {self.config.gamma_discount ** 2 * 100:.0f}% of utility
- The longer negotiations take, the less valuable the final allocation becomes

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility (after discounting)
- No deal means everyone gets zero utility
- Consider both immediate gains and the likelihood of proposals being accepted
- Earlier agreements are worth more due to discounting

Please acknowledge that you understand these rules and are ready to participate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate preference assignment prompt for an agent."""
        items = game_state["items"]
        agent_prefs = game_state["agent_preferences"][agent_id]

        pref_lines = []
        for i, (item, value) in enumerate(zip(items, agent_prefs)):
            priority = self._get_priority_level(value)
            pref_lines.append(f"  {i}: {item['name']} → {value:.2f} ({priority})")

        max_utility = sum(agent_prefs)

        return f"""🔒 CONFIDENTIAL: Your Private Preferences Assignment

{agent_id}, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
{chr(10).join(pref_lines)}

**STRATEGIC ANALYSIS:**
- Your maximum possible utility: {max_utility:.2f} points (if you get ALL items)

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need ALL agents to accept a proposal

Please acknowledge that you understand your private preferences."""

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

        # Build example allocation
        example_alloc = {}
        for i, aid in enumerate(agents):
            example_alloc[aid] = [i] if i < len(items) else []

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Please propose an allocation of items among all agents.

**Current Context:**
- Items: {item_names} (indices 0-{len(items)-1})
- Agents: {agents}
- Round: {round_num}/{self.config.t_rounds}{reasoning_instruction}

**Instructions:**
Respond with ONLY a JSON object in this exact format:
{{
    "allocation": {{
        "{agents[0]}": [0, 2],
        "{agents[1]}": [1, 3, 4]
    }},
    "reasoning": "Brief explanation of your proposed allocation"
}}

**Rules:**
- Use item INDICES (0-{len(items)-1}), not names
- Each item must be assigned to exactly one agent
- All items must be allocated
- An agent can receive zero or multiple items"""

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
            # Try direct JSON parse
            if response.strip().startswith('{'):
                proposal = json.loads(response)
            else:
                # Extract JSON from text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    proposal = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            # Ensure allocation exists and is valid
            if "allocation" not in proposal:
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

        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: proposer gets all items
            fallback_allocation = {aid: [] for aid in agents}
            fallback_allocation[agent_id] = list(range(n_items))

            return {
                "allocation": fallback_allocation,
                "reasoning": "Failed to parse response - defaulting to proposer gets all",
                "proposed_by": agent_id
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

        if round_num == 1 and not discussion_history:
            # First speaker in first round
            context = """**DISCUSSION OBJECTIVES:**
- Share strategic information about your preferences
- Learn about other agents' priorities
- Explore potential coalition opportunities
- Identify mutually beneficial trade possibilities

You are the first to speak. Please share your thoughts on the items and any initial ideas for how we might structure a deal."""
        elif round_num == 1:
            # Continuing discussion in first round
            context = """**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to specific points raised by other agents
- Share your own perspective on the items
- Propose potential trade-offs or areas of agreement
- Ask clarifying questions if needed

Keep the conversation flowing naturally."""
        else:
            urgency = ""
            if round_num >= max_rounds - 1:
                urgency = "\n⏰ **URGENT**: This is one of the final rounds!"

            context = f"""Previous proposals didn't reach consensus. Adjust your approach based on what you learned.{urgency}

**REFLECTION & STRATEGY:**
- What did you learn from previous proposals and votes?
- Which agents have conflicting vs. compatible preferences?
- How can you adjust to build consensus?

Given what happened in previous rounds, what's your updated strategy?"""

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

This is the open discussion phase where all agents can share information about their preferences.

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
        """Generate voting prompt."""
        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""A proposal has been made for item allocation:

PROPOSAL: {json.dumps(proposal.get('allocation', {}), indent=2)}
REASONING: {proposal.get('reasoning', 'No reasoning provided')}
PROPOSED BY: {proposal.get('proposed_by', 'Unknown')}

Please vote on this proposal. Consider:
- How this allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting vs. rejecting{reasoning_instruction}

Respond with ONLY a JSON object in this exact format:
{{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}}

Vote must be either "accept" or "reject"."""

    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]],
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
        items_text = "\n".join([f"  {i}: {item['name']}" for i, item in enumerate(items)])

        urgency = ""
        if round_num >= max_rounds - 1:
            urgency = "\n⚠️ **CRITICAL**: This is one of your final opportunities!"

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"""

**REASONING DEPTH:**
Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."""

        return f"""🧠 PRIVATE THINKING PHASE - Round {round_num}/{max_rounds}

This is your private strategic planning time.

**ITEMS AVAILABLE:**
{items_text}{urgency}

**STRATEGIC ANALYSIS TASKS:**
1. What did you learn about other agents' preferences?
2. Which items do others value less that you value highly?
3. What allocation would maximize your utility while achieving consensus?
4. What concessions might be necessary?{reasoning_instruction}

**OUTPUT REQUIRED:**
Respond with a JSON object:
{{
    "reasoning": "Your analysis of the situation",
    "strategy": "Your overall approach for this round",
    "target_items": [0, 2, 4],
    "anticipated_resistance": ["Agent who might block", "..."]
}}

Remember: This thinking is completely private."""

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

        lines.append(f"  Reasoning: {proposal.get('reasoning', 'None')}")

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

    @staticmethod
    def _get_priority_level(value: float) -> str:
        """Convert numeric preference to priority description."""
        if value >= 7.0:
            return "HIGH PRIORITY"
        elif value >= 4.0:
            return "Medium Priority"
        else:
            return "Low Priority"
