"""
Diplomatic Treaty game environment implementation.

This module implements a multi-issue diplomatic negotiation game where agents
negotiate over continuous issue positions with position preferences and
importance weights.
"""

import json
import re
from typing import Any, Dict, List

import numpy as np

from .base import GameEnvironment, GameType, DiplomaticTreatyConfig


class DiplomaticTreatyGame(GameEnvironment):
    """
    Diplomatic Treaty negotiation game.

    Game mechanics:
    - N agents negotiate over K issues (continuous values in [0,1])
    - Each agent has:
      - Position preferences p_i: ideal outcome on each issue
      - Importance weights w_i: how much they care about each issue
    - Proposals are agreement vectors A = [a_1, ..., a_k] where a_k ∈ [0,1]
    - Utility = Σ_k w_ik × (1 - |p_ik - a_k|)

    Three control parameters:
    - ρ (rho): Preference correlation [-1, 1]
    - θ (theta): Interest overlap [0, 1]
    - λ (lam): Issue compatibility [-1, 1]
    """

    # Thematic issue names for diplomatic negotiations
    ISSUE_NAMES = [
        "Trade Policy",
        "Military Access",
        "Environmental Standards",
        "Resource Sharing",
        "Border Security",
        "Technology Transfer",
        "Financial Cooperation",
        "Cultural Exchange",
        "Immigration Policy",
        "Maritime Rights"
    ]

    def __init__(self, config: DiplomaticTreatyConfig):
        """
        Initialize Diplomatic Treaty game.

        Args:
            config: DiplomaticTreatyConfig with n_issues, rho, theta, lam, etc.
        """
        super().__init__(config)
        self.config: DiplomaticTreatyConfig = config
        self._rng = np.random.RandomState(config.random_seed)

    def create_game_state(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Create issues and generate preferences for diplomatic negotiation.

        Args:
            agents: List of agent objects with agent_id attributes

        Returns:
            Game state with issues, positions, weights, and parameters
        """
        n_issues = self.config.n_issues
        n_agents = self.config.n_agents

        # Generate positions with correlation rho
        positions = self._generate_positions(n_agents, n_issues)

        # Generate weights with overlap theta
        weights = self._generate_weights(n_agents, n_issues)

        # Generate issue types with compatibility lambda
        issue_types = self._generate_issue_types(n_issues, positions)

        # Map to agent IDs
        agent_positions = {}
        agent_weights = {}
        for i, agent in enumerate(agents):
            agent_id = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
            agent_positions[agent_id] = positions[i].tolist()
            agent_weights[agent_id] = weights[i].tolist()

        # Create issue names
        issue_names = [
            self.ISSUE_NAMES[i] if i < len(self.ISSUE_NAMES) else f"Issue_{i+1}"
            for i in range(n_issues)
        ]

        return {
            "issues": issue_names,
            "n_issues": n_issues,
            "agent_positions": agent_positions,
            "agent_weights": agent_weights,
            "issue_types": issue_types.tolist(),
            "parameters": {
                "rho": self.config.rho,
                "theta": self.config.theta,
                "lambda": self.config.lam
            },
            "game_type": "diplomatic_treaty"
        }

    def _generate_positions(self, n_agents: int, n_issues: int) -> np.ndarray:
        """
        Generate position preferences with correlation rho.

        Args:
            n_agents: Number of agents
            n_issues: Number of issues

        Returns:
            positions: Shape (n_agents, n_issues), values in [0, 1]
        """
        sigma = 0.25  # Standard deviation
        cov_matrix = np.full((n_agents, n_agents), self.config.rho * sigma**2)
        np.fill_diagonal(cov_matrix, sigma**2)

        positions = np.zeros((n_agents, n_issues))
        mean = np.full(n_agents, 0.5)

        for k in range(n_issues):
            # Sample from multivariate normal
            pos_k = self._rng.multivariate_normal(mean, cov_matrix)
            # Clip to [0, 1]
            positions[:, k] = np.clip(pos_k, 0, 1)

        return positions

    def _generate_weights(self, n_agents: int, n_issues: int) -> np.ndarray:
        """
        Generate importance weights with overlap theta.

        Args:
            n_agents: Number of agents
            n_issues: Number of issues

        Returns:
            weights: Shape (n_agents, n_issues), normalized to sum to 1
        """
        alpha = 2.0  # Dirichlet concentration parameter

        weights = np.zeros((n_agents, n_issues))
        weights[0] = self._rng.dirichlet(np.full(n_issues, alpha))

        for i in range(1, n_agents):
            # Generate candidate weights
            w_candidate = self._rng.dirichlet(np.full(n_issues, alpha))

            # Mix with first agent to achieve target overlap
            w_mixed = self.config.theta * weights[0] + (1 - self.config.theta) * w_candidate

            # Renormalize
            weights[i] = w_mixed / w_mixed.sum()

        return weights

    def _generate_issue_types(self, n_issues: int, positions: np.ndarray) -> np.ndarray:
        """
        Generate issue types (compatible or conflicting) based on lambda.

        Args:
            n_issues: Number of issues
            positions: Position preferences (modified in place for conflicting issues)

        Returns:
            issue_types: Shape (n_issues,), values in {-1, 1}
        """
        # Probability of compatible issue
        p_compatible = (self.config.lam + 1) / 2

        # Sample issue types
        issue_types = self._rng.choice(
            [1, -1],
            size=n_issues,
            p=[p_compatible, 1 - p_compatible]
        )

        # For conflicting issues with 2 agents, make positions opposing
        if positions.shape[0] == 2:
            for k in range(n_issues):
                if issue_types[k] == -1:
                    positions[1, k] = 1 - positions[0, k]

        return issue_types

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate diplomatic negotiation rules explanation."""
        issues = game_state["issues"]
        params = game_state["parameters"]

        issues_text = "\n".join([f"  - {issue}" for issue in issues])

        # Create clearer phrasing for 2-agent negotiations
        if self.config.n_agents == 2:
            parties_phrase = "another party"
        else:
            parties_phrase = f"{self.config.n_agents - 1} other parties"

        return f"""Welcome to the Diplomatic Treaty Negotiation!

You are participating in a diplomatic negotiation with {parties_phrase} over {len(issues)} key policy issues.

**ISSUES BEING NEGOTIATED:**
{issues_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} parties negotiating (including you)
- The negotiation will last up to {self.config.t_rounds} rounds
- Each issue can be resolved anywhere in the range [0.0, 1.0]
  - 0.0 = One extreme position
  - 1.0 = Opposite extreme position
  - 0.5 = Neutral/compromise position

**YOUR PREFERENCES:**
- You have an IDEAL POSITION on each issue (what outcome you prefer)
- You have IMPORTANCE WEIGHTS (how much you care about each issue)
- Your preferences are PRIVATE - other parties don't know them

**AGREEMENT FORMAT:**
- An agreement is a vector of values, one per issue
- Example: [{', '.join(['0.3', '0.7', '0.5'][:min(3, len(issues))]}, ...]
- Each value represents where that issue is resolved on the [0,1] spectrum

**UTILITY CALCULATION:**
- Your utility = weighted sum of how close each issue is to your ideal
- Formula: Σ (weight_k × (1 - |your_position_k - agreement_k|))
- Maximum utility = 1.0 (all issues at your exact ideal positions)

**VOTING RULES:**
- You vote "accept" or "reject" on each proposed agreement
- A proposal needs UNANIMOUS acceptance from all parties
- Rewards discounted by {self.config.gamma_discount} per round

Please acknowledge that you understand these rules and are ready to negotiate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate diplomatic preference assignment prompt."""
        issues = game_state["issues"]
        positions = game_state["agent_positions"][agent_id]
        weights = game_state["agent_weights"][agent_id]
        issue_types = game_state["issue_types"]

        lines = ["🔒 CONFIDENTIAL: Your Diplomatic Preferences", ""]
        lines.append(f"{agent_id}, you have been assigned the following SECRET preferences:")
        lines.append("")
        lines.append("**YOUR IDEAL POSITIONS** (what outcome you want on each issue):")

        for i, (issue, pos) in enumerate(zip(issues, positions)):
            issue_type = "win-win" if issue_types[i] == 1 else "zero-sum"
            pos_desc = self._describe_position(pos)
            lines.append(f"  {issue}: {pos:.3f} ({pos_desc}, {issue_type})")

        lines.append("")
        lines.append("**YOUR IMPORTANCE WEIGHTS** (how much you care about each issue):")

        for issue, weight in zip(issues, weights):
            priority = "HIGH" if weight > 0.25 else "Medium" if weight > 0.15 else "Low"
            lines.append(f"  {issue}: {weight:.3f} ({priority} priority)")

        lines.append("")
        lines.append("**STRATEGIC INSIGHT:**")
        lines.append("- Focus on issues with HIGH weights - they matter most for your utility")
        lines.append("- Consider trading concessions on low-weight issues for gains on high-weight ones")
        lines.append("- Zero-sum issues have directly opposing interests; win-win issues may allow mutual gains")
        lines.append("")
        lines.append("Please acknowledge that you understand your diplomatic preferences.")

        return "\n".join(lines)

    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str]
    ) -> str:
        """Generate proposal prompt for diplomatic negotiation."""
        issues = game_state["issues"]
        n_issues = len(issues)

        issues_list = "\n".join([f"  {i}: {issue}" for i, issue in enumerate(issues)])

        return f"""Please propose a treaty agreement.

**Current Context:**
- Issues being negotiated:
{issues_list}
- Round: {round_num}/{self.config.t_rounds}

**Instructions:**
Propose a resolution for each issue as a value in [0.0, 1.0].

Respond with ONLY a JSON object in this exact format:
{{
    "agreement": [0.3, 0.7, 0.5, 0.2, 0.8],
    "reasoning": "Brief explanation of your proposed compromise"
}}

**Rules:**
- The "agreement" array must have exactly {n_issues} values (one per issue)
- Each value must be between 0.0 and 1.0
- Consider what would be acceptable to all parties"""

    def parse_proposal(
        self,
        response: str,
        agent_id: str,
        game_state: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """Parse diplomatic proposal from agent response."""
        n_issues = game_state["n_issues"]

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

            # Ensure agreement exists
            agreement = proposal.get("agreement", [])

            # Validate and clean agreement
            if not isinstance(agreement, list):
                raise ValueError("Agreement must be a list")

            # Pad or truncate to correct length
            if len(agreement) < n_issues:
                agreement.extend([0.5] * (n_issues - len(agreement)))
            elif len(agreement) > n_issues:
                agreement = agreement[:n_issues]

            # Clip values to [0, 1]
            agreement = [max(0.0, min(1.0, float(v))) for v in agreement]

            proposal["agreement"] = agreement
            proposal["proposed_by"] = agent_id

            return proposal

        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: propose neutral midpoint on all issues
            return {
                "agreement": [0.5] * n_issues,
                "reasoning": "Proposed neutral compromise on all issues",
                "proposed_by": agent_id
            }

    def validate_proposal(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> bool:
        """Validate diplomatic proposal."""
        agreement = proposal.get("agreement", [])
        n_issues = game_state["n_issues"]

        # Check correct number of issues
        if len(agreement) != n_issues:
            return False

        # Check all values in range
        for val in agreement:
            try:
                v = float(val)
                if not (0 <= v <= 1):
                    return False
            except (ValueError, TypeError):
                return False

        return True

    def calculate_utility(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> float:
        """
        Calculate diplomatic utility.

        Utility = Σ_k w_k × (1 - |p_k - a_k|)

        Args:
            agent_id: ID of the agent
            proposal: The proposal with agreement vector
            game_state: Current game state
            round_num: Current round (for discounting)

        Returns:
            Discounted utility value
        """
        agreement = np.array(proposal.get("agreement", []))
        positions = np.array(game_state["agent_positions"][agent_id])
        weights = np.array(game_state["agent_weights"][agent_id])

        # Value = 1 - |position - agreement| (distance penalty)
        values = 1 - np.abs(positions - agreement)

        # Weighted sum
        raw_utility = float(np.sum(weights * values))

        # Apply discount
        discounted_utility = raw_utility * (self.config.gamma_discount ** (round_num - 1))

        return discounted_utility

    def get_discussion_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """Generate diplomatic discussion prompt."""
        issues = game_state["issues"]
        issues_text = ", ".join(issues[:5])
        if len(issues) > 5:
            issues_text += f", and {len(issues) - 5} more"

        if round_num == 1:
            context = """**DISCUSSION OBJECTIVES:**
- Signal your priorities (without revealing exact preferences)
- Understand other parties' key concerns
- Identify potential areas for compromise
- Explore issue linkages and package deals

Share your diplomatic position and initial thoughts on reaching an agreement."""
        else:
            urgency = ""
            if round_num >= max_rounds - 1:
                urgency = "\n⚠️ **TIME PRESSURE**: Limited rounds remaining for agreement!"

            context = f"""Previous proposals didn't achieve consensus. Consider adjustments.{urgency}

**REFLECTION:**
- What concerns did other parties raise?
- Where might compromise be possible?
- Which issues could be linked for mutual benefit?

Share your updated diplomatic position."""

        return f"""🗣️ DIPLOMATIC DISCUSSION - Round {round_num}/{max_rounds}

Issues under negotiation: {issues_text}

{context}"""

    def get_voting_prompt(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> str:
        """Generate diplomatic voting prompt."""
        issues = game_state["issues"]
        agreement = proposal.get("agreement", [])

        # Format agreement for display
        agreement_lines = []
        for i, (issue, val) in enumerate(zip(issues, agreement)):
            pos_desc = self._describe_position(val)
            agreement_lines.append(f"  {issue}: {val:.3f} ({pos_desc})")

        agreement_display = "\n".join(agreement_lines)

        return f"""A treaty proposal has been submitted:

**PROPOSED AGREEMENT:**
{agreement_display}

**REASONING:** {proposal.get('reasoning', 'No reasoning provided')}
**PROPOSED BY:** {proposal.get('proposed_by', 'Unknown')}

Please vote on this proposal. Consider:
- How close is this agreement to your ideal positions?
- Could you get a better deal by continuing negotiation?
- The cost of delay (utility discounting)

Respond with ONLY a JSON object:
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
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """Generate diplomatic thinking prompt."""
        issues = game_state["issues"]
        positions = game_state["agent_positions"][agent_id]
        weights = game_state["agent_weights"][agent_id]

        # Find top priorities
        priority_indices = np.argsort(weights)[::-1][:3]
        top_priorities = [
            f"{issues[i]} (weight: {weights[i]:.2f}, ideal: {positions[i]:.2f})"
            for i in priority_indices
        ]

        urgency = ""
        if round_num >= max_rounds - 1:
            urgency = "\n⚠️ **CRITICAL**: Final rounds - agreement urgency is high!"

        return f"""🧠 PRIVATE STRATEGIC ANALYSIS - Round {round_num}/{max_rounds}
{urgency}

**YOUR TOP PRIORITIES:**
{chr(10).join(['- ' + p for p in top_priorities])}

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other parties' priorities?
2. Where might they be willing to compromise?
3. What agreement would maximize your utility while being acceptable to all?
4. Which issues could you concede on to gain elsewhere?

**OUTPUT REQUIRED:**
Respond with a JSON object:
{{
    "reasoning": "Your analysis of the diplomatic situation",
    "strategy": "Your negotiation strategy for this round",
    "key_priorities": ["Issue you care most about", "..."],
    "potential_concessions": ["Issue you could concede on", "..."]
}}

Remember: This analysis is completely private."""

    def format_proposal_display(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        """Format diplomatic proposal for display."""
        issues = game_state["issues"]
        agreement = proposal.get("agreement", [])

        lines = [f"TREATY PROPOSAL (by {proposal.get('proposed_by', 'Unknown')}):"]

        for issue, val in zip(issues, agreement):
            pos_desc = self._describe_position(val)
            lines.append(f"  {issue}: {val:.3f} ({pos_desc})")

        lines.append(f"  Reasoning: {proposal.get('reasoning', 'None')}")

        return "\n".join(lines)

    def get_game_type(self) -> GameType:
        """Return game type identifier."""
        return GameType.DIPLOMATIC_TREATY

    def get_agent_preferences_summary(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get diplomatic preferences for logging."""
        return {
            "positions": game_state["agent_positions"][agent_id],
            "weights": game_state["agent_weights"][agent_id],
            "issues": game_state["issues"],
            "parameters": game_state["parameters"]
        }

    @staticmethod
    def _describe_position(value: float) -> str:
        """Convert numeric position to descriptive text."""
        if value < 0.2:
            return "strongly low"
        elif value < 0.4:
            return "moderately low"
        elif value < 0.6:
            return "neutral"
        elif value < 0.8:
            return "moderately high"
        else:
            return "strongly high"
