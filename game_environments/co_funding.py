"""
=============================================================================
Co-Funding (Participatory Budgeting) game environment implementation.
=============================================================================

This module implements a participatory budgeting / co-funding negotiation game
where agents submit individual contribution vectors to co-fund threshold public
goods (projects). Projects are funded only when aggregate contributions meet
or exceed the project cost.

Protocol: Talk-Pledge-Revise (no voting phase)
  1. Discussion - agents discuss strategy publicly
  2. Thinking - private strategic analysis
  3. Pledge Submission - each agent submits a contribution vector
  4. Feedback - agents see aggregate totals (not individual contributions)
  5. Reflection - agents reflect on the round outcome
  6. Repeat until convergence or max rounds

Key differences from Games 1/2:
  - No voting phase -- pledges replace proposals, no accept/reject
  - Decentralized contributions -- each agent submits their own vector
  - Threshold non-linearity -- projects funded only when sum >= cost
  - Only aggregate totals visible (individual contributions private)
  - Final-state-only evaluation (no discount factor)
  - Early termination when all pledges converge for 2 consecutive rounds

Dependencies:
    - numpy, scipy (for preference generation via SLSQP optimization)
    - game_environments.base (GameEnvironment, GameType, CoFundingConfig)
"""

import json
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from .base import GameEnvironment, GameType, CoFundingConfig


class CoFundingGame(GameEnvironment):
    """
    Co-Funding (Participatory Budgeting) negotiation game.

    Game mechanics:
    - N agents collectively fund M projects (threshold public goods)
    - Each agent has:
      - Valuation vector v_i: value derived if project j is funded (sum=100)
      - Budget B_i: maximum total contribution
    - Each round, agents submit contribution vectors x_i = [x_i1, ..., x_iM]
    - Project j is funded if sum_i(x_ij) >= c_j (threshold)
    - Utility: U_i = sum_{j in S} v_ij - sum_{j in S} x_ij
      where S is the set of funded projects
    - Contributions to unfunded projects are refunded (not counted in utility)

    Two control parameters:
    - alpha: Preference alignment [0, 1] (cosine similarity of valuations)
    - sigma: Budget scarcity (0, 1] (total budget / total project cost)
    """

    PROJECT_NAMES = [
        "Project Alpha", "Project Beta", "Project Gamma",
        "Project Delta", "Project Epsilon", "Project Zeta",
        "Project Eta", "Project Theta", "Project Iota", "Project Kappa"
    ]

    def __init__(self, config: CoFundingConfig):
        """
        Initialize Co-Funding game.

        Args:
            config: CoFundingConfig with m_projects, alpha, sigma, etc.
        """
        super().__init__(config)
        self.config: CoFundingConfig = config
        self._rng = np.random.RandomState(config.random_seed)

    def create_game_state(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Create projects, generate valuations and budgets.

        Args:
            agents: List of agent objects with agent_id attributes

        Returns:
            Game state dict with projects, valuations, budgets, and tracking fields
        """
        m = self.config.m_projects
        n = self.config.n_agents

        # Generate project costs: c_j ~ Uniform(c_min, c_max)
        project_costs = self._rng.uniform(
            self.config.c_min, self.config.c_max, size=m
        ).tolist()
        total_cost = sum(project_costs)

        # Generate budgets: B = sigma * C, B_i = B / N
        total_budget = self.config.sigma * total_cost
        per_agent_budget = total_budget / n

        # Generate valuation vectors with target cosine similarity alpha
        valuations = self._generate_valuations(n, m)

        # Create project list
        projects = []
        for j in range(m):
            name = self.PROJECT_NAMES[j] if j < len(self.PROJECT_NAMES) else f"Project_{j+1}"
            projects.append({"name": name, "cost": round(project_costs[j], 2)})

        # Map to agent IDs
        agent_ids = []
        agent_valuations = {}
        agent_budgets = {}
        for i, agent in enumerate(agents):
            aid = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
            agent_ids.append(aid)
            agent_valuations[aid] = valuations[i].tolist()
            agent_budgets[aid] = round(per_agent_budget, 2)

        return {
            "projects": projects,
            "m_projects": m,
            "project_costs": [round(c, 2) for c in project_costs],
            "total_cost": round(total_cost, 2),
            "agent_budgets": agent_budgets,
            "total_budget": round(total_budget, 2),
            "agent_valuations": agent_valuations,
            "parameters": {
                "alpha": self.config.alpha,
                "sigma": self.config.sigma,
            },
            "game_type": "co_funding",
            # Mutable tracking fields
            "round_pledges": [],
            "current_pledges": {},
            "aggregate_totals": [0.0] * m,
            "funded_projects": [],
        }

    def _generate_valuations(self, n_agents: int, m_projects: int) -> np.ndarray:
        """
        Generate valuation vectors with target cosine similarity alpha.

        Each agent's valuation vector sums to 100 with non-negative entries.
        Uses SLSQP optimization (same pattern as diplomatic_treaty.py).

        Args:
            n_agents: Number of agents
            m_projects: Number of projects

        Returns:
            valuations: Shape (n_agents, m_projects), each row sums to 100
        """
        alpha = self.config.alpha
        M = m_projects
        dir_alpha = 2.0  # Dirichlet concentration

        if alpha == 1.0:
            # Special case: identical valuations
            v0 = self._rng.dirichlet(np.full(M, dir_alpha)) * 100.0
            return np.tile(v0, (n_agents, 1))

        if n_agents == 2:
            return self._generate_valuations_2agents(M, alpha, dir_alpha)
        else:
            return self._generate_valuations_nagents(n_agents, M, alpha, dir_alpha)

    def _generate_valuations_2agents(self, M: int, alpha: float, dir_alpha: float) -> np.ndarray:
        """Generate valuations for exactly 2 agents with target cosine similarity."""
        best_x = None
        best_error = float('inf')
        max_attempts = 15

        def objective(x):
            v1 = x[:M]
            v2 = x[M:]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                return 1e10
            cos_sim = np.dot(v1, v2) / (n1 * n2)
            return (cos_sim - alpha) ** 2

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:M]) - 100.0},
            {'type': 'eq', 'fun': lambda x: np.sum(x[M:]) - 100.0},
        ]
        bounds = [(0.0, 100.0)] * (2 * M)

        for _ in range(max_attempts):
            conc = self._rng.uniform(0.5, 3.0)
            x0 = np.concatenate([
                self._rng.dirichlet(np.full(M, conc)) * 100.0,
                self._rng.dirichlet(np.full(M, conc)) * 100.0,
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-14}
                )

            error = objective(result.x)
            if error < best_error:
                best_error = error
                best_x = result.x.copy()
            if best_error < 1e-10:
                break

        if best_x is None or best_error > 0.0001:
            raise RuntimeError(
                f"Failed to generate valuations with cosine similarity "
                f"alpha={alpha} after {max_attempts} attempts. "
                f"Best error: {best_error:.6f}"
            )

        valuations = np.zeros((2, M))
        valuations[0] = np.maximum(best_x[:M], 0.0)
        valuations[0] = valuations[0] / valuations[0].sum() * 100.0
        valuations[1] = np.maximum(best_x[M:], 0.0)
        valuations[1] = valuations[1] / valuations[1].sum() * 100.0
        return valuations

    def _generate_valuations_nagents(self, n_agents: int, M: int, alpha: float, dir_alpha: float) -> np.ndarray:
        """Generate valuations for N>2 agents with target cosine similarity against reference."""
        valuations = np.zeros((n_agents, M))
        valuations[0] = self._rng.dirichlet(np.full(M, dir_alpha)) * 100.0
        v0 = valuations[0]
        v0_norm = np.linalg.norm(v0)

        for i in range(1, n_agents):
            best_x = None
            best_error = float('inf')
            max_attempts = 20

            def objective(v):
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    return 1e10
                cos_sim = np.dot(v0, v) / (v0_norm * v_norm)
                return (cos_sim - alpha) ** 2

            constraints = [
                {'type': 'eq', 'fun': lambda v: np.sum(v) - 100.0}
            ]
            bounds = [(0.0, 100.0)] * M

            for _ in range(max_attempts):
                conc = self._rng.uniform(0.5, 3.0)
                x0 = self._rng.dirichlet(np.full(M, conc)) * 100.0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 2000, 'ftol': 1e-14}
                    )

                error = objective(result.x)
                if error < best_error:
                    best_error = error
                    best_x = result.x.copy()
                if best_error < 1e-10:
                    break

            if best_x is None or best_error > 0.0001:
                raise RuntimeError(
                    f"Failed to generate valuations with cosine similarity "
                    f"alpha={alpha} for agent {i} after "
                    f"{max_attempts} attempts. Best error: {best_error:.6f}"
                )

            v = np.maximum(best_x, 0.0)
            v = v / v.sum() * 100.0
            valuations[i] = v

        return valuations

    # ---- Abstract method implementations ----

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate co-funding game rules explanation."""
        projects = game_state["projects"]

        projects_text = "\n".join([
            f"  - {p['name']}: cost = {p['cost']:.2f}"
            for p in projects
        ])

        if self.config.n_agents == 2:
            parties_phrase = "one other participant"
        else:
            parties_phrase = f"{self.config.n_agents - 1} other participants"

        return f"""Welcome to the Participatory Budgeting (Co-Funding) Game!

You are participating in a co-funding exercise with {parties_phrase} to fund public projects.

**PROJECTS AVAILABLE FOR FUNDING:**
{projects_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} participants (including you)
- The game lasts up to {self.config.t_rounds} rounds
- Each round follows a Talk-Pledge-Revise cycle

**HOW IT WORKS:**
- Each participant has a PRIVATE BUDGET they can allocate across projects
- Each round, you submit a CONTRIBUTION VECTOR specifying how much you pledge to each project
- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost
- Contributions to UNFUNDED projects are REFUNDED (you don't lose that money)

**WHAT YOU CAN SEE:**
- After each round, you see the AGGREGATE total contributions per project
- You do NOT see individual contributions from other participants

**YOUR UTILITY:**
- Utility = (sum of your valuations for funded projects) - (your contributions to funded projects)
- You gain value from funded projects but pay for your contributions to them
- Contributions to unfunded projects cost you nothing (refunded)

**IMPORTANT RULES:**
- No discount factor -- only the final round's outcome matters
- The game ends early if all participants submit identical pledges for 2 consecutive rounds
- Your goal: maximize your utility by strategically choosing contributions

Please acknowledge that you understand these rules and are ready to participate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate preference assignment prompt showing budget, costs, and valuations."""
        projects = game_state["projects"]
        valuations = game_state["agent_valuations"][agent_id]
        budget = game_state["agent_budgets"][agent_id]
        costs = game_state["project_costs"]

        lines = ["CONFIDENTIAL: Your Co-Funding Preferences", ""]
        lines.append(f"{agent_id}, you have been assigned the following:")
        lines.append("")
        lines.append(f"**YOUR BUDGET:** {budget:.2f} (maximum total you can contribute across all projects)")
        lines.append("")
        lines.append("**PROJECT DETAILS AND YOUR VALUATIONS:**")

        for j, (proj, val, cost) in enumerate(zip(projects, valuations, costs)):
            priority = "HIGH" if val > 30 else "Medium" if val > 15 else "Low"
            lines.append(f"  {proj['name']} (cost: {cost:.2f}): Your valuation = {val:.2f} ({priority} priority)")

        lines.append("")
        lines.append(f"**TOTAL VALUATIONS:** {sum(valuations):.2f}")
        lines.append(f"**TOTAL PROJECT COSTS:** {game_state['total_cost']:.2f}")
        lines.append(f"**TOTAL BUDGET (all participants):** {game_state['total_budget']:.2f}")
        lines.append("")
        lines.append("**STRATEGIC INSIGHT:**")
        lines.append("- Focus contributions on projects you value highly")
        lines.append("- Coordinate with others to meet project cost thresholds")
        lines.append("- Don't over-contribute to projects others will fund")
        lines.append("")
        lines.append("Please acknowledge that you understand your preferences and budget.")

        return "\n".join(lines)

    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate pledge submission prompt."""
        projects = game_state["projects"]
        m = len(projects)
        budget = game_state["agent_budgets"][agent_id]
        valuations = game_state["agent_valuations"][agent_id]
        costs = game_state["project_costs"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]

        project_lines = []
        for j, (proj, cost, val, agg) in enumerate(zip(projects, costs, valuations, aggregates)):
            status = "FUNDED" if j in funded else f"needs {max(0, cost - agg):.2f} more"
            project_lines.append(
                f"  {j}: {proj['name']} (cost={cost:.2f}, your_val={val:.2f}, "
                f"aggregate={agg:.2f}, {status})"
            )

        projects_text = "\n".join(project_lines)

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Please submit your contribution pledge for Round {round_num}/{self.config.t_rounds}.

**YOUR BUDGET:** {budget:.2f}

**PROJECT STATUS:**
{projects_text}

**Currently funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}{reasoning_instruction}

**Instructions:**
Submit a contribution vector specifying how much you pledge to each project.

Respond with ONLY a JSON object in this exact format:
{{
    "contributions": [5.0, 10.0, 0.0, 8.0, 2.0],
    "reasoning": "Brief explanation of your contribution strategy"
}}

**Rules:**
- The "contributions" array must have exactly {m} values (one per project)
- Each value must be non-negative (>= 0)
- The sum of all contributions must not exceed your budget ({budget:.2f})
- Contributions to unfunded projects will be refunded"""

    def parse_proposal(
        self,
        response: str,
        agent_id: str,
        game_state: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """Parse pledge response into contribution vector."""
        m = game_state["m_projects"]

        try:
            # Try direct JSON parse
            if response.strip().startswith('{'):
                parsed = json.loads(response)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            contributions = parsed.get("contributions", [])

            if not isinstance(contributions, list):
                raise ValueError("Contributions must be a list")

            # Pad or truncate
            if len(contributions) < m:
                contributions.extend([0.0] * (m - len(contributions)))
            elif len(contributions) > m:
                contributions = contributions[:m]

            # Ensure numeric and non-negative
            contributions = [max(0.0, float(v)) for v in contributions]

            return {
                "contributions": contributions,
                "reasoning": parsed.get("reasoning", ""),
                "proposed_by": agent_id,
            }

        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: zero contributions
            return {
                "contributions": [0.0] * m,
                "reasoning": "Failed to parse response - defaulting to zero contributions",
                "proposed_by": agent_id,
            }

    def validate_proposal(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> bool:
        """Validate pledge: non-negative, within budget, correct length."""
        contributions = proposal.get("contributions", [])
        agent_id = proposal.get("proposed_by", "")
        m = game_state["m_projects"]
        budget = game_state["agent_budgets"].get(agent_id, 0.0)

        if len(contributions) != m:
            return False

        for v in contributions:
            try:
                if float(v) < -1e-9:  # small tolerance for float
                    return False
            except (ValueError, TypeError):
                return False

        if sum(float(v) for v in contributions) > budget + 1e-6:
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
        Calculate utility for co-funding game.

        U_i = sum_{j in S} v_ij - sum_{j in S} x_ij

        where S is the funded set. No discount factor (final-state-only).
        The round_num parameter is ignored for this game.
        """
        valuations = game_state["agent_valuations"][agent_id]
        funded = game_state.get("funded_projects", [])
        contributions = proposal.get("contributions", [0.0] * game_state["m_projects"])

        utility = 0.0
        for j in funded:
            utility += valuations[j] - contributions[j]

        return utility

    def get_discussion_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate discussion prompt for co-funding."""
        projects = game_state["projects"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        costs = game_state["project_costs"]

        # Project status
        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            if j in funded:
                status_lines.append(f"  {proj['name']}: FUNDED (aggregate={agg:.2f} >= cost={cost:.2f})")
            else:
                gap = cost - agg
                status_lines.append(f"  {proj['name']}: needs {gap:.2f} more (aggregate={agg:.2f} / cost={cost:.2f})")
        status_text = "\n".join(status_lines)

        if round_num == 1:
            context = """**DISCUSSION OBJECTIVES:**
- Signal which projects you believe are most valuable to fund
- Understand other participants' priorities
- Coordinate to avoid spreading contributions too thin
- Identify projects with enough collective support to be funded

Share your initial thoughts on which projects to prioritize."""
        else:
            urgency = ""
            if round_num >= max_rounds - 1:
                urgency = "\n**TIME PRESSURE**: Limited rounds remaining!"

            context = f"""Previous pledges did not fully fund all viable projects.{urgency}

**REFLECTION:**
- Which projects are close to being funded?
- Where should contributions be concentrated?
- Are there projects that should be abandoned to focus resources?

Share your updated strategy for this round."""

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""DISCUSSION PHASE - Round {round_num}/{max_rounds}

**CURRENT PROJECT STATUS:**
{status_text}

**Funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}

{context}{reasoning_instruction}"""

    def get_voting_prompt(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int,
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Co-funding uses Talk-Pledge-Revise protocol; voting is not applicable."""
        # This method exists only to satisfy the abstract interface.
        # The orchestrator never calls it for co-funding games.
        _msg = "Co-funding uses Talk-Pledge-Revise, not voting"
        raise RuntimeError(_msg)

    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate private thinking prompt for co-funding."""
        projects = game_state["projects"]
        valuations = game_state["agent_valuations"][agent_id]
        budget = game_state["agent_budgets"][agent_id]
        costs = game_state["project_costs"]
        aggregates = game_state["aggregate_totals"]

        # Own contributions if available
        own_contribs = game_state.get("current_pledges", {}).get(agent_id, {}).get("contributions", [0.0] * len(projects))

        priority_indices = np.argsort(valuations)[::-1][:3]
        top_priorities = [
            f"{projects[i]['name']} (val={valuations[i]:.2f}, cost={costs[i]:.2f})"
            for i in priority_indices
        ]

        urgency = ""
        if round_num >= max_rounds - 1:
            urgency = "\n**CRITICAL**: Final rounds -- decide on your strategy now!"

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"""

**REASONING DEPTH:**
Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."""

        return f"""PRIVATE STRATEGIC ANALYSIS - Round {round_num}/{max_rounds}
{urgency}

**YOUR SITUATION:**
- Budget: {budget:.2f}
- Your current contributions: {[round(c, 2) for c in own_contribs]}
- Aggregate totals: {[round(a, 2) for a in aggregates]}

**YOUR TOP PRIORITIES:**
{chr(10).join(['- ' + p for p in top_priorities])}

**STRATEGIC ANALYSIS:**
1. Which projects are viable to fund given current aggregates?
2. Where can you shift contributions for maximum impact?
3. Are other participants likely to increase/decrease their pledges?
4. Should you free-ride on projects others are funding?{reasoning_instruction}

**OUTPUT REQUIRED:**
Respond with a JSON object:
{{
    "reasoning": "Your analysis of the co-funding situation",
    "strategy": "Your contribution strategy for this round",
    "key_priorities": ["Project you want funded most", "..."],
    "potential_concessions": ["Project you might reduce contributions to", "..."]
}}

Remember: This analysis is completely private."""

    def format_proposal_display(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        """Format pledge for display."""
        projects = game_state["projects"]
        contributions = proposal.get("contributions", [])

        lines = [f"PLEDGE (by {proposal.get('proposed_by', 'Unknown')}):"]

        for j, (proj, contrib) in enumerate(zip(projects, contributions)):
            lines.append(f"  {proj['name']}: {contrib:.2f}")

        total = sum(contributions)
        lines.append(f"  Total pledged: {total:.2f}")
        lines.append(f"  Reasoning: {proposal.get('reasoning', 'None')}")

        return "\n".join(lines)

    def get_game_type(self) -> GameType:
        """Return game type identifier."""
        return GameType.CO_FUNDING

    def get_protocol_type(self) -> str:
        """Return Talk-Pledge-Revise protocol type."""
        return "talk_pledge_revise"

    def get_agent_preferences_summary(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get co-funding preferences for logging."""
        return {
            "valuations": game_state["agent_valuations"][agent_id],
            "budget": game_state["agent_budgets"][agent_id],
            "projects": [p["name"] for p in game_state["projects"]],
            "project_costs": game_state["project_costs"],
            "parameters": game_state["parameters"],
        }

    def get_reflection_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        tabulation_result: Dict[str, Any],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate reflection prompt after a co-funding round."""
        projects = game_state["projects"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        costs = game_state["project_costs"]
        valuations = game_state["agent_valuations"][agent_id]

        # Compute current utility
        own_pledge = game_state.get("current_pledges", {}).get(agent_id, {})
        own_contribs = own_pledge.get("contributions", [0.0] * len(projects))
        utility = sum(valuations[j] - own_contribs[j] for j in funded) if funded else 0.0

        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            status = "FUNDED" if j in funded else f"gap={cost - agg:.2f}"
            status_lines.append(f"  {proj['name']}: aggregate={agg:.2f}, cost={cost:.2f} ({status})")

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Reflect on the outcome of Round {round_num}.

**CURRENT STATUS:**
{chr(10).join(status_lines)}

**Funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}
**Your estimated utility:** {utility:.2f}

Consider what adjustments to your contributions might improve the outcome.
- Are there projects close to being funded that deserve more support?
- Are you over-contributing to already-funded projects?
- Should you shift focus to different projects?{reasoning_instruction}"""

    # ---- Co-funding specific helper methods ----

    def update_game_state_with_pledges(
        self,
        game_state: Dict[str, Any],
        pledges: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Update game state with new pledges: compute aggregates and funded set.

        Mutates game_state in-place.

        Args:
            game_state: Current game state dict
            pledges: Dict mapping agent_id -> parsed pledge dict with "contributions"
        """
        m = game_state["m_projects"]
        costs = game_state["project_costs"]

        # Store current pledges
        game_state["current_pledges"] = pledges

        # Append to round history
        game_state["round_pledges"].append(
            {aid: p.get("contributions", [0.0] * m) for aid, p in pledges.items()}
        )

        # Compute aggregate totals
        aggregates = [0.0] * m
        for aid, pledge in pledges.items():
            contribs = pledge.get("contributions", [0.0] * m)
            for j in range(m):
                aggregates[j] += contribs[j]

        game_state["aggregate_totals"] = aggregates

        # Determine funded set
        funded = []
        for j in range(m):
            if aggregates[j] >= costs[j] - 1e-6:  # small tolerance
                funded.append(j)
        game_state["funded_projects"] = funded

    def compute_final_outcome(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute final utilities from the last round's pledges.

        Args:
            game_state: Game state with current_pledges and funded_projects

        Returns:
            Dict with utilities, funded_projects, contributions
        """
        funded = game_state["funded_projects"]
        valuations = game_state["agent_valuations"]
        pledges = game_state["current_pledges"]

        utilities = {}
        for aid in valuations:
            contribs = pledges.get(aid, {}).get("contributions", [0.0] * game_state["m_projects"])
            utility = 0.0
            for j in funded:
                utility += valuations[aid][j] - contribs[j]
            utilities[aid] = utility

        return {
            "utilities": utilities,
            "funded_projects": funded,
            "contributions": {
                aid: pledges.get(aid, {}).get("contributions", [])
                for aid in valuations
            },
            "aggregate_totals": game_state["aggregate_totals"],
        }

    def check_early_termination(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if pledges have converged for 2 consecutive rounds.

        All agents must have submitted identical pledges (within tolerance)
        for the last 2 rounds.

        Args:
            game_state: Game state with round_pledges history

        Returns:
            True if early termination condition is met
        """
        history = game_state["round_pledges"]
        if len(history) < 2:
            return False

        last = history[-1]
        prev = history[-2]

        # Check that the same agents submitted in both rounds
        if set(last.keys()) != set(prev.keys()):
            return False

        # Check all agents' pledges are close
        for aid in last:
            v_last = np.array(last[aid])
            v_prev = np.array(prev[aid])
            if not np.allclose(v_last, v_prev, atol=0.01):
                return False

        return True

    def get_feedback_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """
        Generate feedback message showing aggregate totals and funded status.

        Args:
            agent_id: ID of agent receiving feedback
            game_state: Current game state (after update_game_state_with_pledges)

        Returns:
            Feedback prompt string
        """
        projects = game_state["projects"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        costs = game_state["project_costs"]

        lines = ["ROUND RESULTS - Aggregate Contributions:", ""]
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            if j in funded:
                lines.append(f"  {proj['name']}: {agg:.2f} / {cost:.2f} -- FUNDED")
            else:
                gap = cost - agg
                pct = (agg / cost * 100) if cost > 0 else 0
                lines.append(f"  {proj['name']}: {agg:.2f} / {cost:.2f} ({pct:.0f}%) -- needs {gap:.2f} more")

        lines.append("")
        funded_names = [projects[j]["name"] for j in funded]
        lines.append(f"Funded projects: {funded_names if funded_names else 'None'}")
        lines.append("")
        lines.append("Consider adjusting your contributions based on these aggregate results.")

        return "\n".join(lines)
