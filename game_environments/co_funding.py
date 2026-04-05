"""
=============================================================================
Co-Funding (Participatory Budgeting) game environment implementation.
=============================================================================

This module implements a participatory budgeting / co-funding negotiation game
where agents submit individual contribution vectors to co-fund threshold public
goods (projects). Projects are funded only when aggregate contributions meet
or exceed the project cost.

Protocol: Propose-and-Vote
  1. Discussion - agents discuss strategy publicly
  2. Thinking - private strategic analysis
  3. Proposal - each agent submits their proposed contribution vector
  4. Voting - agents vote accept/reject on the single joint proposal
  5. Reflection - agents reflect on the round outcome
  6. Repeat until consensus or max rounds

Key differences from Games 1/2:
  - Individual submissions are aggregated into one joint proposal per round
  - Threshold non-linearity -- projects funded only when sum >= cost
  - Final-state utility with optional time discounting
  - No deal by the final round yields zero utility for all agents

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
    - The round's joint proposal aggregates all submitted contribution vectors
    - The joint proposal only takes effect if unanimously accepted
    - Project j is funded if sum_i(x_ij) >= c_j (threshold)
    - Utility: U_i = sum_{j in S} v_ij - sum_{j in S} x_ij
      where S is the set of funded projects
    - If no joint proposal is accepted, utility is 0 for all agents

    Two control parameters:
    - alpha: Preference alignment [0, 1] (cosine similarity of valuations)
    - sigma: Budget abundance scale (0, 1]
      total budget ratio = 0.5 + 0.5 * sigma (from 50% to 100% of total cost)
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

    @staticmethod
    def _round_to_nearest_int(value: float) -> int:
        """Round a scalar to the nearest integer using half-up semantics."""
        return int(np.floor(float(value) + 0.5))

    @staticmethod
    def _format_display_number(value: float) -> str:
        """Render integer-valued floats without trailing .00."""
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.2f}"

    @staticmethod
    def _round_to_integer_values(
        values: np.ndarray,
        total: Optional[int] = None,
    ) -> np.ndarray:
        """Round non-negative values to integers, optionally preserving an exact total."""
        clipped = np.maximum(np.asarray(values, dtype=float), 0.0)

        if total is None:
            return np.floor(clipped + 0.5).astype(int)

        current_sum = clipped.sum()
        if current_sum <= 0.0:
            rounded = np.zeros_like(clipped, dtype=int)
            if rounded.size:
                rounded[0] = total
            return rounded

        scaled = clipped / current_sum * total
        floored = np.floor(scaled).astype(int)
        remainder = int(total - floored.sum())

        if remainder > 0:
            fractional = scaled - floored
            indices = np.argsort(-fractional)[:remainder]
            floored[indices] += 1
        elif remainder < 0:
            fractional = scaled - floored
            positive_indices = np.where(floored > 0)[0]
            indices = positive_indices[
                np.argsort(fractional[positive_indices])[:(-remainder)]
            ]
            floored[indices] -= 1

        return floored

    @staticmethod
    def _pairwise_cosine_error(value_matrix: np.ndarray, target_alpha: float) -> float:
        """Compute total pairwise cosine-similarity error for integer valuation vectors."""
        n_agents = value_matrix.shape[0]
        total_error = 0.0

        for i in range(n_agents):
            norm_i = np.linalg.norm(value_matrix[i])
            if norm_i < 1e-12:
                return float("inf")

            for j in range(i + 1, n_agents):
                norm_j = np.linalg.norm(value_matrix[j])
                if norm_j < 1e-12:
                    return float("inf")
                cosine = np.dot(value_matrix[i], value_matrix[j]) / (norm_i * norm_j)
                total_error += (cosine - target_alpha) ** 2

        return total_error

    @classmethod
    def _refine_integer_valuations_for_target_cosine(
        cls,
        value_matrix: np.ndarray,
        target_alpha: float,
        max_iterations: int = 200,
    ) -> np.ndarray:
        """
        Improve integer-rounded valuations with local search while preserving per-agent sum=100.

        Starting from largest-remainder rounding, move one utility point at a time
        within a single agent's vector if doing so reduces total pairwise cosine error.
        """
        current = np.asarray(value_matrix, dtype=int).copy()
        current_error = cls._pairwise_cosine_error(current, target_alpha)

        for _ in range(max_iterations):
            best_move = None
            best_error = current_error

            for agent_idx in range(current.shape[0]):
                for src_idx in range(current.shape[1]):
                    if current[agent_idx, src_idx] <= 0:
                        continue

                    for dst_idx in range(current.shape[1]):
                        if src_idx == dst_idx:
                            continue

                        candidate = current.copy()
                        candidate[agent_idx, src_idx] -= 1
                        candidate[agent_idx, dst_idx] += 1
                        candidate_error = cls._pairwise_cosine_error(
                            candidate,
                            target_alpha,
                        )

                        if candidate_error + 1e-12 < best_error:
                            best_error = candidate_error
                            best_move = (agent_idx, src_idx, dst_idx)

            if best_move is None:
                break

            agent_idx, src_idx, dst_idx = best_move
            current[agent_idx, src_idx] -= 1
            current[agent_idx, dst_idx] += 1
            current_error = best_error

            if current_error < 1e-12:
                break

        return current

    def _generate_integer_project_costs(self, m_projects: int) -> List[int]:
        """Sample integer project costs uniformly from the feasible cost range."""
        min_cost = int(np.ceil(self.config.c_min))
        max_cost = int(np.floor(self.config.c_max))

        if min_cost > max_cost:
            raise ValueError(
                "CoFundingGame requires at least one integer project cost in "
                f"[c_min, c_max], got c_min={self.config.c_min}, c_max={self.config.c_max}"
            )

        return self._rng.randint(
            min_cost,
            max_cost + 1,
            size=m_projects,
        ).astype(int).tolist()

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

        # Generate integer project costs uniformly over the feasible cost range.
        project_costs = self._generate_integer_project_costs(m)
        total_cost = int(sum(project_costs))

        # Generate budgets proportional to project scale while preventing
        # pathological under-provision at very low sigma.
        # sigma interpolates from "can fund ~half the total cost" to "can fund all."
        budget_ratio = self.config.sigma
        total_budget = budget_ratio * total_cost
        # Keep budgets symmetric across agents after integerization.
        per_agent_budget = self._round_to_nearest_int(total_budget / n)
        total_budget = per_agent_budget * n

        # Generate valuation vectors with target cosine similarity alpha
        valuations = self._generate_valuations(n, m)

        # Create project list
        projects = []
        for j in range(m):
            name = self.PROJECT_NAMES[j] if j < len(self.PROJECT_NAMES) else f"Project_{j+1}"
            projects.append({"name": name, "cost": int(project_costs[j])})

        # Map to agent IDs
        agent_ids = []
        agent_valuations = {}
        agent_budgets = {}
        for i, agent in enumerate(agents):
            aid = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
            agent_ids.append(aid)
            agent_valuations[aid] = valuations[i].tolist()
            agent_budgets[aid] = per_agent_budget

        return {
            "projects": projects,
            "m_projects": m,
            "project_costs": [int(c) for c in project_costs],
            "total_cost": total_cost,
            "agent_budgets": agent_budgets,
            "total_budget": total_budget,
            "agent_valuations": agent_valuations,
            "parameters": {
                "alpha": self.config.alpha,
                "sigma": self.config.sigma,
                "budget_ratio": round(budget_ratio, 4),
                "discussion_transparency": self.config.discussion_transparency,
                "enable_commit_vote": self.config.enable_commit_vote,
                "enable_time_discount": self.config.enable_time_discount,
                "gamma_discount": self.config.gamma_discount,
            },
            "game_type": "co_funding",
            "pledge_mode": self.config.pledge_mode,
            # Mutable tracking fields
            "round_pledges": [],
            "current_pledges": {},
            "aggregate_totals": [0.0] * m,
            "funded_projects": [],
            "joint_plans": {},  # stores latest joint plans per agent (joint mode only)
            "accepted_proposal": None,
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
            valuations: Shape (n_agents, m_projects), integer-valued rows that each sum to 100
        """
        alpha = self.config.alpha
        M = m_projects
        dir_alpha = 2.0  # Dirichlet concentration

        if alpha == 1.0:
            # Special case: identical valuations
            v0 = self._rng.dirichlet(np.full(M, dir_alpha)) * 100.0
            v0 = self._round_to_integer_values(v0, total=100)
            return np.tile(v0, (n_agents, 1))

        if n_agents == 2:
            valuations = self._generate_valuations_2agents(M, alpha, dir_alpha)
        else:
            valuations = self._generate_valuations_nagents(n_agents, M, alpha, dir_alpha)

        integer_valuations = np.vstack([
            self._round_to_integer_values(row, total=100)
            for row in valuations
        ])
        return self._refine_integer_valuations_for_target_cosine(
            integer_valuations,
            alpha,
        )

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
        """Generate valuations for N>2 agents with all-pairs target cosine similarity.

        Uses joint SLSQP optimization over all N vectors simultaneously,
        minimizing the sum of squared errors across all N(N-1)/2 pairs.
        """
        dim = n_agents * M

        def objective(x):
            vecs = x.reshape(n_agents, M)
            total_err = 0.0
            for i in range(n_agents):
                ni = np.linalg.norm(vecs[i])
                if ni < 1e-12:
                    return 1e10
                for j in range(i + 1, n_agents):
                    nj = np.linalg.norm(vecs[j])
                    if nj < 1e-12:
                        return 1e10
                    cos_sim = np.dot(vecs[i], vecs[j]) / (ni * nj)
                    total_err += (cos_sim - alpha) ** 2
            return total_err

        constraints = [
            {'type': 'eq', 'fun': lambda x, i=i: np.sum(x[i * M:(i + 1) * M]) - 100.0}
            for i in range(n_agents)
        ]
        bounds = [(0.0, 100.0)] * dim

        best_x = None
        best_error = float('inf')
        max_attempts = 30

        for _ in range(max_attempts):
            conc = self._rng.uniform(0.5, 3.0)
            x0 = np.concatenate([
                self._rng.dirichlet(np.full(M, conc)) * 100.0
                for _ in range(n_agents)
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 3000, 'ftol': 1e-14}
                )

            error = objective(result.x)
            if error < best_error:
                best_error = error
                best_x = result.x.copy()
            if best_error < 1e-10:
                break

        if best_x is None or best_error > 0.001:
            raise RuntimeError(
                f"Failed to generate valuations with all-pairs cosine similarity "
                f"alpha={alpha} for {n_agents} agents after "
                f"{max_attempts} attempts. Best error: {best_error:.6f}"
            )

        valuations = best_x.reshape(n_agents, M)
        for i in range(n_agents):
            valuations[i] = np.maximum(valuations[i], 0.0)
            valuations[i] = valuations[i] / valuations[i].sum() * 100.0

        return valuations

    # ---- Abstract method implementations ----

    def _get_parties_phrase(self) -> str:
        """Create clearer phrasing for 2-agent co-funding games."""
        if self.config.n_agents == 2:
            return "one other participant"
        return f"{self.config.n_agents - 1} other participants"

    def _get_rules_block(self, game_state: Dict[str, Any]) -> str:
        """Build the shared setup/rules block for co-funding."""
        projects = game_state["projects"]

        projects_text = "\n".join([
            f"  - {p['name']}: cost = {self._format_display_number(p['cost'])}"
            for p in projects
        ])

        parties_phrase = self._get_parties_phrase()

        return f"""Welcome to the Participatory Budgeting (Co-Funding) Game!

You are participating in a co-funding exercise with {parties_phrase} to fund public projects. Here is your full setup information:

**PROJECTS AVAILABLE FOR FUNDING:**
{projects_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} participants (including you)
- The game lasts up to {self.config.t_rounds} rounds
- This message is the one-time setup phase
- Each round follows a Propose-and-Vote cycle:
  Discussion -> Private Thinking -> Proposal -> Voting -> Reflection

**PRIVATE INFORMATION:**
- You have a SECRET contribution budget
- You have SECRET project valuations
- These budget and valuation details are PRIVATE and specific to you

**HOW IT WORKS:**
- Each participant has a PRIVATE BUDGET they can allocate across projects
- In the PROPOSAL phase, each participant submits a contribution vector — how much they propose contributing to each project
- Those submitted vectors are combined into ONE JOINT PROPOSAL for the round
- In the VOTING phase, every participant votes accept/reject on that joint proposal
- A project is FUNDED if and only if the TOTAL contributions from ALL participants meet or exceed its cost
- **ALL-OR-NOTHING**: Funding is binary — a project either reaches its full cost threshold (funded) or it doesn't (unfunded). There is no partial benefit from contributing to a project that falls short of its threshold.
- Contributions to UNFUNDED projects do not reduce your utility

**WHAT YOU CAN SEE:**
- During discussion, you may see previous-round aggregate project status
- During voting, you see the aggregate project status the joint proposal would create, but not other participants' individual contribution vectors
- You do NOT see other participants' private preferences

**YOUR UTILITY:**
- Utility = (sum of your valuations for funded projects) - (your contributions to funded projects)
- You gain value from funded projects but pay for your contributions to them
- **IMPORTANT**: If your contribution to a funded project exceeds your valuation, your net utility from that project is NEGATIVE
- Contributions to unfunded projects cost you nothing
- If no joint proposal is unanimously accepted by the final round, everyone gets zero utility

**REWARD DISCOUNTING:**
- If time discounting is enabled, your utility from the final funded outcome is multiplied by gamma^(round - 1)
- Round 1 rewards: 100% of utility
- Round 2 rewards: {self.config.gamma_discount * 100:.0f}% of utility
- Round 3 rewards: {self.config.gamma_discount ** 2 * 100:.0f}% of utility
- The longer it takes to settle on a final funding outcome, the less valuable that outcome becomes
- If time discounting is disabled for this run, no round-based multiplier is applied

**IMPORTANT RULES:**
- Time discounting: {"enabled" if self.config.enable_time_discount else "disabled"}
- Discount factor (if enabled): gamma = {self.config.gamma_discount}
- Your goal: maximize your utility by strategically choosing contributions

**BUDGET CONSTRAINT:**
- The combined budgets of all participants may NOT be sufficient to fund all projects
- You MUST prioritize — coordinate on a subset of projects you can collectively afford to fully fund
"""

    def _get_private_preferences_block(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Build the per-agent private preference block."""
        projects = game_state["projects"]
        valuations = game_state["agent_valuations"][agent_id]
        budget = game_state["agent_budgets"][agent_id]
        costs = game_state["project_costs"]
        budget_text = self._format_display_number(budget)

        lines = ["LOCKED PRIVATE PREFERENCES", ""]
        lines.append(
            f"{agent_id}, you have been assigned the following SECRET co-funding preferences:"
        )
        lines.append("")
        lines.append(
            f"**YOUR PRIVATE BUDGET:** {budget_text} (maximum total you can contribute across all projects)"
        )
        lines.append("")
        lines.append("**PROJECT DETAILS AND YOUR VALUATIONS:**")

        for j, (proj, val, cost) in enumerate(zip(projects, valuations, costs)):
            priority = "HIGH" if val > 30 else "Medium" if val > 15 else "Low"
            cost_text = self._format_display_number(cost)
            val_text = self._format_display_number(val)
            lines.append(f"  {proj['name']} (cost: {cost_text}): Your valuation = {val_text} ({priority} priority)")

        lines.append("")
        lines.append(f"**TOTAL VALUATIONS:** {self._format_display_number(sum(valuations))}")
        lines.append(f"**TOTAL PROJECT COSTS:** {self._format_display_number(game_state['total_cost'])}")
        lines.append(f"**TOTAL BUDGET (all participants):** {self._format_display_number(game_state['total_budget'])}")
        _coverage = round(game_state['total_budget'] / game_state['total_cost'] * 100) if game_state['total_cost'] > 0 else 0
        lines.append(f"**COLLECTIVE COVERAGE:** {_coverage}% of total project costs — you cannot fund all projects; coordinate on a subset")
        lines.append("")
        lines.append("**HOW YOUR UTILITY IS COMPUTED:**")
        lines.append("- For each FUNDED project: your_utility = your_valuation \u2212 your_contribution (negative if you over-contribute)")
        lines.append("- For UNFUNDED projects: you do not pay that contribution in the final outcome. But within a proposal, budget assigned to one project is not available for any other project.")
        lines.append("- Total utility = sum of (valuation \u2212 contribution) across ALL funded projects, including projects funded entirely by others (where your contribution = 0, giving you full valuation as free utility)")
        lines.append("")
        lines.append("**STRATEGIC INSIGHT:**")
        lines.append("- Focus contributions on projects you value highly")
        lines.append("- Coordinate with others to meet project cost thresholds")
        lines.append("- Don't over-contribute to projects others will fund")

        return "\n".join(lines)

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate co-funding game rules explanation."""
        rules_block = self._get_rules_block(game_state)
        return f"""{rules_block}

Please acknowledge that you understand these rules and are ready to participate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate preference assignment prompt showing budget, costs, and valuations."""
        preferences_block = self._get_private_preferences_block(agent_id, game_state)
        return f"""{preferences_block}

Please acknowledge that you understand your preferences and budget."""

    def uses_combined_setup_phase(self) -> bool:
        """Co-Funding merges private preference assignment into setup."""
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
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private budget that was assigned to you, along with the project costs and your project valuations."""

    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate proposal prompt for an agent's contribution vector."""
        projects = game_state["projects"]
        m = len(projects)
        budget = game_state["agent_budgets"][agent_id]
        valuations = game_state["agent_valuations"][agent_id]
        costs = game_state["project_costs"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        pledge_mode = game_state.get("pledge_mode", "individual")
        budget_text = self._format_display_number(budget)

        # Own contributions from last round (so agent sees their own prior pledge at submission time)
        own_prev = game_state.get("current_pledges", {}).get(agent_id, {}).get("contributions", [0.0] * m)
        if len(own_prev) != m:
            own_prev = [0.0] * m

        project_lines = []
        for j, (proj, cost, val, agg) in enumerate(zip(projects, costs, valuations, aggregates)):
            status = "PROVISIONALLY FUNDED" if j in funded else f"needs {max(0, cost - agg):.2f} more"
            cost_text = self._format_display_number(cost)
            val_text = self._format_display_number(val)
            project_lines.append(
                f"  {j}: {proj['name']} (cost={cost_text}, your_val={val_text}, "
                f"aggregate={agg:.2f}, your_prev={own_prev[j]:.2f}, {status})"
            )

        projects_text = "\n".join(project_lines)

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        all_budgets = game_state["agent_budgets"]

        if pledge_mode == "joint":
            # Legacy joint mode: agents submit contribution vectors for all participants
            # Build example JSON with all agent IDs
            agent_example_entries = []
            for aid in sorted(all_budgets.keys()):
                agent_example_entries.append(
                    f'        "{aid}": [5.0, 10.0, 0.0, 8.0, 2.0]'
                )
            example_contributions = ",\n".join(agent_example_entries)

            budget_lines = "\n".join(
                f"  - {aid}: {self._format_display_number(b)}"
                for aid, b in sorted(all_budgets.items())
            )

            format_section = f"""**Instructions:**
Submit a JOINT FUNDING PLAN: a dictionary specifying contribution vectors for ALL participants.
Your plan proposes how every participant (including yourself) should allocate their budget.
The round's JOINT PROPOSAL will be constructed from the self-assignment that each participant submits.

**Participant budgets:**
{budget_lines}

Respond with ONLY a JSON object in this exact format:
{{
    "contributions": {{
{example_contributions}
    }},
    "reasoning": "Brief explanation of your joint funding plan"
}}

**Rules:**
- "contributions" must be a dictionary with one entry per participant
- Each entry must be an array of exactly {m} non-negative values (one per project)
- Each participant's total contributions must not exceed their budget
- Contributions to unfunded projects will be refunded"""
        else:
            # Individual mode (default): each agent submits only their own contribution vector
            format_section = f"""**Instructions:**
Submit a contribution vector specifying how much YOU propose contributing to each project.
All participants' submitted vectors will be combined into one JOINT PROPOSAL before voting.

Respond with ONLY a JSON object in this exact format:
{{
    "contributions": [5.0, 10.0, 0.0, 8.0, 2.0],
    "reasoning": "Brief explanation of your contribution strategy"
}}

**Rules:**
- The "contributions" array must have exactly {m} values (one per project)
- Each value must be non-negative (>= 0)
- The sum of all contributions must not exceed your budget ({budget_text})
- Contributions to unfunded projects will not reduce your utility"""

        return f"""Please submit your proposal for Round {round_num}/{self.config.t_rounds}.

**YOUR BUDGET:** {budget_text}

**PROJECT STATUS:**
{projects_text}

**Provisionally funded projects (PREVIOUS ROUND):** {[projects[j]['name'] for j in funded] if funded else 'None'}
**NOTE:** All status above reflects the PREVIOUS ROUND only; projects that were provisionally funded then are not automatically funded this round unless enough contributions are proposed again.{reasoning_instruction}

{format_section}"""

    def parse_proposal(
        self,
        response: str,
        agent_id: str,
        game_state: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """Parse pledge response into contribution vector.

        In joint mode, the agent submits a dict of contributions for all agents.
        We extract the full joint_plan and use the agent's self-assignment as
        the effective contributions.

        In individual mode, the agent submits a single contribution vector.
        """
        m = game_state["m_projects"]
        pledge_mode = game_state.get("pledge_mode", "individual")

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

            raw_contributions = parsed.get("contributions", [])

            if pledge_mode == "joint" and isinstance(raw_contributions, dict):
                # Joint mode: contributions is a dict {agent_id: [values...]}
                joint_plan = {}
                for aid, vals in raw_contributions.items():
                    if not isinstance(vals, list):
                        vals = [0.0] * m
                    if len(vals) < m:
                        vals = vals + [0.0] * (m - len(vals))
                    elif len(vals) > m:
                        vals = vals[:m]
                    joint_plan[aid] = [max(0.0, float(v)) for v in vals]

                # Extract self-assignment
                self_contributions = joint_plan.get(agent_id, [0.0] * m)

                return {
                    "contributions": self_contributions,
                    "joint_plan": joint_plan,
                    "reasoning": parsed.get("reasoning", ""),
                    "proposed_by": agent_id,
                }
            else:
                # Individual mode (or joint mode with list fallback)
                contributions = raw_contributions
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
        """Validate pledge: non-negative, within budget, correct length.

        In joint mode, also validates the joint_plan if present: each agent's
        entry must have correct length, non-negative values, and total within
        that agent's budget.
        """
        contributions = proposal.get("contributions", [])
        agent_id = proposal.get("proposed_by", "")
        m = game_state["m_projects"]
        budget = game_state["agent_budgets"].get(agent_id, 0.0)
        pledge_mode = game_state.get("pledge_mode", "individual")

        # Validate the effective contributions (self-assignment)
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

        # In joint mode, validate the full joint_plan
        if pledge_mode == "joint" and "joint_plan" in proposal:
            joint_plan = proposal["joint_plan"]
            all_budgets = game_state["agent_budgets"]

            for aid, plan_contribs in joint_plan.items():
                if len(plan_contribs) != m:
                    return False
                for v in plan_contribs:
                    try:
                        if float(v) < -1e-9:
                            return False
                    except (ValueError, TypeError):
                        return False
                aid_budget = all_budgets.get(aid, 0.0)
                if sum(float(v) for v in plan_contribs) > aid_budget + 1e-6:
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

        where S is the funded set. If time discounting is enabled:

        discounted_U_i = U_i * gamma^(round_num - 1)
        """
        valuations = game_state["agent_valuations"][agent_id]

        if "contributions_by_agent" in proposal:
            funded = proposal.get("funded_projects", [])
            contributions = proposal.get("contributions_by_agent", {}).get(
                agent_id, [0.0] * game_state["m_projects"]
            )
        else:
            funded = game_state.get("funded_projects", [])
            contributions = proposal.get("contributions", [0.0] * game_state["m_projects"])

        raw_utility = 0.0
        for j in funded:
            raw_utility += valuations[j] - contributions[j]

        if self.config.enable_time_discount:
            discount_factor = self.config.gamma_discount ** max(0, round_num - 1)
            return raw_utility * discount_factor
        return raw_utility

    def get_discussion_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate discussion prompt for co-funding."""
        projects = game_state["projects"]
        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        costs = game_state["project_costs"]
        all_budgets = game_state.get("agent_budgets", {})
        current_pledges = game_state.get("current_pledges", {})
        transparency_mode = getattr(self.config, "discussion_transparency", "own")
        m = len(projects)

        # Previous-round attribution from this agent's perspective.
        own_prev = current_pledges.get(agent_id, {}).get("contributions", [0.0] * m)
        if len(own_prev) != m:
            own_prev = [0.0] * m
        others_prev = [max(0.0, aggregates[j] - own_prev[j]) for j in range(m)]

        # Project status
        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            cost_text = self._format_display_number(cost)
            if transparency_mode == "aggregate":
                if j in funded:
                    status_lines.append(f"  {proj['name']}: PROVISIONALLY FUNDED (aggregate={agg:.2f} >= cost={cost_text})")
                else:
                    gap = cost - agg
                    status_lines.append(
                        f"  {proj['name']}: needs {gap:.2f} more (aggregate={agg:.2f} / cost={cost_text})"
                    )
            else:
                if j in funded:
                    line = (
                        f"  {proj['name']}: PROVISIONALLY FUNDED (aggregate={agg:.2f} >= cost={cost_text}); "
                        f"your_prev={own_prev[j]:.2f}, others_prev={others_prev[j]:.2f}"
                    )
                else:
                    gap = cost - agg
                    line = (
                        f"  {proj['name']}: needs {gap:.2f} more (aggregate={agg:.2f} / cost={cost_text}); "
                        f"your_prev={own_prev[j]:.2f}, others_prev={others_prev[j]:.2f}"
                    )
                status_lines.append(line)
        status_text = "\n".join(status_lines)

        extra_transparency_block = ""
        if transparency_mode != "aggregate":
            attribution_section = ""
            if transparency_mode == "full":
                attribution_lines = ["**PREVIOUS ROUND PROJECT ATTRIBUTION (who pledged what):**"]
                if not current_pledges:
                    attribution_lines.append("- No prior pledges yet (round 1).")
                else:
                    for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
                        cost_text = self._format_display_number(cost)
                        per_agent = []
                        for aid in sorted(all_budgets.keys()):
                            contribs = current_pledges.get(aid, {}).get("contributions", [0.0] * m)
                            if len(contribs) != m:
                                contribs = [0.0] * m
                            per_agent.append(f"{aid}={contribs[j]:.2f}")
                        funded_tag = "PROVISIONALLY FUNDED" if j in funded else "UNFUNDED"
                        attribution_lines.append(
                            f"- {proj['name']}: {', '.join(per_agent)} | "
                            f"aggregate={agg:.2f}/{cost_text} ({funded_tag})"
                        )
                attribution_section = "\n\n" + "\n".join(attribution_lines)

            budget_lines = []
            for aid in sorted(all_budgets.keys()):
                prev = current_pledges.get(aid, {}).get("contributions", [0.0] * m)
                if len(prev) != m:
                    prev = [0.0] * m
                spent_prev = sum(float(x) for x in prev)
                remaining_prev = max(0.0, all_budgets[aid] - spent_prev)
                label = " (you)" if aid == agent_id else ""
                budget_lines.append(
                    f"  {aid}{label}: budget={self._format_display_number(all_budgets[aid])}, "
                    f"prev_round_pledged={spent_prev:.2f}, prev_round_unallocated={remaining_prev:.2f}"
                )
            budget_section = "\n".join(budget_lines)
            extra_transparency_block = f"""

**IMPORTANT: any provisionally funded status above reflects the PREVIOUS ROUND only.**
If contributions change this round, projects that were provisionally funded in the previous round may no longer clear their cost threshold, so any support you still want must be proposed again.

**PREVIOUS ROUND BUDGET USAGE (before this round's revision):**
{budget_section}{attribution_section}"""

        # Inject prior turns so each speaker sees what has been said this round
        history_section = ""
        if discussion_history:
            history_section = "\n**DISCUSSION SO FAR THIS ROUND:**\n"
            for msg in discussion_history:
                history_section += f"{msg}\n\n"
            history_section += "---\n"

        if round_num == 1 and not discussion_history:
            context = """**DISCUSSION OBJECTIVES:**
- Signal which projects you believe are most valuable to fund
- Understand other participants' priorities
- Coordinate to avoid spreading contributions too thin
- Identify projects with enough collective support to be funded

You are the first to speak. Share your initial thoughts on which projects to prioritize."""
        elif discussion_history:
            context = """**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- React to other participants' stated priorities
- Coordinate on which projects to focus collective contributions
- Signal your own funding intentions

Keep the discussion focused on reaching a funded consensus."""
        else:
            urgency = ""
            if round_num >= max_rounds - 1:
                urgency = "\n**TIME PRESSURE**: Limited rounds remaining!"

            context = f"""Previous round's joint proposal did not achieve unanimous acceptance.{urgency}

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

**Provisionally funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}
{extra_transparency_block}
{history_section}
{context}{reasoning_instruction}"""

    def get_voting_prompt(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int,
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate a voting prompt for the round's single joint proposal."""
        projects = game_state["projects"]
        costs = game_state["project_costs"]
        aggregates = proposal.get("aggregate_totals", [0.0] * len(projects))
        funded = set(proposal.get("funded_projects", []))

        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            status = "PROVISIONALLY FUNDED" if j in funded else f"needs {max(0.0, cost - agg):.2f} more"
            cost_text = self._format_display_number(cost)
            status_lines.append(
                f"  {proj['name']}: aggregate={agg:.2f} / cost={cost_text} ({status})"
            )

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""The following JOINT FUNDING PROPOSAL has been constructed from all submitted contribution vectors this round:

**Aggregate project status if accepted:**
{chr(10).join(status_lines)}

Please vote on this proposal. Consider:
- Which projects would be provisionally funded if this proposal is accepted
- How much you would contribute under this proposal
- Your utility from the resulting funded set after subtracting your own contributions
- If no joint proposal is unanimously accepted by the final round, your utility is 0{reasoning_instruction}

Respond with ONLY a JSON object in this exact format:
{{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}}

Vote must be either "accept" or "reject"."""

    def get_commit_vote_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
    ) -> str:
        """Generate post-pledge commit vote prompt (yay/nay)."""
        projects = game_state["projects"]
        costs = game_state["project_costs"]
        aggregates = game_state["aggregate_totals"]
        funded = set(game_state.get("funded_projects", []))

        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            status = "PROVISIONALLY FUNDED" if j in funded else f"needs {max(0.0, cost - agg):.2f} more"
            cost_text = self._format_display_number(cost)
            status_lines.append(
                f"  {proj['name']}: aggregate={agg:.2f} / cost={cost_text} ({status})"
            )

        return f"""POST-PLEDGE COMMIT VOTE - Round {round_num}/{max_rounds}

You are voting on whether to LOCK IN the current aggregate project status immediately.

**Current aggregate project status:**
{chr(10).join(status_lines)}

Vote **yay** if you are satisfied with the current aggregate project status and your own current contribution vector, and want to finalize now.
Vote **nay** if you want another revision round to improve contributions.

**CONSEQUENCE:** If ALL participants vote yay, the game ends immediately with this round's contributions as the final outcome. If ANY participant votes nay, another revision round occurs.

Respond with ONLY JSON:
{{
    "commit_vote": "yay",
    "reasoning": "brief explanation"
}}"""

    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate private thinking prompt for co-funding."""
        projects = game_state["projects"]
        valuations = game_state["agent_valuations"][agent_id]
        budget = game_state["agent_budgets"][agent_id]
        costs = game_state["project_costs"]
        aggregates = game_state["aggregate_totals"]
        budget_text = self._format_display_number(budget)

        # Own contributions if available
        own_contribs = game_state.get("current_pledges", {}).get(agent_id, {}).get("contributions", [0.0] * len(projects))

        preference_lines = [
            (
                f"  {projects[i]['name']} "
                f"(val={self._format_display_number(valuations[i])}, cost={self._format_display_number(costs[i])})"
            )
            for i in range(len(projects))
        ]

        # Include what was said in the discussion this round
        discussion_section = ""
        if discussion_history:
            discussion_section = "\n**DISCUSSION THIS ROUND:**\n"
            for msg in discussion_history:
                discussion_section += f"{msg}\n\n"
            discussion_section += "---\n"

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
{discussion_section}
**YOUR SITUATION:**
- Budget: {budget_text}
- Your current contributions: {[round(c, 2) for c in own_contribs]}
- Aggregate totals: {[round(a, 2) for a in aggregates]}

**YOUR FULL PREFERENCE REMINDER:**
{chr(10).join(preference_lines)}

**STRATEGIC ANALYSIS:**
1. Which projects are viable to fund given current aggregates?
2. Where can you shift contributions for maximum impact?
3. Based on the discussion above, what are other participants likely to do?
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
        """Format proposal for display."""
        projects = game_state["projects"]
        if "contributions_by_agent" in proposal:
            lines = ["JOINT PROPOSAL (aggregated from all participants):"]
            contributions_by_agent = proposal.get("contributions_by_agent", {})
            for aid in sorted(contributions_by_agent.keys()):
                contribs = contributions_by_agent[aid]
                lines.append(f"  {aid}: {[round(float(x), 2) for x in contribs]}")
            lines.append("  Aggregate project status:")
            aggregates = proposal.get("aggregate_totals", [0.0] * len(projects))
            funded = set(proposal.get("funded_projects", []))
            for j, proj in enumerate(projects):
                agg = aggregates[j] if j < len(aggregates) else 0.0
                cost = game_state["project_costs"][j]
                status = "FUNDED" if j in funded else f"needs {max(0.0, cost - agg):.2f} more"
                cost_text = self._format_display_number(cost)
                lines.append(
                    f"    {proj['name']}: aggregate={agg:.2f} / cost={cost_text} ({status})"
                )
            lines.append(f"  Reasoning: {proposal.get('reasoning', 'None')}")
            return "\n".join(lines)

        contributions = proposal.get("contributions", [])
        lines = [f"PROPOSAL (by {proposal.get('proposed_by', 'Unknown')}):"]

        for j, (proj, contrib) in enumerate(zip(projects, contributions)):
            lines.append(f"  {proj['name']}: {contrib:.2f}")

        total = sum(contributions)
        lines.append(f"  Total proposed: {total:.2f}")
        lines.append(f"  Reasoning: {proposal.get('reasoning', 'None')}")

        return "\n".join(lines)

    def get_game_type(self) -> GameType:
        """Return game type identifier."""
        return GameType.CO_FUNDING

    def get_protocol_type(self) -> str:
        """Return Propose-and-Vote protocol type."""
        return "propose_and_vote"

    def prepare_proposals_for_voting(
        self,
        proposals: List[Dict[str, Any]],
        game_state: Dict[str, Any],
        round_num: int,
    ) -> List[Dict[str, Any]]:
        """Aggregate submitted per-agent contributions into one joint proposal."""
        m = game_state["m_projects"]
        agent_ids = sorted(game_state["agent_budgets"].keys())
        current_pledges: Dict[str, Dict[str, Any]] = {}

        for aid in agent_ids:
            matching = next((p for p in proposals if p.get("proposed_by") == aid), None)
            if matching is None:
                current_pledges[aid] = {
                    "contributions": [0.0] * m,
                    "reasoning": "Missing proposal; defaulted to zero vector",
                    "proposed_by": aid,
                }
            else:
                current_pledges[aid] = {
                    "contributions": list(matching.get("contributions", [0.0] * m)),
                    "reasoning": matching.get("reasoning", ""),
                    "proposed_by": aid,
                }

        self.update_game_state_with_pledges(game_state, current_pledges)
        current_pledges = game_state["current_pledges"]

        return [{
            "contributions_by_agent": {
                aid: current_pledges[aid]["contributions"] for aid in agent_ids
            },
            "aggregate_totals": list(game_state["aggregate_totals"]),
            "funded_projects": list(game_state["funded_projects"]),
            "reasoning": "System-aggregated joint proposal formed from all submitted per-agent contribution vectors.",
            "proposed_by": "system",
            "round": round_num,
        }]

    def record_accepted_proposal(
        self,
        game_state: Dict[str, Any],
        proposal: Dict[str, Any],
    ) -> None:
        """Persist the accepted joint proposal for final utility calculation."""
        if "contributions_by_agent" not in proposal:
            return

        m = game_state["m_projects"]
        budgets = game_state["agent_budgets"]
        costs = game_state["project_costs"]
        raw_contributions_by_agent = proposal.get("contributions_by_agent", {})
        contributions_by_agent = {}
        accepted_adjustments = {}

        for aid in sorted(budgets.keys()):
            result = self._sanitize_contribution_vector(
                raw_contributions_by_agent.get(aid, [0.0] * m),
                budgets.get(aid, 0.0),
                m,
            )
            contributions_by_agent[aid] = result["contributions"]
            if result["adjustment"] is not None:
                accepted_adjustments[aid] = result["adjustment"]

        aggregate_totals = [0.0] * m
        for contribs in contributions_by_agent.values():
            for j in range(m):
                aggregate_totals[j] += contribs[j]

        funded_projects = [
            j for j in range(m) if aggregate_totals[j] >= costs[j] - 1e-6
        ]

        game_state["accepted_proposal"] = {
            "contributions_by_agent": contributions_by_agent,
            "aggregate_totals": aggregate_totals,
            "funded_projects": funded_projects,
        }
        if accepted_adjustments:
            game_state["accepted_proposal"]["auto_corrected"] = accepted_adjustments

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
        raw_utility = sum(valuations[j] - own_contribs[j] for j in funded) if funded else 0.0
        if self.config.enable_time_discount:
            discount_factor = self.config.gamma_discount ** max(0, round_num - 1)
        else:
            discount_factor = 1.0
        utility = raw_utility * discount_factor

        status_lines = []
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            status = "PROVISIONALLY FUNDED" if j in funded else f"gap={cost - agg:.2f}"
            cost_text = self._format_display_number(cost)
            status_lines.append(f"  {proj['name']}: aggregate={agg:.2f}, cost={cost_text} ({status})")

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Reflect on the outcome of Round {round_num}.

**CURRENT STATUS:**
{chr(10).join(status_lines)}

**Provisionally funded projects:** {[projects[j]['name'] for j in funded] if funded else 'None'}
**Vote outcome this round:** {"accepted unanimously" if tabulation_result.get("consensus_reached", False) else "not accepted unanimously"}
**Your utility under this round's joint proposal:** {utility:.2f}
**Raw utility (before discount):** {raw_utility:.2f}
**Discount factor this round:** {discount_factor:.2f}

Consider what adjustments to your contributions might improve the outcome.
- Are there projects close to being funded that deserve more support?
- Are you over-contributing to already-funded projects?
- Should you shift focus to different projects?{reasoning_instruction}"""

    # ---- Co-funding specific helper methods ----

    def _sanitize_contribution_vector(
        self,
        raw_contributions: Any,
        budget: float,
        m: int,
    ) -> Dict[str, Any]:
        """Clamp a contribution vector to valid game constraints without failing the round."""
        if not isinstance(raw_contributions, list):
            raw_contributions = []

        try:
            budget_value = max(0.0, float(budget))
        except (TypeError, ValueError):
            budget_value = 0.0

        contributions = []
        for j in range(m):
            value = raw_contributions[j] if j < len(raw_contributions) else 0.0
            try:
                contributions.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                contributions.append(0.0)

        total = sum(contributions)
        if total <= budget_value + 1e-6 or total <= 1e-12:
            return {"contributions": contributions, "adjustment": None}

        scale = budget_value / total if budget_value > 0 else 0.0
        scaled = [v * scale for v in contributions]
        return {
            "contributions": scaled,
            "adjustment": {
                "reason": "scaled_to_budget",
                "original_total": total,
                "budget": budget_value,
                "scale_factor": scale,
            },
        }

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
        budgets = game_state["agent_budgets"]

        sanitized_pledges = {}
        joint_plans = {}
        for aid, pledge in pledges.items():
            clean = dict(pledge)
            own_result = self._sanitize_contribution_vector(
                clean.get("contributions", []),
                budgets.get(aid, 0.0),
                m,
            )
            clean["contributions"] = own_result["contributions"]
            if own_result["adjustment"] is not None:
                clean["auto_corrected"] = own_result["adjustment"]

            if "joint_plan" in clean and isinstance(clean["joint_plan"], dict):
                sanitized_joint_plan = {}
                joint_adjustments = {}
                for plan_aid, plan_contribs in clean["joint_plan"].items():
                    plan_result = self._sanitize_contribution_vector(
                        plan_contribs,
                        budgets.get(plan_aid, 0.0),
                        m,
                    )
                    sanitized_joint_plan[plan_aid] = plan_result["contributions"]
                    if plan_result["adjustment"] is not None:
                        joint_adjustments[plan_aid] = plan_result["adjustment"]

                clean["joint_plan"] = sanitized_joint_plan
                joint_plans[aid] = sanitized_joint_plan

                if aid in sanitized_joint_plan:
                    clean["contributions"] = sanitized_joint_plan[aid]
                    if aid in joint_adjustments:
                        clean["auto_corrected"] = joint_adjustments[aid]

                if joint_adjustments:
                    clean["joint_plan_auto_corrected"] = joint_adjustments

            sanitized_pledges[aid] = clean

        # Store current pledges
        game_state["current_pledges"] = sanitized_pledges
        game_state["accepted_proposal"] = None
        game_state["joint_plans"] = joint_plans

        # Append to round history
        game_state["round_pledges"].append(
            {aid: p.get("contributions", [0.0] * m) for aid, p in sanitized_pledges.items()}
        )

        # Compute aggregate totals
        aggregates = [0.0] * m
        for aid, pledge in sanitized_pledges.items():
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

    def compute_final_outcome(
        self,
        game_state: Dict[str, Any],
        final_round: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute final utilities from the accepted joint proposal, if any.

        Args:
            game_state: Game state with accepted_proposal and preference data
            final_round: Round in which outcome is finalized. If None, inferred
                from pledge history length.

        Returns:
            Dict with utilities, funded_projects, contributions
        """
        valuations = game_state["agent_valuations"]
        accepted_proposal = game_state.get("accepted_proposal")
        resolved_round = final_round if final_round is not None else max(1, len(game_state.get("round_pledges", [])))
        discount_factor = (
            self.config.gamma_discount ** max(0, resolved_round - 1)
            if self.config.enable_time_discount
            else 1.0
        )

        if not accepted_proposal:
            zero_utilities = {aid: 0.0 for aid in valuations}
            zero_contributions = {
                aid: [0.0] * game_state["m_projects"] for aid in valuations
            }
            return {
                "utilities": zero_utilities,
                "raw_utilities": zero_utilities.copy(),
                "discount_factor": discount_factor,
                "final_round": resolved_round,
                "funded_projects": [],
                "contributions": zero_contributions,
                "aggregate_totals": [0.0] * game_state["m_projects"],
            }

        utilities = {}
        raw_utilities = {}
        for aid in valuations:
            contribs = accepted_proposal.get("contributions_by_agent", {}).get(
                aid, [0.0] * game_state["m_projects"]
            )
            raw_utility = 0.0
            for j in accepted_proposal.get("funded_projects", []):
                raw_utility += valuations[aid][j] - contribs[j]
            raw_utilities[aid] = raw_utility
            utilities[aid] = raw_utility * discount_factor

        return {
            "utilities": utilities,
            "raw_utilities": raw_utilities,
            "discount_factor": discount_factor,
            "final_round": resolved_round,
            "funded_projects": accepted_proposal.get("funded_projects", []),
            "contributions": {
                aid: accepted_proposal.get("contributions_by_agent", {}).get(aid, [])
                for aid in valuations
            },
            "aggregate_totals": accepted_proposal.get(
                "aggregate_totals", [0.0] * game_state["m_projects"]
            ),
        }

    def check_early_termination(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if pledges have converged.

        In joint mode: checks if all agents' proposed joint plans agree within
        tolerance in the current round (cross-agent agreement). For all agents
        i,j and all target agents k: joint_plan_i[k] ≈ joint_plan_j[k].

        In individual mode: checks if all agents submitted identical pledges
        (within tolerance) for 2 consecutive rounds.

        Args:
            game_state: Game state with round_pledges history and joint_plans

        Returns:
            True if early termination condition is met
        """
        pledge_mode = game_state.get("pledge_mode", "individual")

        if pledge_mode == "joint":
            # Joint mode: check if all agents' joint plans agree this round
            joint_plans = game_state.get("joint_plans", {})
            if len(joint_plans) < 2:
                return False

            agent_ids = sorted(joint_plans.keys())
            reference_plan = joint_plans[agent_ids[0]]

            for aid in agent_ids[1:]:
                other_plan = joint_plans[aid]
                # Compare all target agent entries
                all_targets = set(reference_plan.keys()) | set(other_plan.keys())
                for target_aid in all_targets:
                    ref_vec = reference_plan.get(target_aid)
                    oth_vec = other_plan.get(target_aid)
                    if ref_vec is None or oth_vec is None:
                        return False
                    if not np.allclose(ref_vec, oth_vec, atol=0.5):
                        return False

            return True
        else:
            # Individual mode: 2 consecutive identical rounds
            history = game_state["round_pledges"]
            if len(history) < 2:
                return False

            last = history[-1]
            prev = history[-2]

            if set(last.keys()) != set(prev.keys()):
                return False

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
        m = len(projects)

        own_prev = game_state.get("current_pledges", {}).get(agent_id, {}).get("contributions", [0.0] * m)
        if len(own_prev) != m:
            own_prev = [0.0] * m

        lines = ["ROUND RESULTS - Aggregate Contributions:", ""]
        for j, (proj, cost, agg) in enumerate(zip(projects, costs, aggregates)):
            cost_text = self._format_display_number(cost)
            if j in funded:
                lines.append(f"  {proj['name']}: {agg:.2f} / {cost_text} -- PROVISIONALLY FUNDED (your_prev={own_prev[j]:.2f})")
            else:
                gap = cost - agg
                pct = (agg / cost * 100) if cost > 0 else 0
                lines.append(f"  {proj['name']}: {agg:.2f} / {cost_text} ({pct:.0f}%) -- needs {gap:.2f} more (your_prev={own_prev[j]:.2f})")

        lines.append("")
        funded_names = [projects[j]["name"] for j in funded]
        lines.append(f"Provisionally funded projects: {funded_names if funded_names else 'None'}")
        lines.append("")
        lines.append("Consider adjusting your contributions based on these aggregate results.")

        return "\n".join(lines)
