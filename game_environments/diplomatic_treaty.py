"""
Diplomatic Treaty game environment implementation.

This module implements a multi-issue diplomatic negotiation game where agents
negotiate over continuous issue positions with position preferences and
importance weights.
"""

import json
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

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
    - Utility = 100 × Σ_k w_ik × (1 - |p_ik - a_k|)

    Two control parameters:
    - ρ (rho): Preference correlation [-1, 1]
    - θ (theta): Interest overlap [0, 1]
    """

    # Each issue is a continuous policy rate in [0, 1].
    # An agent's position is their PREFERRED RATE on that issue:
    #   0.0 = 0%  (minimum level of that policy)
    #   1.0 = 100% (maximum level of that policy)
    # Intermediate values are semantically meaningful — 0.35 literally means "35% of X".
    ISSUE_NAMES = [
        "AI chip export quota",
        "Critical mineral emergency stockpile contribution",
        "Nuclear warhead reduction",
        "Fentanyl precursor control breadth",
        "Carbon cost on imports",
        "High-seas fishing quota reduction",
        "Shipping emissions reduction target",
        "Orbital debris mitigation requirement",
        "Routine antibiotic-use restriction in livestock",
        "Deep-sea mining moratorium coverage",
    ]

    # Scale endpoints shown in the game-rules prompt — "0% = X | 100% = Y".
    ISSUE_PROPOSITIONS = [
        "0% = total ban on H200-class AI chip exports | 100% = unrestricted export of all advanced AI chips",
        "0% = no designated critical minerals contributed to the accord's emergency stockpile | 100% = each party contributes its full target amount of designated critical minerals to the accord's emergency stockpile",
        "0% = no warheads eliminated | 100% = complete multilateral nuclear disarmament",
        "0% = only the highest-risk direct fentanyl precursors are subject to mandatory export inspection and seizure | 100% = the full watchlist of flagged fentanyl-related precursor and pre-precursor chemicals is subject to mandatory export inspection and seizure",
        "0% = no carbon cost on covered imports | 100% = full domestic carbon price applied to covered imports",
        "0% = no reduction in catch limits for covered high-seas fisheries | 100% = complete moratorium on commercial catch for covered high-seas fisheries",
        "0% = no emissions reduction required for covered international shipping | 100% = net-zero emissions required for covered international shipping by the accord deadline",
        "0% = no mandatory post-mission disposal rule for covered satellites | 100% = all covered satellites must meet the accord's strictest post-mission disposal rule",
        "0% = routine antibiotic use allowed in all covered livestock production | 100% = routine antibiotic use prohibited in all covered livestock production except narrow emergency exemptions",
        "0% = no proposed commercial deep-sea mining zones covered by a moratorium | 100% = all proposed commercial deep-sea mining zones covered by a moratorium",
    ]

    # Plain-English interpretation of a position on each issue.
    # Used in the preference assignment prompt: "your position of X means {template.format(pct=X*100)}"
    ISSUE_INTERP_TEMPLATES = [
        "{pct}% of advanced AI chip production cleared for export",
        "{pct}% of each party's target contribution committed to the accord's emergency critical mineral stockpile",
        "{pct}% of multilateral nuclear warheads eliminated",
        "{pct}% of the accord's flagged fentanyl-related chemical watchlist subject to mandatory export inspection and seizure",
        "{pct}% of the domestic carbon price applied to covered imports",
        "{pct}% reduction in catch limits for covered high-seas fisheries",
        "{pct}% emissions reduction required for covered international shipping by the accord deadline",
        "{pct}% of covered satellites required to meet the accord's strictest post-mission disposal rule",
        "{pct}% of covered livestock production subject to a ban on routine antibiotic use",
        "{pct}% of proposed commercial deep-sea mining zones covered by the moratorium",
    ]

    def __init__(self, config: DiplomaticTreatyConfig):
        """
        Initialize Diplomatic Treaty game.

        Args:
            config: DiplomaticTreatyConfig with n_issues, rho, theta, etc.
        """
        super().__init__(config)
        self.config: DiplomaticTreatyConfig = config
        self._rng = np.random.RandomState(config.random_seed)

    @staticmethod
    def _round_to_integer_percentages(
        values: np.ndarray,
        total: Optional[int] = None,
    ) -> np.ndarray:
        """Round normalized values to integer percentage points."""
        clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        scaled = clipped * 100.0

        if total is None:
            return np.floor(scaled + 0.5).astype(int)

        floored = np.floor(scaled).astype(int)
        remainder = int(round(total - floored.sum()))

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
    def _percentages_to_unit_interval(percentages: np.ndarray) -> np.ndarray:
        """Convert integer percentage points back to normalized [0, 1] values."""
        return np.asarray(percentages, dtype=float) / 100.0

    @staticmethod
    def _pairwise_cosine_error(weight_matrix: np.ndarray, target_theta: float) -> float:
        """Compute total pairwise cosine-similarity error for integer weight vectors."""
        n_agents = weight_matrix.shape[0]
        total_error = 0.0

        for i in range(n_agents):
            n_i = np.linalg.norm(weight_matrix[i])
            if n_i < 1e-12:
                return float("inf")

            for j in range(i + 1, n_agents):
                n_j = np.linalg.norm(weight_matrix[j])
                if n_j < 1e-12:
                    return float("inf")
                cosine = np.dot(weight_matrix[i], weight_matrix[j]) / (n_i * n_j)
                total_error += (cosine - target_theta) ** 2

        return total_error

    @classmethod
    def _refine_integer_weights_for_target_cosine(
        cls,
        weight_matrix: np.ndarray,
        target_theta: float,
        max_iterations: int = 200,
    ) -> np.ndarray:
        """
        Improve integer-rounded weights with local search while preserving per-agent sum=100.

        Starting from a largest-remainder rounding, move one percentage point at a time
        within a single agent's vector if doing so reduces total pairwise cosine error.
        """
        current = np.asarray(weight_matrix, dtype=int).copy()
        current_error = cls._pairwise_cosine_error(current, target_theta)

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
                            target_theta,
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

    @staticmethod
    def _format_percentage(value: float) -> str:
        """Render either a normalized [0,1] value or raw percentage point as `NN%`."""
        numeric_value = float(value)
        if 0.0 <= numeric_value <= 100.0 and numeric_value.is_integer() and numeric_value > 1.0:
            return f"{int(numeric_value)}%"
        return f"{int(round(numeric_value * 100))}%"

    @staticmethod
    def _is_integer_percentage(value: Any) -> bool:
        """Check whether a value is an integer percentage point in [0, 100]."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False
        return 0.0 <= numeric_value <= 100.0 and numeric_value.is_integer()

    @staticmethod
    def _looks_like_percentage_vector(values: List[Any]) -> bool:
        """
        Detect whether a proposal vector is using integer percentage points.

        Accept in-range percentage vectors like [30, 50, 70], and also out-of-range
        integer percentage inputs like [-50, 150, 50] so they can be clipped as
        percentages rather than misread as legacy [0, 1] floats.
        """
        numeric_values = []
        for value in values:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                return False

        if not numeric_values:
            return False

        if all(0.0 <= numeric_value <= 1.0 for numeric_value in numeric_values):
            return False

        return all(numeric_value.is_integer() for numeric_value in numeric_values)

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
        positions = self._percentages_to_unit_interval(
            self._round_to_integer_percentages(positions)
        )

        # Generate weights with overlap theta
        weights = self._generate_weights(n_agents, n_issues)
        integer_weights = np.vstack([
            self._round_to_integer_percentages(weight_row, total=100)
            for weight_row in weights
        ])
        integer_weights = self._refine_integer_weights_for_target_cosine(
            integer_weights,
            self.config.theta,
        )
        weights = self._percentages_to_unit_interval(integer_weights)

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
            "parameters": {
                "rho": self.config.rho,
                "theta": self.config.theta,
            },
            "game_type": "diplomatic_treaty"
        }

    def _generate_positions(self, n_agents: int, n_issues: int) -> np.ndarray:
        """
        Generate position preferences using the Gaussian copula (Section 3.1).

        Samples latent z_k ~ N(0, Σ_z) for each issue k, then transforms
        p_ik = Φ(z_ik) to obtain Uniform[0,1] marginals with the desired
        rank correlation controlled by rho.

        Args:
            n_agents: Number of agents
            n_issues: Number of issues

        Returns:
            positions: Shape (n_agents, n_issues), values in [0, 1]
        """
        # Convert target correlation to latent correlation via inverse relationship
        rho_z = 2 * np.sin(np.pi * self.config.rho / 6)

        # Build equicorrelation matrix Σ_z
        cov_matrix = np.full((n_agents, n_agents), rho_z)
        np.fill_diagonal(cov_matrix, 1.0)

        # PSD validation (should already be caught by config, but defensive)
        if rho_z < -1.0 / (n_agents - 1):
            raise ValueError(
                f"Latent correlation rho_z={rho_z:.4f} (from rho={self.config.rho}) "
                f"violates PSD constraint for N={n_agents} agents. "
                f"Need rho_z >= {-1.0/(n_agents-1):.4f}."
            )

        positions = np.zeros((n_agents, n_issues))
        mean = np.zeros(n_agents)

        for k in range(n_issues):
            # Sample from multivariate normal in latent space
            z_k = self._rng.multivariate_normal(mean, cov_matrix)
            # Transform through Φ (standard normal CDF) -> Uniform[0,1]
            positions[:, k] = norm.cdf(z_k)

        return positions

    def _generate_weights(self, n_agents: int, n_issues: int) -> np.ndarray:
        """
        Generate importance weights with exact cosine similarity theta (Section 3.2).

        Uses SLSQP joint optimization of all weight vectors on the simplex,
        following the same pattern as RandomVectorGenerator._optimize_vectors()
        but with sum=1 instead of sum=100.

        Args:
            n_agents: Number of agents
            n_issues: Number of issues

        Returns:
            weights: Shape (n_agents, n_issues), each row sums to 1, all >= 0
        """
        alpha = 2.0  # Dirichlet concentration for random initialization
        theta = self.config.theta

        if theta == 1.0:
            # Special case: identical weights
            w0 = self._rng.dirichlet(np.full(n_issues, alpha))
            return np.tile(w0, (n_agents, 1))

        K = n_issues

        # For 2 agents: jointly optimize both vectors for exact cosine
        if n_agents == 2:
            best_x = None
            best_error = float('inf')
            max_attempts = 15

            def objective_2(x):
                w1 = x[:K]
                w2 = x[K:]
                n1 = np.linalg.norm(w1)
                n2 = np.linalg.norm(w2)
                if n1 < 1e-12 or n2 < 1e-12:
                    return 1e10
                cos_sim = np.dot(w1, w2) / (n1 * n2)
                return (cos_sim - theta) ** 2

            constraints_2 = [
                {'type': 'eq', 'fun': lambda x: np.sum(x[:K]) - 1.0},
                {'type': 'eq', 'fun': lambda x: np.sum(x[K:]) - 1.0},
            ]
            bounds_2 = [(0.0, 1.0)] * (2 * K)

            for attempt in range(max_attempts):
                conc = self._rng.uniform(0.5, 3.0)
                x0 = np.concatenate([
                    self._rng.dirichlet(np.full(K, conc)),
                    self._rng.dirichlet(np.full(K, conc)),
                ])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective_2, x0, method='SLSQP',
                        bounds=bounds_2, constraints=constraints_2,
                        options={'maxiter': 2000, 'ftol': 1e-14}
                    )

                error = objective_2(result.x)
                if error < best_error:
                    best_error = error
                    best_x = result.x.copy()
                if best_error < 1e-10:
                    break

            if best_x is None or best_error > 0.0001:
                raise RuntimeError(
                    f"Failed to generate weights with cosine similarity "
                    f"theta={theta} after {max_attempts} attempts. "
                    f"Best error: {best_error:.6f}"
                )

            weights = np.zeros((2, K))
            weights[0] = np.maximum(best_x[:K], 0.0)
            weights[0] /= weights[0].sum()
            weights[1] = np.maximum(best_x[K:], 0.0)
            weights[1] /= weights[1].sum()
            return weights

        # For N>2 agents: joint optimization over all N vectors simultaneously
        # Minimizes sum of squared errors across all N(N-1)/2 pairs
        K = n_issues
        dim = n_agents * K

        def objective_n(x):
            vecs = x.reshape(n_agents, K)
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
                    total_err += (cos_sim - theta) ** 2
            return total_err

        constraints_n = [
            {'type': 'eq', 'fun': lambda x, i=i: np.sum(x[i * K:(i + 1) * K]) - 1.0}
            for i in range(n_agents)
        ]
        bounds_n = [(0.0, 1.0)] * dim

        best_x = None
        best_error = float('inf')
        max_attempts = 30

        for _ in range(max_attempts):
            conc = self._rng.uniform(0.5, 3.0)
            x0 = np.concatenate([
                self._rng.dirichlet(np.full(K, conc))
                for _ in range(n_agents)
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective_n, x0, method='SLSQP',
                    bounds=bounds_n, constraints=constraints_n,
                    options={'maxiter': 3000, 'ftol': 1e-14}
                )

            error = objective_n(result.x)
            if error < best_error:
                best_error = error
                best_x = result.x.copy()
            if best_error < 1e-10:
                break

        if best_x is None or best_error > 0.001:
            raise RuntimeError(
                f"Failed to generate weights with all-pairs cosine similarity "
                f"theta={theta} for {n_agents} agents after "
                f"{max_attempts} attempts. Best error: {best_error:.6f}"
            )

        weights = best_x.reshape(n_agents, K)
        for i in range(n_agents):
            weights[i] = np.maximum(weights[i], 0.0)
            weights[i] = weights[i] / weights[i].sum()

        return weights

    def _get_parties_phrase(self) -> str:
        """Create clearer phrasing for 2-agent negotiations."""
        if self.config.n_agents == 2:
            return "another party"
        return f"{self.config.n_agents - 1} other parties"

    def _get_rules_block(self, game_state: Dict[str, Any]) -> str:
        """Build the shared rules/setup block for diplomatic treaty negotiation."""
        issues = game_state["issues"]
        round_2_pct = round(self.config.gamma_discount * 100)
        round_3_pct = round((self.config.gamma_discount ** 2) * 100)

        # Build issues list with proposition text
        issue_lines = []
        for i, issue in enumerate(issues):
            proposition = (
                self.ISSUE_PROPOSITIONS[i]
                if i < len(self.ISSUE_PROPOSITIONS)
                else "(see preference assignment)"
            )
            issue_lines.append(f"  {i+1}. **{issue}**\n     Scale: {proposition}")
        issues_text = "\n".join(issue_lines)

        parties_phrase = self._get_parties_phrase()

        return f"""Welcome to the Diplomatic Treaty Negotiation!

You are participating in a diplomatic negotiation with {parties_phrase} over {len(issues)} policy issues. Here is your full setup information:

**ISSUES UNDER NEGOTIATION:**
Each issue is a continuous policy rate expressed as an integer percentage from 0% to 100%, where:
  - **0%** = the minimum level of that policy (see scale below)
  - **100%** = the maximum level of that policy (see scale below)
  - **50%** = the exact midpoint between minimum and maximum

**Your position IS your preferred rate.** A position of 35 means you want 35% of that policy measure. Intermediate values are meaningful — there is no "neutral"; every number reflects a specific policy level.

{issues_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} parties negotiating (including you)
- The negotiation will last up to {self.config.t_rounds} rounds
- This message is the one-time setup phase
- After setup, each round follows: Discussion -> Private Thinking -> Proposal -> Voting -> Reflection
- An agreement vector resolves every issue simultaneously

**PRIVATE INFORMATION:**
- You have a SECRET IDEAL POSITION on each issue (your preferred percentage)
- You have SECRET IMPORTANCE WEIGHTS on each issue that sum to 100%
- These positions and weights are PRIVATE and specific to you

**AGREEMENT FORMAT:**
- An agreement is a vector of {len(issues)} integer percentages, one per issue
- Example: [30, 70, 50, ...]
- Each value is the agreed rate on that issue's 0% to 100% scale

**UTILITY CALCULATION:**
- Your utility = weighted sum of how close each resolved score is to your ideal
- Formula: Σ (weight_k × (1 - |your_position_k - agreement_k| / 100))
- Maximum utility = 100 (every issue resolved at your exact ideal score)

**VOTING RULES:**
- You vote "accept" or "reject" on each proposed agreement
- A proposal needs UNANIMOUS acceptance from all parties to take effect
- If no agreement is reached by the final round, then all parties walk away with zero utility.

**REWARD DISCOUNTING:**
- Each additional round multiplies utility by {round_2_pct}%
- Round 1 rewards: 100% of utility
- Round 2 rewards: {round_2_pct}% of utility
- Round 3 rewards: {round_3_pct}% of utility
- The longer negotiations take, the less valuable the final agreement becomes

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility (after discounting)
- Utility depends on both closeness to your ideal positions and the importance weights on each issue
- No deal means everyone gets zero utility
- Consider both the substantive agreement and the likelihood it will be accepted
- Earlier agreements are worth more due to discounting"""

    def _get_private_preferences_block(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Build the per-agent private preference block."""
        issues = game_state["issues"]
        positions = game_state["agent_positions"][agent_id]
        weights = game_state["agent_weights"][agent_id]

        lines = ["LOCKED PRIVATE PREFERENCES", ""]
        lines.append(
            f"{agent_id}, you have been assigned the following SECRET treaty preferences:"
        )
        lines.append("")
        lines.append("**YOUR PRIVATE IDEAL POSITIONS:**")
        lines.append("  Each issue is a continuous policy rate: 0% = minimum, 100% = maximum.")
        lines.append("  Your position is the rate you ideally want.")
        lines.append("")

        for i, (issue, pos) in enumerate(zip(issues, positions)):
            pct = round(pos * 100)
            if i < len(self.ISSUE_INTERP_TEMPLATES):
                interp = self.ISSUE_INTERP_TEMPLATES[i].format(pct=pct)
            else:
                interp = f"{pct}% level"
            lines.append(f"  {issue}: {pct}% -> {interp}")

        lines.append("")
        lines.append("**YOUR PRIVATE IMPORTANCE WEIGHTS:**")
        lines.append(
            "  These weights sum to 100% and determine how much each issue contributes to your utility."
        )
        for issue, weight in zip(issues, weights):
            lines.append(f"  {issue}: {self._format_percentage(weight)}")

        lines.append("")
        lines.append("**STRATEGIC ANALYSIS:**")
        lines.append(
            "- Your maximum possible utility is 100 points if every issue is resolved exactly at your ideal position"
        )
        lines.append(
            "- Focus more on issues with higher weights, since they generally matter more for your utility"
        )
        lines.append("")
        lines.append("**STRATEGIC CONSIDERATIONS:**")
        lines.append("1. Other parties don't know your exact ideal positions or weights")
        lines.append(
            "2. You may choose to reveal some preferences precisely, vaguely, or not at all"
        )
        lines.append(
            "3. Consider where lower-weight issues could be traded for gains on higher-weight issues"
        )
        lines.append("4. Remember: you need ALL parties to accept a proposal")

        return "\n".join(lines)

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate diplomatic negotiation rules explanation."""
        rules_block = self._get_rules_block(game_state)
        return f"""{rules_block}

Please acknowledge that you understand these rules and are ready to negotiate!"""

    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """Generate diplomatic preference assignment prompt."""
        preferences_block = self._get_private_preferences_block(agent_id, game_state)
        return f"""{preferences_block}

Please acknowledge that you understand your diplomatic preferences."""

    def uses_combined_setup_phase(self) -> bool:
        """Diplomatic Treaty merges private preference assignment into setup."""
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
In your response, just acknowledge the setup, summarize the game structure and rules, and reiterate the private ideal positions and importance weights that were assigned to you."""

    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate proposal prompt for diplomatic negotiation."""
        issues = game_state["issues"]
        n_issues = len(issues)

        issues_list = "\n".join([f"  {i}: {issue}" for i, issue in enumerate(issues)])

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""Please propose a treaty agreement.

**Current Context:**
- Issues being negotiated:
{issues_list}
- Round: {round_num}/{self.config.t_rounds}{reasoning_instruction}

**Instructions:**
Propose a resolution for each issue as an integer percentage between 0 and 100.

Respond with ONLY a JSON object in this exact format:
{{
    "agreement": [30, 70, 50, 20, 80],
    "reasoning": "Brief explanation of your proposed compromise"
}}

**Rules:**
- The "agreement" array must have exactly {n_issues} values (one per issue)
- Each value must be an integer between 0 and 100
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
            agreement = agreement[:n_issues]
            is_percentage_vector = self._looks_like_percentage_vector(agreement)
            if len(agreement) < n_issues:
                pad_value = 50 if is_percentage_vector else 0.5
                agreement.extend([pad_value] * (n_issues - len(agreement)))

            # Prefer integer-percentage vectors, but retain legacy [0, 1] support.
            if is_percentage_vector:
                agreement = [
                    max(0.0, min(1.0, float(v) / 100.0))
                    for v in agreement
                ]
            else:
                agreement = [
                    max(0.0, min(1.0, float(v)))
                    for v in agreement
                ]

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

        numeric_values = []
        for val in agreement:
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                return False

        if all(0.0 <= v <= 1.0 for v in numeric_values):
            return True

        return all(0.0 <= v <= 100.0 and v.is_integer() for v in numeric_values)

    def calculate_utility(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> float:
        """
        Calculate diplomatic utility.

        Utility = 100 × Σ_k w_k × (1 - |p_k - a_k|)

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

        # Weighted sum in [0, 1]; scale to [0, 100]
        raw_utility = float(np.sum(weights * values)) * 100.0

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
        """Generate diplomatic discussion prompt."""
        issues = game_state["issues"]
        issues_text = ", ".join(issues[:5])
        if len(issues) > 5:
            issues_text += f", and {len(issues) - 5} more"

        # Inject prior turns so each speaker sees what has been said this round
        history_section = ""
        if discussion_history:
            history_section = "\n**DISCUSSION SO FAR THIS ROUND:**\n"
            for msg in discussion_history:
                history_section += f"{msg}\n\n"
            history_section += "---\n"

        if round_num == 1 and not discussion_history:
            context = """**DISCUSSION OBJECTIVES:**
- Signal your priorities and general stance on the issues
- Understand the other party's concerns and interests
- Identify potential areas for agreement and trade-offs
- Explore package deals across multiple issues

Each issue is a continuous rate (0%–100%), so you may communicate as precisely or as broadly as your strategy dictates — naming specific target rates, ranges, or simply signaling direction. How much you reveal is up to you.

You are the first to speak. Share your diplomatic position and opening thoughts."""
        elif discussion_history:
            context = """**YOUR TURN TO RESPOND:**
Based on what others have said above, please:
- Respond to points raised and share your own position as you see fit
- Propose trade-offs or areas of potential agreement
- Move the conversation toward a concrete proposal

How precisely you communicate your preferred rates is a strategic choice."""
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

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""🗣️ DIPLOMATIC DISCUSSION - Round {round_num}/{max_rounds}

Issues under negotiation: {issues_text}
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
        """Generate diplomatic voting prompt."""
        issues = game_state["issues"]
        agreement = proposal.get("agreement", [])
        round_2_pct = round(self.config.gamma_discount * 100)

        # Format agreement for display
        agreement_lines = []
        for i, (issue, val) in enumerate(zip(issues, agreement)):
            pos_desc = self._describe_position(val)
            agreement_lines.append(f"  {issue}: {pos_desc}")

        agreement_display = "\n".join(agreement_lines)

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"\n\n**REASONING DEPTH:** Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."

        return f"""A treaty proposal has been submitted:

**PROPOSED AGREEMENT:**
{agreement_display}

**REASONING:** {proposal.get('reasoning', 'No reasoning provided')}
**PROPOSED BY:** {proposal.get('proposed_by', 'Unknown')}

**REMINDER — HOW YOUR UTILITY IS CALCULATED:**
- Your utility = weighted sum of how close each resolved score is to your ideal position
- Formula: Σ (weight_k × (1 - |your_position_k - agreement_k| / 100))
- A rate of 0 means 0% (minimum policy level); 100 means 100% (maximum policy level) on each issue
- Maximum utility = 100 (every issue resolved at your exact ideal rate)
- Each additional round multiplies utility by {round_2_pct}% — delaying costs you

Please vote on this proposal. Consider:
- How close is each resolved score to your ideal position on each issue?
- Could you realistically negotiate a better agreement before the final round?
- The cost of delay: each additional round reduces your eventual payoff{reasoning_instruction}

Respond with ONLY a JSON object:
{{
    "vote": "accept",
    "reasoning": "Explanation of your vote, referencing specific propositions and how they compare to your ideal positions"
}}

Vote must be either "accept" or "reject"."""

    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[str],
        reasoning_token_budget: Optional[int] = None
    ) -> str:
        """Generate diplomatic thinking prompt."""
        issues = game_state["issues"]
        positions = game_state["agent_positions"][agent_id]
        weights = game_state["agent_weights"][agent_id]

        # Find top priorities
        priority_indices = np.argsort(weights)[::-1][:3]
        top_priorities = [
            (
                f"{issues[i]} (weight: {self._format_percentage(weights[i])}, "
                f"ideal: {self._format_percentage(positions[i])})"
            )
            for i in priority_indices
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
            urgency = "\n⚠️ **CRITICAL**: Final rounds - agreement urgency is high!"

        reasoning_instruction = ""
        if reasoning_token_budget:
            reasoning_instruction = f"""

**REASONING DEPTH:**
Please use approximately {reasoning_token_budget} tokens in your internal reasoning before outputting your response for this stage."""

        return f"""🧠 PRIVATE STRATEGIC ANALYSIS - Round {round_num}/{max_rounds}
{urgency}
{discussion_section}
**YOUR TOP PRIORITIES:**
{chr(10).join(['- ' + p for p in top_priorities])}

**STRATEGIC ANALYSIS TASKS:**
1. What have you learned about other parties' priorities from the discussion above?
2. Where might they be willing to compromise?
3. What agreement would maximize your utility while being acceptable to all?
4. Which issues could you concede on to gain elsewhere?{reasoning_instruction}

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
            lines.append(f"  {issue}: {pos_desc}")

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
        """Convert numeric position to an integer percentage string."""
        return DiplomaticTreatyGame._format_percentage(value)
