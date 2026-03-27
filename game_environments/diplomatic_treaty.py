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
        "Autonomous weapons human oversight",
        "Critical mineral revenue share",
        "Disputed territory restoration",
        "Nuclear warhead reduction",
        "AI training data localization",
        "Fentanyl precursor interdiction",
        "Carbon border adjustment",
        "Domestic content requirement",
        "Bilateral sanctions relief",
    ]

    # Scale endpoints shown in the game-rules prompt — "0% = X | 100% = Y".
    ISSUE_PROPOSITIONS = [
        "0% = total ban on H200-class AI chip exports | 100% = unrestricted export of all advanced AI chips",
        "0% = fully autonomous lethal decisions (no human required) | 100% = every strike requires explicit human authorization",
        "0% = host nation keeps all extraction revenues | 100% = partner nation receives all extraction revenues",
        "0% = no territory returned (status quo frozen) | 100% = full pre-conflict borders restored",
        "0% = no warheads eliminated | 100% = complete bilateral nuclear disarmament",
        "0% = citizen AI training data freely processed abroad | 100% = all citizen AI data must be stored domestically",
        "0% = no precursor shipments interdicted | 100% = all suspected precursor exports seized at border",
        "0% = no carbon cost on imports | 100% = full domestic carbon price applied to all partner imports",
        "0% = no domestic sourcing required | 100% = all goods must be locally produced for preferential tariff rates",
        "0% = no sanctions lifted | 100% = all existing bilateral sanctions removed",
    ]

    # Plain-English interpretation of a position on each issue.
    # Used in the preference assignment prompt: "your position of X means {template.format(pct=X*100)}"
    ISSUE_INTERP_TEMPLATES = [
        "~{pct}% of advanced AI chip production cleared for export",
        "~{pct}% of lethal autonomous strikes require explicit human authorization",
        "~{pct}% of extraction revenues paid to partner nation",
        "~{pct}% of disputed territory returned to pre-conflict control",
        "~{pct}% of bilateral nuclear warheads eliminated",
        "~{pct}% of citizen AI training data must be stored domestically",
        "~{pct}% of suspected precursor shipments interdicted at the border",
        "~{pct}% of domestic carbon price applied to partner imports",
        "~{pct}% domestic content required for preferential tariff rates",
        "~{pct}% of existing bilateral sanctions lifted",
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

    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate diplomatic negotiation rules explanation."""
        issues = game_state["issues"]

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

        # Create clearer phrasing for 2-agent negotiations
        if self.config.n_agents == 2:
            parties_phrase = "another party"
        else:
            parties_phrase = f"{self.config.n_agents - 1} other parties"

        return f"""Welcome to the Diplomatic Treaty Negotiation!

You are participating in a diplomatic negotiation with {parties_phrase} over {len(issues)} policy issues.

**ISSUES UNDER NEGOTIATION:**
Each issue is a continuous policy rate. Positions and agreed resolutions are scores in [0.0, 1.0], where:
  - **0.0** = 0% — the minimum level of that policy (see scale below)
  - **1.0** = 100% — the maximum level of that policy (see scale below)
  - **0.5** = 50% — the exact midpoint between minimum and maximum

**Your position IS your preferred rate.** A position of 0.35 literally means you want ~35% of that policy measure. Intermediate values are meaningful — there is no "neutral"; every number reflects a specific policy level.

{issues_text}

**GAME STRUCTURE:**
- There are {self.config.n_agents} parties negotiating (including you)
- The negotiation will last up to {self.config.t_rounds} rounds
- An agreement vector resolves every proposition simultaneously

**YOUR PREFERENCES:**
- You have a SECRET IDEAL POSITION on each proposition (your preferred score)
- You have IMPORTANCE WEIGHTS (how much you care about each proposition)
- Your preferences are PRIVATE — the other party does not know them

**AGREEMENT FORMAT:**
- An agreement is a vector of {len(issues)} values, one per proposition
- Example: [0.3, 0.7, 0.5, ...]
- Each value is the agreed score on that proposition's [0, 1] support scale

**UTILITY CALCULATION:**
- Your utility = weighted sum of how close each resolved score is to your ideal
- Formula: 100 × Σ (weight_k × (1 - |your_position_k - agreement_k|))
- Maximum utility = 100.0 (every proposition resolved at your exact ideal score)

**VOTING RULES:**
- You vote "accept" or "reject" on each proposed agreement
- A proposal needs UNANIMOUS acceptance from all parties to take effect
- Utility is discounted by {self.config.gamma_discount} per round — early agreement is better

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

        lines = ["🔒 CONFIDENTIAL: Your Diplomatic Preferences", ""]
        lines.append(f"{agent_id}, you have been assigned the following SECRET preferences:")
        lines.append("")
        lines.append("**YOUR IDEAL POSITIONS** (your preferred rate on each issue):")
        lines.append("  Each issue is a continuous policy rate: 0.0 = 0%, 1.0 = 100%.")
        lines.append("  Your position is the rate you ideally want.")
        lines.append("")

        for i, (issue, pos) in enumerate(zip(issues, positions)):
            pct = round(pos * 100)
            if i < len(self.ISSUE_INTERP_TEMPLATES):
                interp = self.ISSUE_INTERP_TEMPLATES[i].format(pct=pct)
            else:
                interp = f"~{pct}% level"
            lines.append(f"  {issue}: {pos:.3f} → {interp}")

        lines.append("")
        lines.append("**YOUR IMPORTANCE WEIGHTS** (how much you care about each issue):")

        for issue, weight in zip(issues, weights):
            priority = "HIGH" if weight > 0.25 else "Medium" if weight > 0.15 else "Low"
            lines.append(f"  {issue}: {weight:.3f} ({priority} priority)")

        lines.append("")
        lines.append("**STRATEGIC INSIGHT:**")
        lines.append("- Focus on issues with HIGH weights - they matter most for your utility")
        lines.append("- Consider trading concessions on low-weight issues for gains on high-weight ones")
        lines.append("")
        lines.append("Please acknowledge that you understand your diplomatic preferences.")

        return "\n".join(lines)

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

        # Format agreement for display
        agreement_lines = []
        for i, (issue, val) in enumerate(zip(issues, agreement)):
            pos_desc = self._describe_position(val)
            agreement_lines.append(f"  {issue}: {val:.3f} ({pos_desc})")

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
- Formula: 100 × Σ (weight_k × (1 - |your_position_k - agreement_k|))
- A score of 0.0 means fully opposed; 1.0 means fully supportive on each proposition
- Maximum utility = 100.0 (every proposition at your exact ideal score)
- Utility is discounted by a factor each round — delaying costs you

Please vote on this proposal. Consider:
- How close is each resolved score to your ideal position on each proposition?
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
            f"{issues[i]} (weight: {weights[i]:.2f}, ideal: {positions[i]:.2f})"
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
        """Convert numeric position to a percentage string (issues are continuous rates)."""
        return f"~{round(value * 100)}%"
