import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class DiplomaticTreatyGame:
    """
    Multi-agent diplomatic treaty negotiation environment.
    
    Parameters control competition level:
    - rho (ρ): Preference correlation ∈ [-1, 1]
    - theta (θ): Interest overlap ∈ [0, 1]
    - lambda (λ): Issue compatibility ∈ [-1, 1]
    """
    
    def __init__(self, n_agents: int = 2, n_issues: int = 5, 
                 rho: float = 0.0, theta: float = 0.5, lam: float = 0.0,
                 seed: int = None):
        """
        Initialize negotiation environment.
        
        Args:
            n_agents: Number of negotiating agents
            n_issues: Number of issues to negotiate
            rho: Preference correlation (-1 to 1)
            theta: Interest overlap (0 to 1)
            lam: Issue compatibility (-1 to 1)
            seed: Random seed for reproducibility
        """
        self.n_agents = n_agents
        self.n_issues = n_issues
        self.rho = rho
        self.theta = theta
        self.lam = lam
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate agent preferences
        self.positions = self._generate_positions()
        self.weights = self._generate_weights()
        self.issue_types = self._generate_issue_types()
        
    def _generate_positions(self) -> np.ndarray:
        """
        Generate position preferences with correlation rho.
        
        Returns:
            positions: Shape (n_agents, n_issues), values in [0, 1]
        """
        # Construct covariance matrix
        sigma = 0.25  # Standard deviation
        cov_matrix = np.full((self.n_agents, self.n_agents), self.rho * sigma**2)
        np.fill_diagonal(cov_matrix, sigma**2)
        
        # Generate correlated preferences for each issue
        positions = np.zeros((self.n_agents, self.n_issues))
        mean = np.full(self.n_agents, 0.5)
        
        for k in range(self.n_issues):
            # Sample from multivariate normal
            pos_k = np.random.multivariate_normal(mean, cov_matrix)
            # Clip to [0, 1]
            positions[:, k] = np.clip(pos_k, 0, 1)
        
        return positions
    
    def _generate_weights(self) -> np.ndarray:
        """
        Generate importance weights with overlap theta.
        
        Returns:
            weights: Shape (n_agents, n_issues), normalized to sum to 1
        """
        alpha = 2.0  # Dirichlet concentration parameter
        
        # Generate first agent's weights
        weights = np.zeros((self.n_agents, self.n_issues))
        weights[0] = np.random.dirichlet(np.full(self.n_issues, alpha))
        
        # Generate subsequent agents' weights with target overlap
        for i in range(1, self.n_agents):
            # Generate candidate weights
            w_candidate = np.random.dirichlet(np.full(self.n_issues, alpha))
            
            # Mix with first agent to achieve target overlap
            w_mixed = self.theta * weights[0] + (1 - self.theta) * w_candidate
            
            # Renormalize
            weights[i] = w_mixed / w_mixed.sum()
        
        return weights
    
    def _generate_issue_types(self) -> np.ndarray:
        """
        Generate issue types (compatible or conflicting) based on lambda.
        
        Returns:
            issue_types: Shape (n_issues,), values in {-1, 1}
        """
        # Probability of compatible issue
        p_compatible = (self.lam + 1) / 2
        
        # Sample issue types
        issue_types = np.random.choice(
            [1, -1], 
            size=self.n_issues, 
            p=[p_compatible, 1 - p_compatible]
        )
        
        # For conflicting issues, set opposing preferences
        for k in range(self.n_issues):
            if issue_types[k] == -1 and self.n_agents == 2:
                # Make preferences opposing for zero-sum issues
                self.positions[1, k] = 1 - self.positions[0, k]
        
        return issue_types
    
    def compute_utility(self, agent_id: int, agreement: np.ndarray) -> float:
        """
        Compute agent's utility from an agreement.
        
        Args:
            agent_id: Which agent (0 to n_agents-1)
            agreement: Array of shape (n_issues,) with values in [0, 1]
        
        Returns:
            utility: Weighted sum of issue values
        """
        positions = self.positions[agent_id]
        weights = self.weights[agent_id]
        
        # Value function: v_ik(a_k) = 1 - |p_ik - a_k|
        values = 1 - np.abs(positions - agreement)
        
        # Utility: sum of weighted values
        utility = np.sum(weights * values)
        
        return utility
    
    def compute_social_welfare(self, agreement: np.ndarray) -> float:
        """Compute total utility across all agents."""
        return sum(self.compute_utility(i, agreement) 
                   for i in range(self.n_agents))
    
    def find_optimal_agreement(self) -> Tuple[np.ndarray, float]:
        """
        Find agreement that maximizes social welfare (Pareto optimal).
        Uses grid search for simplicity.
        
        Returns:
            best_agreement: Optimal agreement vector
            best_welfare: Maximum social welfare
        """
        # Grid search over agreement space
        grid_size = 20
        best_welfare = -np.inf
        best_agreement = None
        
        # Generate all possible agreements on grid
        if self.n_issues <= 3:
            # Exhaustive search for small problems
            from itertools import product
            grid_points = np.linspace(0, 1, grid_size)
            
            for agreement_tuple in product(grid_points, repeat=self.n_issues):
                agreement = np.array(agreement_tuple)
                welfare = self.compute_social_welfare(agreement)
                
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_agreement = agreement
        else:
            # Random search for larger problems
            for _ in range(10000):
                agreement = np.random.uniform(0, 1, self.n_issues)
                welfare = self.compute_social_welfare(agreement)
                
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_agreement = agreement
        
        return best_agreement, best_welfare
    
    def visualize_preferences(self):
        """Visualize agent preferences and weights."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot positions
        x = np.arange(self.n_issues)
        width = 0.35
        
        for i in range(self.n_agents):
            ax1.bar(x + i*width, self.positions[i], width, 
                   label=f'Agent {i+1}', alpha=0.8)
        
        ax1.set_xlabel('Issue')
        ax1.set_ylabel('Position Preference')
        ax1.set_title(f'Position Preferences (ρ={self.rho:.2f})')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels([f'Issue {k+1}' for k in range(self.n_issues)])
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Plot weights
        for i in range(self.n_agents):
            ax2.bar(x + i*width, self.weights[i], width, 
                   label=f'Agent {i+1}', alpha=0.8)
        
        ax2.set_xlabel('Issue')
        ax2.set_ylabel('Importance Weight')
        ax2.set_title(f'Importance Weights (θ={self.theta:.2f})')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels([f'Issue {k+1}' for k in range(self.n_issues)])
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """Print summary of the negotiation scenario."""
        print(f"=== Diplomatic Treaty Negotiation ===")
        print(f"Agents: {self.n_agents}, Issues: {self.n_issues}")
        print(f"\nParameters:")
        print(f"  ρ (preference correlation): {self.rho:.2f}")
        print(f"  θ (interest overlap): {self.theta:.2f}")
        print(f"  λ (issue compatibility): {self.lam:.2f}")
        
        # Compute actual correlation and overlap
        if self.n_agents == 2:
            actual_corr = np.corrcoef(
                self.positions[0], self.positions[1]
            )[0, 1]
            actual_overlap = np.dot(self.weights[0], self.weights[1])
            print(f"\nActual values:")
            print(f"  Position correlation: {actual_corr:.2f}")
            print(f"  Weight overlap: {actual_overlap:.2f}")
        
        print(f"\nIssue types:")
        for k in range(self.n_issues):
            issue_type = "Compatible" if self.issue_types[k] == 1 else "Conflicting"
            print(f"  Issue {k+1}: {issue_type}")
        
        print(f"\nPreferences:")
        for i in range(self.n_agents):
            print(f"  Agent {i+1}:")
            print(f"    Positions: {self.positions[i]}")
            print(f"    Weights: {self.weights[i]}")


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Pure Cooperation")
    print("=" * 60)
    game1 = DiplomaticTreatyGame(
        n_agents=2, n_issues=5,
        rho=1.0,      # Aligned preferences
        theta=0.0,    # Different priorities
        lam=1.0,      # All win-win issues
        seed=42
    )
    game1.print_summary()
    
    # Find optimal agreement
    optimal_agreement, optimal_welfare = game1.find_optimal_agreement()
    print(f"\nOptimal agreement: {optimal_agreement}")
    print(f"Social welfare: {optimal_welfare:.3f}")
    
    for i in range(game1.n_agents):
        utility = game1.compute_utility(i, optimal_agreement)
        print(f"  Agent {i+1} utility: {utility:.3f}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Pure Competition")
    print("=" * 60)
    game2 = DiplomaticTreatyGame(
        n_agents=2, n_issues=5,
        rho=-1.0,     # Opposing preferences
        theta=1.0,    # Same priorities
        lam=-1.0,     # All zero-sum issues
        seed=42
    )
    game2.print_summary()
    
    optimal_agreement, optimal_welfare = game2.find_optimal_agreement()
    print(f"\nOptimal agreement: {optimal_agreement}")
    print(f"Social welfare: {optimal_welfare:.3f}")
    
    for i in range(game2.n_agents):
        utility = game2.compute_utility(i, optimal_agreement)
        print(f"  Agent {i+1} utility: {utility:.3f}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Integrative Bargaining")
    print("=" * 60)
    game3 = DiplomaticTreatyGame(
        n_agents=2, n_issues=5,
        rho=0.0,      # Uncorrelated preferences
        theta=0.3,    # Different priorities (logrolling potential)
        lam=0.5,      # Mostly compatible issues
        seed=42
    )
    game3.print_summary()
    
    optimal_agreement, optimal_welfare = game3.find_optimal_agreement()
    print(f"\nOptimal agreement: {optimal_agreement}")
    print(f"Social welfare: {optimal_welfare:.3f}")
    
    for i in range(game3.n_agents):
        utility = game3.compute_utility(i, optimal_agreement)
        print(f"  Agent {i+1} utility: {utility:.3f}")
    
    # Visualize the integrative bargaining scenario
    print("\nGenerating visualization...")
    fig = game3.visualize_preferences()
    plt.savefig('diplomatic_treaty_example.png', dpi=150, bbox_inches='tight')
    print("Saved to: diplomatic_treaty_example.png")
