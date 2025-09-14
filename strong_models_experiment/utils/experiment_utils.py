"""Utility functions for experiment management."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from negotiation import create_competitive_preferences


class ExperimentUtils:
    """Utility functions for experiment management."""
    
    @staticmethod
    def create_items(num_items: int) -> List[Dict[str, str]]:
        """Create items for negotiation."""
        item_names = ["Apple", "Jewel", "Stone", "Quill", "Pencil", "Book", "Hat", "Camera"]
        return [{"name": item_names[i] if i < len(item_names) else f"Item_{i}"} 
                for i in range(num_items)]
    
    @staticmethod
    def create_preferences(agents: List[Any], items: List[Dict], config: Dict) -> Dict:
        """Create preferences for agents."""
        preference_manager = create_competitive_preferences(
            n_agents=len(agents),
            m_items=len(items),
            cosine_similarity=config["competition_level"]
        )
        
        preferences_data = preference_manager.generate_preferences()
        
        # Map to agent IDs
        agent_preferences = {}
        for i, agent in enumerate(agents):
            if "agent_preferences" in preferences_data:
                agent_preferences[agent.agent_id] = preferences_data["agent_preferences"][f"agent_{i}"]
            else:
                # Fallback to basic preference generation
                agent_preferences[agent.agent_id] = [random.uniform(0, 10) for _ in range(len(items))]
        
        return {
            "agent_preferences": agent_preferences,
            "cosine_similarities": preferences_data.get("cosine_similarities", {})
        }
    
    @staticmethod
    def create_enhanced_config(config: Dict, agents: List[Any], preferences: Dict, 
                              items: List[Dict], start_time: float, experiment_id: str) -> Dict:
        """Create enhanced configuration with all experiment details."""
        return {
            **config,
            "experiment_id": experiment_id,
            "start_time": start_time,
            "agents": [agent.agent_id for agent in agents],
            "items": items,
            "preferences_summary": {
                "type": "competitive",
                "cosine_similarities": preferences.get("cosine_similarities", {})
            }
        }
    
    @staticmethod
    def generate_experiment_id(batch_id: Optional[str] = None, run_number: Optional[int] = None) -> str:
        """Generate a unique experiment ID."""
        if batch_id and run_number:
            return f"{batch_id}_run_{run_number}"
        else:
            import os
            return f"strong_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


class FileManager:
    """Manages file operations for experiment results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_interaction(self, interaction: Dict, exp_dir: Path, batch_mode: bool = False, 
                        run_number: Optional[int] = None) -> None:
        """Save a single interaction to file."""
        if batch_mode and run_number:
            # In batch mode with run number
            agent_interactions_dir = exp_dir / "agent_interactions"
            agent_interactions_dir.mkdir(parents=True, exist_ok=True)
            
            agent_id = interaction.get("agent_id", "unknown")
            agent_file = agent_interactions_dir / f"run_{run_number}_agent_{agent_id}_interactions.json"
        else:
            # Single experiment mode
            agent_id = interaction.get("agent_id", "unknown")
            agent_file = exp_dir / f"agent_{agent_id}_interactions.json"
        
        # Load existing interactions or create new list
        if agent_file.exists():
            with open(agent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                interactions = data.get("interactions", [])
        else:
            interactions = []
        
        interactions.append(interaction)
        
        # Save updated interactions
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump({
                "agent_id": agent_id,
                "total_interactions": len(interactions),
                "interactions": interactions
            }, f, indent=2, default=str, ensure_ascii=False)
    
    def save_all_interactions(self, interactions: List[Dict], exp_dir: Path, 
                            batch_mode: bool = False, run_number: Optional[int] = None) -> None:
        """Save all interactions to a single file."""
        if batch_mode and run_number:
            all_interactions_file = exp_dir / f"run_{run_number}_all_interactions.json"
        else:
            all_interactions_file = exp_dir / "all_interactions.json"
        
        with open(all_interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, indent=2, default=str, ensure_ascii=False)
    
    def save_experiment_result(self, result: Dict, exp_dir: Path, batch_mode: bool = False,
                              run_number: Optional[int] = None) -> None:
        """Save experiment results to file."""
        if batch_mode and run_number:
            filename = exp_dir / f"run_{run_number}_experiment_results.json"
        else:
            filename = exp_dir / "experiment_results.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    
    def save_batch_summary(self, batch_results: Dict, batch_id: str) -> None:
        """Save batch results summary."""
        filename = self.results_dir / f"{batch_id}_summary.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)