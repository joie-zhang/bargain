"""Analysis utilities for experiment results and structured behavior summaries."""

from typing import List, Dict, Any

from ..configs import STRONG_MODELS_CONFIG


class ExperimentAnalyzer:
    """Analyze experiment outcomes and structured (non-keyword) behavior signals."""

    @staticmethod
    def detect_exploitation(conversation_logs: List[Dict[str, Any]]) -> bool:
        """Reserved for explicit structured exploitation tags.

        We intentionally avoid keyword-based detection from free-form text.
        """
        for log in conversation_logs:
            if log.get("phase") == "structured_exploitation_signal":
                return bool(log.get("detected", False))
        return False

    @staticmethod
    def analyze_strategic_behaviors(conversation_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize phase-level behavior from structured logs only."""
        behaviors: Dict[str, Any] = {
            "discussion_message_count": 0,
            "pledge_submission_count": 0,
            "feedback_message_count": 0,
            "commit_vote_count": 0,
            "commit_vote_yay_count": 0,
            "commit_vote_nay_count": 0,
            "unanimous_commit_round_count": 0,
        }

        for log in conversation_logs:
            phase = log.get("phase")
            if phase == "discussion":
                behaviors["discussion_message_count"] += 1
            elif phase == "pledge_submission":
                behaviors["pledge_submission_count"] += 1
            elif phase == "feedback":
                behaviors["feedback_message_count"] += 1
            elif phase == "cofunding_commit_vote":
                behaviors["commit_vote_count"] += 1
                parsed_vote = log.get("parsed_vote", {})
                vote = str(parsed_vote.get("commit_vote", "")).strip().lower()
                if vote == "yay":
                    behaviors["commit_vote_yay_count"] += 1
                elif vote == "nay":
                    behaviors["commit_vote_nay_count"] += 1
            elif phase == "cofunding_commit_vote_summary":
                if bool(log.get("unanimous_yay", False)):
                    behaviors["unanimous_commit_round_count"] += 1

        return behaviors

    @staticmethod
    def analyze_agent_performance(agents: List[Any], final_utilities: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze performance of each agent."""
        performance = {}
        for agent in agents:
            model_name = ExperimentAnalyzer.get_model_name_from_agent(agent)
            performance[agent.agent_id] = {
                "final_utility": final_utilities.get(agent.agent_id, 0),
                "model": model_name,
            }
        return performance

    @staticmethod
    def get_model_name_from_agent(agent: Any) -> str:
        """Get model name from agent object."""
        if hasattr(agent, "model_name") and agent.model_name:
            model_name = agent.model_name
            if model_name in STRONG_MODELS_CONFIG:
                return model_name
            for config_key, config_data in STRONG_MODELS_CONFIG.items():
                config_model_id = config_data.get("model_id", "")
                if model_name == config_model_id:
                    return config_key
                if model_name.startswith(config_key + "-") or model_name.startswith(config_key + "_"):
                    return config_key
                if config_model_id and (config_key in model_name or model_name in config_model_id):
                    return config_key
            return model_name
        return ExperimentAnalyzer.get_model_name(agent.agent_id)

    @staticmethod
    def get_model_name(agent_id: str) -> str:
        """Extract model name from agent ID (fallback method)."""
        for model_key in STRONG_MODELS_CONFIG.keys():
            if model_key.replace("-", "_") in agent_id:
                return model_key
        return "unknown"

    @staticmethod
    def aggregate_strategic_behaviors(experiments: List[Any]) -> Dict[str, float]:
        """Aggregate structured behavior summaries across experiments."""
        summary = {
            "avg_discussion_messages": 0.0,
            "avg_pledge_submissions": 0.0,
            "avg_feedback_messages": 0.0,
            "avg_commit_votes": 0.0,
            "commit_vote_yay_rate": 0.0,
            "unanimous_commit_rate": 0.0,
        }

        if not experiments:
            return summary

        total_yay = 0
        total_votes = 0
        total_unanimous = 0

        for exp in experiments:
            behaviors = getattr(exp, "strategic_behaviors", {}) or {}
            summary["avg_discussion_messages"] += float(behaviors.get("discussion_message_count", 0))
            summary["avg_pledge_submissions"] += float(behaviors.get("pledge_submission_count", 0))
            summary["avg_feedback_messages"] += float(behaviors.get("feedback_message_count", 0))
            summary["avg_commit_votes"] += float(behaviors.get("commit_vote_count", 0))
            total_yay += int(behaviors.get("commit_vote_yay_count", 0))
            total_votes += int(behaviors.get("commit_vote_count", 0))
            total_unanimous += 1 if int(behaviors.get("unanimous_commit_round_count", 0)) > 0 else 0

        num_exp = len(experiments)
        summary["avg_discussion_messages"] /= num_exp
        summary["avg_pledge_submissions"] /= num_exp
        summary["avg_feedback_messages"] /= num_exp
        summary["avg_commit_votes"] /= num_exp
        summary["commit_vote_yay_rate"] = (total_yay / total_votes) if total_votes > 0 else 0.0
        summary["unanimous_commit_rate"] = total_unanimous / num_exp

        return summary
    
