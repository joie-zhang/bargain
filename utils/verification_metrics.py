"""
Metrics and Tracking System for Self-Verification Loops

This module provides comprehensive metrics collection, analysis, and visualization
for tracking the effectiveness of self-verification loops in AI agent systems.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum


class MetricType(Enum):
    """Types of metrics we track"""

    SUCCESS_RATE = "success_rate"
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    ERROR_RATE = "error_rate"
    RETRY_COUNT = "retry_count"
    FIX_TIME = "fix_time"
    COMPLEXITY = "complexity"
    COST = "cost"


@dataclass
class VerificationMetric:
    """Individual metric measurement"""

    timestamp: datetime
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationMetric":
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_type=MetricType(data["metric_type"]),
            value=data["value"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class VerificationSession:
    """Track metrics for a complete verification session"""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    verifier_name: str = ""
    success: bool = False
    iterations: int = 0
    metrics: List[VerificationMetric] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_metric(
        self, metric_type: MetricType, value: float, metadata: Optional[Dict] = None
    ):
        """Add a metric to this session"""
        self.metrics.append(
            VerificationMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                metadata=metadata or {},
            )
        )

    def duration_seconds(self) -> float:
        """Calculate session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    def get_metrics_by_type(self, metric_type: MetricType) -> List[VerificationMetric]:
        """Get all metrics of a specific type"""
        return [m for m in self.metrics if m.metric_type == metric_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "verifier_name": self.verifier_name,
            "success": self.success,
            "iterations": self.iterations,
            "metrics": [m.to_dict() for m in self.metrics],
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationSession":
        """Create from dictionary"""
        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"])
            if data["end_time"]
            else None,
            verifier_name=data.get("verifier_name", ""),
            success=data.get("success", False),
            iterations=data.get("iterations", 0),
            metrics=[VerificationMetric.from_dict(m) for m in data.get("metrics", [])],
            errors=data.get("errors", []),
        )


class MetricsTracker:
    """Main metrics tracking system"""

    def __init__(self, storage_path: Path = Path("verification_metrics")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.current_sessions: Dict[str, VerificationSession] = {}
        self.completed_sessions: List[VerificationSession] = []
        self._load_historical_data()

    def start_session(self, session_id: str, verifier_name: str) -> VerificationSession:
        """Start tracking a new verification session"""
        session = VerificationSession(
            session_id=session_id,
            start_time=datetime.now(),
            verifier_name=verifier_name,
        )
        self.current_sessions[session_id] = session
        return session

    def end_session(self, session_id: str, success: bool):
        """End a verification session"""
        if session_id in self.current_sessions:
            session = self.current_sessions.pop(session_id)
            session.end_time = datetime.now()
            session.success = success
            self.completed_sessions.append(session)
            self._save_session(session)

    def track_metric(
        self,
        session_id: str,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict] = None,
    ):
        """Track a metric for a specific session"""
        if session_id in self.current_sessions:
            self.current_sessions[session_id].add_metric(metric_type, value, metadata)

    def track_error(self, session_id: str, error: Dict[str, Any]):
        """Track an error in a session"""
        if session_id in self.current_sessions:
            self.current_sessions[session_id].errors.append(
                {"timestamp": datetime.now().isoformat(), **error}
            )

    def increment_iteration(self, session_id: str):
        """Increment the iteration count for a session"""
        if session_id in self.current_sessions:
            self.current_sessions[session_id].iterations += 1

    def _save_session(self, session: VerificationSession):
        """Save a session to disk"""
        filename = self.storage_path / f"session_{session.session_id}.json"
        with open(filename, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _load_historical_data(self):
        """Load historical session data from disk"""
        for filepath in self.storage_path.glob("session_*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                session = VerificationSession.from_dict(data)
                if session.end_time:  # Only load completed sessions
                    self.completed_sessions.append(session)
            except Exception as e:
                print(f"Error loading session {filepath}: {e}")

    def get_aggregate_metrics(
        self,
        verifier_name: Optional[str] = None,
        time_window: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """Get aggregated metrics across sessions"""
        # Filter sessions
        sessions = self.completed_sessions
        if verifier_name:
            sessions = [s for s in sessions if s.verifier_name == verifier_name]
        if time_window:
            cutoff = datetime.now() - time_window
            sessions = [s for s in sessions if s.start_time >= cutoff]

        if not sessions:
            return {"error": "No sessions found matching criteria"}

        # Calculate aggregates
        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions if s.success)

        # Latency metrics
        latencies = []
        for session in sessions:
            latency_metrics = session.get_metrics_by_type(MetricType.LATENCY)
            latencies.extend([m.value for m in latency_metrics])

        # Token usage
        token_usage = []
        for session in sessions:
            token_metrics = session.get_metrics_by_type(MetricType.TOKEN_USAGE)
            token_usage.extend([m.value for m in token_metrics])

        # Cost metrics
        total_cost = 0.0
        for session in sessions:
            cost_metrics = session.get_metrics_by_type(MetricType.COST)
            total_cost += sum(m.value for m in cost_metrics)

        # Fix time metrics
        fix_times = []
        for session in sessions:
            fix_time_metrics = session.get_metrics_by_type(MetricType.FIX_TIME)
            fix_times.extend([m.value for m in fix_time_metrics])

        # Retry counts
        retry_counts = [s.iterations for s in sessions]

        return {
            "total_sessions": total_sessions,
            "success_rate": successful_sessions / total_sessions
            if total_sessions > 0
            else 0,
            "successful_sessions": successful_sessions,
            "failed_sessions": total_sessions - successful_sessions,
            "latency": {
                "mean": statistics.mean(latencies) if latencies else 0,
                "median": statistics.median(latencies) if latencies else 0,
                "p95": statistics.quantiles(latencies, n=20)[18]
                if len(latencies) > 1
                else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
            },
            "token_usage": {
                "total": sum(token_usage),
                "mean_per_session": statistics.mean(token_usage) if token_usage else 0,
                "max": max(token_usage) if token_usage else 0,
            },
            "cost": {
                "total": total_cost,
                "mean_per_session": total_cost / total_sessions
                if total_sessions > 0
                else 0,
            },
            "fix_time": {
                "mean": statistics.mean(fix_times) if fix_times else 0,
                "median": statistics.median(fix_times) if fix_times else 0,
            },
            "retry_counts": {
                "mean": statistics.mean(retry_counts) if retry_counts else 0,
                "max": max(retry_counts) if retry_counts else 0,
                "distribution": dict(
                    pd.Series(retry_counts).value_counts().sort_index()
                ),
            },
            "error_analysis": self._analyze_errors(sessions),
        }

    def _analyze_errors(self, sessions: List[VerificationSession]) -> Dict[str, Any]:
        """Analyze error patterns across sessions"""
        all_errors = []
        for session in sessions:
            all_errors.extend(session.errors)

        if not all_errors:
            return {"total_errors": 0}

        # Categorize errors
        error_categories = defaultdict(int)
        error_messages = []

        for error in all_errors:
            category = error.get("category", "unknown")
            error_categories[category] += 1
            if "message" in error:
                error_messages.append(error["message"])

        return {
            "total_errors": len(all_errors),
            "errors_per_session": len(all_errors) / len(sessions) if sessions else 0,
            "error_categories": dict(error_categories),
            "common_errors": pd.Series(error_messages).value_counts().head(5).to_dict()
            if error_messages
            else {},
        }

    def generate_report(
        self,
        verifier_name: Optional[str] = None,
        time_window: Optional[timedelta] = None,
    ) -> str:
        """Generate a comprehensive metrics report"""
        metrics = self.get_aggregate_metrics(verifier_name, time_window)

        if "error" in metrics:
            return f"# Metrics Report\n\n{metrics['error']}"

        report = ["# Verification Metrics Report"]
        report.append(f"Generated: {datetime.now().isoformat()}")

        if verifier_name:
            report.append(f"Verifier: {verifier_name}")
        if time_window:
            report.append(f"Time Window: Last {time_window.days} days")

        report.append("")
        report.append("## Summary")
        report.append(f"- Total Sessions: {metrics['total_sessions']}")
        report.append(f"- Success Rate: {metrics['success_rate']:.1%}")
        report.append(f"- Successful: {metrics['successful_sessions']}")
        report.append(f"- Failed: {metrics['failed_sessions']}")

        report.append("")
        report.append("## Performance Metrics")
        report.append("### Latency (ms)")
        lat = metrics["latency"]
        report.append(f"- Mean: {lat['mean']:.2f}")
        report.append(f"- Median: {lat['median']:.2f}")
        report.append(f"- P95: {lat['p95']:.2f}")
        report.append(f"- Range: {lat['min']:.2f} - {lat['max']:.2f}")

        report.append("")
        report.append("### Token Usage")
        tokens = metrics["token_usage"]
        report.append(f"- Total: {tokens['total']:,.0f}")
        report.append(f"- Mean per Session: {tokens['mean_per_session']:.0f}")
        report.append(f"- Max per Session: {tokens['max']:.0f}")

        report.append("")
        report.append("### Cost Analysis")
        cost = metrics["cost"]
        report.append(f"- Total Cost: ${cost['total']:.2f}")
        report.append(f"- Mean per Session: ${cost['mean_per_session']:.2f}")

        report.append("")
        report.append("### Fix Time (seconds)")
        fix = metrics["fix_time"]
        report.append(f"- Mean: {fix['mean']:.2f}")
        report.append(f"- Median: {fix['median']:.2f}")

        report.append("")
        report.append("### Retry Analysis")
        retry = metrics["retry_counts"]
        report.append(f"- Mean Retries: {retry['mean']:.2f}")
        report.append(f"- Max Retries: {retry['max']}")
        report.append("- Distribution:")
        for count, freq in retry["distribution"].items():
            report.append(f"  - {count} retries: {freq} sessions")

        report.append("")
        report.append("## Error Analysis")
        errors = metrics["error_analysis"]
        report.append(f"- Total Errors: {errors['total_errors']}")
        report.append(f"- Errors per Session: {errors['errors_per_session']:.2f}")

        if "error_categories" in errors:
            report.append("- Error Categories:")
            for category, count in errors["error_categories"].items():
                report.append(f"  - {category}: {count}")

        if "common_errors" in errors:
            report.append("- Most Common Errors:")
            for error, count in errors["common_errors"].items():
                report.append(f'  - "{error[:50]}...": {count} occurrences')

        return "\n".join(report)

    def plot_metrics(
        self,
        metric_type: MetricType,
        verifier_name: Optional[str] = None,
        time_window: Optional[timedelta] = None,
        save_path: Optional[Path] = None,
    ):
        """Plot metrics over time"""
        # Filter sessions
        sessions = self.completed_sessions
        if verifier_name:
            sessions = [s for s in sessions if s.verifier_name == verifier_name]
        if time_window:
            cutoff = datetime.now() - time_window
            sessions = [s for s in sessions if s.start_time >= cutoff]

        # Extract metric data
        data_points = []
        for session in sessions:
            metrics = session.get_metrics_by_type(metric_type)
            for metric in metrics:
                data_points.append(
                    {
                        "timestamp": metric.timestamp,
                        "value": metric.value,
                        "session_id": session.session_id,
                        "success": session.success,
                    }
                )

        if not data_points:
            print(f"No data points found for {metric_type.value}")
            return

        # Create DataFrame for easier plotting
        df = pd.DataFrame(data_points)
        df = df.sort_values("timestamp")

        # Create plot
        plt.figure(figsize=(12, 6))

        # Scatter plot colored by success
        success_df = df[df["success"]]
        failure_df = df[~df["success"]]

        if not success_df.empty:
            plt.scatter(
                success_df["timestamp"],
                success_df["value"],
                color="green",
                alpha=0.6,
                label="Successful",
            )
        if not failure_df.empty:
            plt.scatter(
                failure_df["timestamp"],
                failure_df["value"],
                color="red",
                alpha=0.6,
                label="Failed",
            )

        # Add trend line
        if len(df) > 1:
            z = np.polyfit(range(len(df)), df["value"], 1)
            p = np.poly1d(z)
            plt.plot(
                df["timestamp"], p(range(len(df))), "b--", alpha=0.8, label="Trend"
            )

        plt.xlabel("Time")
        plt.ylabel(f"{metric_type.value.replace('_', ' ').title()}")
        plt.title(f"{metric_type.value.replace('_', ' ').title()} Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics analysis"""
        metrics = self.get_aggregate_metrics()
        recommendations = []

        # Success rate recommendations
        if metrics["success_rate"] < 0.8:
            recommendations.append(
                f"Success rate is {metrics['success_rate']:.1%}. Consider:\n"
                "  - Improving error handling and recovery strategies\n"
                "  - Adding more comprehensive validation before verification\n"
                "  - Implementing better fallback mechanisms"
            )

        # Latency recommendations
        if metrics["latency"]["p95"] > 5000:  # 5 seconds
            recommendations.append(
                f"P95 latency is {metrics['latency']['p95']:.0f}ms. Consider:\n"
                "  - Implementing caching for frequently verified items\n"
                "  - Parallelizing independent verifications\n"
                "  - Optimizing slow verification steps"
            )

        # Retry recommendations
        if metrics["retry_counts"]["mean"] > 1.5:
            recommendations.append(
                f"Average retry count is {metrics['retry_counts']['mean']:.1f}. Consider:\n"
                "  - Investigating root causes of verification failures\n"
                "  - Implementing smarter retry strategies\n"
                "  - Adding pre-verification checks"
            )

        # Cost recommendations
        if metrics["cost"]["mean_per_session"] > 0.10:  # $0.10 per session
            recommendations.append(
                f"Average cost per session is ${metrics['cost']['mean_per_session']:.2f}. Consider:\n"
                "  - Optimizing token usage in prompts\n"
                "  - Implementing result caching\n"
                "  - Using smaller models for simple verifications"
            )

        # Error recommendations
        error_rate = metrics["error_analysis"]["errors_per_session"]
        if error_rate > 0.5:
            recommendations.append(
                f"Error rate is {error_rate:.1f} errors per session. Consider:\n"
                "  - Implementing better error categorization\n"
                "  - Adding automated error recovery\n"
                "  - Improving input validation"
            )

        return recommendations


class TokenCostCalculator:
    """Calculate costs based on token usage"""

    # Default pricing (can be customized)
    DEFAULT_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }

    @classmethod
    def calculate_cost(
        cls, input_tokens: int, output_tokens: int, model: str = "gpt-4"
    ) -> float:
        """Calculate cost for token usage"""
        if model not in cls.DEFAULT_PRICING:
            model = "gpt-4"  # Default fallback

        pricing = cls.DEFAULT_PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost


# Integration helper for verification framework
class VerificationMetricsIntegration:
    """Helper to integrate metrics with the verification framework"""

    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics = metrics_tracker
        self.session_map: Dict[str, str] = {}  # verifier_name -> session_id

    def start_verification(self, verifier_name: str) -> str:
        """Start tracking metrics for a verification"""
        session_id = f"{verifier_name}_{int(time.time())}"
        session = self.metrics.start_session(session_id, verifier_name)
        self.session_map[verifier_name] = session_id
        return session_id

    def track_verification_result(
        self, verifier_name: str, result: Any, duration_ms: float
    ):
        """Track the result of a verification"""
        if verifier_name not in self.session_map:
            return

        session_id = self.session_map[verifier_name]

        # Track latency
        self.metrics.track_metric(session_id, MetricType.LATENCY, duration_ms)

        # Track success/failure
        if hasattr(result, "status"):
            success = result.status.value == "success"
            if not success:
                self.metrics.track_error(
                    session_id,
                    {
                        "category": result.error_category.value
                        if hasattr(result, "error_category")
                        else "unknown",
                        "message": result.message
                        if hasattr(result, "message")
                        else "Unknown error",
                    },
                )

    def track_token_usage(
        self,
        verifier_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4",
    ):
        """Track token usage and cost"""
        if verifier_name not in self.session_map:
            return

        session_id = self.session_map[verifier_name]

        # Track tokens
        total_tokens = input_tokens + output_tokens
        self.metrics.track_metric(
            session_id,
            MetricType.TOKEN_USAGE,
            total_tokens,
            {"input": input_tokens, "output": output_tokens},
        )

        # Track cost
        cost = TokenCostCalculator.calculate_cost(input_tokens, output_tokens, model)
        self.metrics.track_metric(session_id, MetricType.COST, cost, {"model": model})

    def track_iteration(self, verifier_name: str):
        """Track a retry/iteration"""
        if verifier_name not in self.session_map:
            return

        session_id = self.session_map[verifier_name]
        self.metrics.increment_iteration(session_id)
        self.metrics.track_metric(session_id, MetricType.RETRY_COUNT, 1)

    def end_verification(self, verifier_name: str, success: bool):
        """End tracking for a verification"""
        if verifier_name not in self.session_map:
            return

        session_id = self.session_map[verifier_name]
        self.metrics.end_session(session_id, success)
        del self.session_map[verifier_name]


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Create metrics tracker
    tracker = MetricsTracker()

    # Simulate some verification sessions
    for i in range(20):
        session_id = f"test_session_{i}"
        session = tracker.start_session(session_id, "api_verifier")

        # Simulate metrics
        tracker.track_metric(
            session_id, MetricType.LATENCY, 100 + np.random.normal(0, 20)
        )
        tracker.track_metric(
            session_id, MetricType.TOKEN_USAGE, 1000 + np.random.randint(-200, 500)
        )
        tracker.track_metric(
            session_id, MetricType.COST, 0.05 + np.random.uniform(-0.02, 0.05)
        )

        # Simulate iterations
        iterations = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        for _ in range(iterations):
            tracker.increment_iteration(session_id)
            tracker.track_metric(
                session_id, MetricType.FIX_TIME, 5 + np.random.exponential(2)
            )

        # Simulate errors
        if np.random.random() < 0.2:  # 20% error rate
            tracker.track_error(
                session_id,
                {
                    "category": np.random.choice(
                        ["transient", "permanent", "recoverable"]
                    ),
                    "message": np.random.choice(
                        [
                            "Connection timeout",
                            "Invalid response format",
                            "Rate limit exceeded",
                            "Authentication failed",
                        ]
                    ),
                },
            )

        # End session
        success = np.random.random() > 0.2  # 80% success rate
        tracker.end_session(session_id, success)

    # Generate report
    print(tracker.generate_report())
    print("\n## Recommendations")
    for rec in tracker.get_recommendations():
        print(f"\n{rec}")

    # Plot metrics
    tracker.plot_metrics(MetricType.LATENCY, save_path=Path("latency_plot.png"))
