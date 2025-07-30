"""
Verification Framework for Self-Validating AI Agents

This module provides a structured approach to creating self-verification loops
for AI coding agents, addressing the gaps identified in the foundational approach.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import asyncio
from abc import ABC, abstractmethod


class VerificationStatus(Enum):
    """Status of a verification attempt"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY = "retry"


class ErrorCategory(Enum):
    """Categories of errors for proper handling"""

    TRANSIENT = "transient"  # Network issues, timeouts
    PERMANENT = "permanent"  # Logic errors, invalid inputs
    RECOVERABLE = "recoverable"  # Can be fixed with retry/adjustment
    CRITICAL = "critical"  # Requires human intervention


@dataclass
class VerificationResult:
    """Result of a verification attempt"""

    status: VerificationStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_category: Optional[ErrorCategory] = None
    suggested_fix: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None


@dataclass
class VerificationHistory:
    """Track verification attempts over time"""

    results: List[VerificationResult] = field(default_factory=list)

    def success_rate(self, window: Optional[int] = None) -> float:
        """Calculate success rate over last N attempts"""
        results = self.results[-window:] if window else self.results
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.status == VerificationStatus.SUCCESS)
        return successes / len(results)

    def average_duration(self) -> float:
        """Average duration of verification attempts"""
        durations = [r.duration_ms for r in self.results if r.duration_ms]
        return sum(durations) / len(durations) if durations else 0.0

    def improvement_trend(self) -> Dict[str, Any]:
        """Analyze improvement over time"""
        if len(self.results) < 2:
            return {"trend": "insufficient_data"}

        # Compare first half to second half
        mid = len(self.results) // 2
        first_half = self.results[:mid]
        second_half = self.results[mid:]

        first_success = sum(
            1 for r in first_half if r.status == VerificationStatus.SUCCESS
        ) / len(first_half)
        second_success = sum(
            1 for r in second_half if r.status == VerificationStatus.SUCCESS
        ) / len(second_half)

        return {
            "trend": "improving" if second_success > first_success else "declining",
            "first_half_success_rate": first_success,
            "second_half_success_rate": second_success,
            "improvement": second_success - first_success,
        }


class BaseVerifier(ABC):
    """Base class for all verifiers"""

    def __init__(self, name: str, timeout_seconds: float = 30.0):
        self.name = name
        self.timeout = timeout_seconds
        self.history = VerificationHistory()
        self.logger = logging.getLogger(f"verifier.{name}")

    @abstractmethod
    async def verify(self, target: Any) -> VerificationResult:
        """Perform verification on target"""
        pass

    async def verify_with_retry(
        self, target: Any, max_retries: int = 3, backoff_factor: float = 2.0
    ) -> VerificationResult:
        """Verify with exponential backoff retry"""
        for attempt in range(max_retries):
            result = await self.verify(target)
            self.history.results.append(result)

            if result.status == VerificationStatus.SUCCESS:
                return result

            if result.error_category == ErrorCategory.PERMANENT:
                return result

            if attempt < max_retries - 1:
                wait_time = backoff_factor**attempt
                self.logger.info(
                    f"Retry {attempt + 1}/{max_retries} after {wait_time}s"
                )
                await asyncio.sleep(wait_time)

        return result


class CommandVerifier(BaseVerifier):
    """Verify by running shell commands"""

    def __init__(
        self,
        name: str,
        command: Union[str, List[str]],
        success_criteria: Optional[Callable[[str], bool]] = None,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(name, timeout_seconds)
        self.command = command
        self.success_criteria = success_criteria or (lambda output: True)

    async def verify(self, target: Any) -> VerificationResult:
        """Run command and check output"""
        start_time = time.time()

        try:
            # Prepare command
            if isinstance(self.command, list):
                cmd = self.command
            else:
                cmd = (
                    self.command.format(target=target)
                    if "{target}" in self.command
                    else self.command
                )

            # Run command
            process = await asyncio.create_subprocess_shell(
                cmd if isinstance(cmd, str) else " ".join(cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return VerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    message=f"Command timed out after {self.timeout}s",
                    error_category=ErrorCategory.TRANSIENT,
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Check success
            output = stdout.decode("utf-8")
            if process.returncode == 0 and self.success_criteria(output):
                return VerificationResult(
                    status=VerificationStatus.SUCCESS,
                    message="Command succeeded",
                    metrics={"output_length": len(output)},
                    duration_ms=(time.time() - start_time) * 1000,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    message=f"Command failed: {stderr.decode('utf-8')}",
                    error_category=ErrorCategory.RECOVERABLE,
                    suggested_fix="Check command syntax and target state",
                    duration_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                message=f"Exception: {str(e)}",
                error_category=ErrorCategory.PERMANENT,
                duration_ms=(time.time() - start_time) * 1000,
            )


class APIVerifier(BaseVerifier):
    """Verify API endpoints"""

    def __init__(
        self,
        name: str,
        endpoint: str,
        expected_status: int = 200,
        response_validator: Optional[Callable[[Dict], bool]] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name, timeout_seconds)
        self.endpoint = endpoint
        self.expected_status = expected_status
        self.response_validator = response_validator

    async def verify(self, target: Any) -> VerificationResult:
        """Check API endpoint health"""
        start_time = time.time()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.endpoint, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == self.expected_status:
                        data = (
                            await response.json()
                            if response.content_type == "application/json"
                            else None
                        )

                        if self.response_validator and data:
                            if self.response_validator(data):
                                return VerificationResult(
                                    status=VerificationStatus.SUCCESS,
                                    message="API endpoint verified",
                                    metrics={
                                        "response_time_ms": (time.time() - start_time)
                                        * 1000
                                    },
                                    duration_ms=(time.time() - start_time) * 1000,
                                )
                            else:
                                return VerificationResult(
                                    status=VerificationStatus.FAILURE,
                                    message="Response validation failed",
                                    error_category=ErrorCategory.RECOVERABLE,
                                    duration_ms=(time.time() - start_time) * 1000,
                                )
                        else:
                            return VerificationResult(
                                status=VerificationStatus.SUCCESS,
                                message=f"API returned {response.status}",
                                duration_ms=(time.time() - start_time) * 1000,
                            )
                    else:
                        return VerificationResult(
                            status=VerificationStatus.FAILURE,
                            message=f"Unexpected status: {response.status}",
                            error_category=ErrorCategory.TRANSIENT,
                            duration_ms=(time.time() - start_time) * 1000,
                        )

        except asyncio.TimeoutError:
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                message="API request timed out",
                error_category=ErrorCategory.TRANSIENT,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                message=f"API error: {str(e)}",
                error_category=ErrorCategory.TRANSIENT,
                duration_ms=(time.time() - start_time) * 1000,
            )


class PerformanceVerifier(BaseVerifier):
    """Verify performance characteristics"""

    def __init__(
        self,
        name: str,
        benchmark_command: str,
        threshold_ms: float,
        percentile: float = 0.95,
    ):
        super().__init__(name)
        self.benchmark_command = benchmark_command
        self.threshold_ms = threshold_ms
        self.percentile = percentile

    async def verify(self, target: Any) -> VerificationResult:
        """Run performance benchmark"""
        # Implementation would run benchmarks and check against thresholds
        # This is a simplified version
        start_time = time.time()

        try:
            # Run benchmark command multiple times
            measurements = []
            for _ in range(10):
                result = await self._run_single_benchmark()
                measurements.append(result)

            # Calculate percentile
            measurements.sort()
            p95_index = int(len(measurements) * self.percentile)
            p95_value = measurements[p95_index]

            if p95_value <= self.threshold_ms:
                return VerificationResult(
                    status=VerificationStatus.SUCCESS,
                    message=f"P{int(self.percentile * 100)} latency: {p95_value:.2f}ms",
                    metrics={
                        "p95_ms": p95_value,
                        "median_ms": measurements[len(measurements) // 2],
                        "min_ms": measurements[0],
                        "max_ms": measurements[-1],
                    },
                    duration_ms=(time.time() - start_time) * 1000,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    message=f"Performance threshold exceeded: {p95_value:.2f}ms > {self.threshold_ms}ms",
                    error_category=ErrorCategory.RECOVERABLE,
                    suggested_fix="Profile code for bottlenecks",
                    metrics={"p95_ms": p95_value},
                    duration_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                message=f"Benchmark error: {str(e)}",
                error_category=ErrorCategory.PERMANENT,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _run_single_benchmark(self) -> float:
        """Run a single benchmark iteration"""
        # Simplified - would actually run the benchmark command
        await asyncio.sleep(0.01)  # Simulate work
        return 50.0 + (time.time() % 20)  # Simulated measurement


class VerificationOrchestrator:
    """Orchestrate multiple verifiers with dependencies"""

    def __init__(self):
        self.verifiers: Dict[str, BaseVerifier] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.results_cache: Dict[str, VerificationResult] = {}

    def add_verifier(
        self, verifier: BaseVerifier, depends_on: Optional[List[str]] = None
    ):
        """Add a verifier with optional dependencies"""
        self.verifiers[verifier.name] = verifier
        self.dependencies[verifier.name] = depends_on or []

    async def run_verification_suite(
        self, target: Any, parallel: bool = True
    ) -> Dict[str, VerificationResult]:
        """Run all verifiers respecting dependencies"""
        results = {}

        if parallel:
            # Run in parallel where possible
            pending = set(self.verifiers.keys())
            in_progress = {}

            while pending or in_progress:
                # Start verifiers whose dependencies are met
                ready = []
                for name in pending:
                    deps = self.dependencies[name]
                    if all(d in results for d in deps):
                        ready.append(name)

                # Launch ready verifiers
                for name in ready:
                    pending.remove(name)
                    verifier = self.verifiers[name]
                    in_progress[name] = asyncio.create_task(
                        verifier.verify_with_retry(target)
                    )

                # Wait for any to complete
                if in_progress:
                    done, _ = await asyncio.wait(
                        in_progress.values(), return_when=asyncio.FIRST_COMPLETED
                    )

                    # Collect results
                    for task in done:
                        for name, task_ref in list(in_progress.items()):
                            if task_ref == task:
                                results[name] = await task
                                del in_progress[name]
                                break
        else:
            # Run sequentially
            for name, verifier in self.verifiers.items():
                # Check dependencies
                for dep in self.dependencies[name]:
                    if (
                        dep not in results
                        or results[dep].status != VerificationStatus.SUCCESS
                    ):
                        results[name] = VerificationResult(
                            status=VerificationStatus.FAILURE,
                            message=f"Dependency {dep} failed",
                            error_category=ErrorCategory.PERMANENT,
                        )
                        continue

                results[name] = await verifier.verify_with_retry(target)

        return results

    def generate_report(self, results: Dict[str, VerificationResult]) -> str:
        """Generate a comprehensive verification report"""
        report = [
            "# Verification Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        # Summary
        total = len(results)
        successes = sum(
            1 for r in results.values() if r.status == VerificationStatus.SUCCESS
        )
        report.append(
            f"## Summary: {successes}/{total} passed ({successes / total * 100:.1f}%)"
        )
        report.append("")

        # Details
        report.append("## Verification Details")
        for name, result in results.items():
            status_emoji = "✅" if result.status == VerificationStatus.SUCCESS else "❌"
            report.append(f"### {status_emoji} {name}")
            report.append(f"- Status: {result.status.value}")
            report.append(f"- Message: {result.message}")
            if result.duration_ms:
                report.append(f"- Duration: {result.duration_ms:.2f}ms")
            if result.metrics:
                report.append(f"- Metrics: {json.dumps(result.metrics, indent=2)}")
            if result.suggested_fix:
                report.append(f"- Suggested Fix: {result.suggested_fix}")
            report.append("")

        # Historical trends
        report.append("## Historical Trends")
        for name, verifier in self.verifiers.items():
            if verifier.history.results:
                trend = verifier.history.improvement_trend()
                report.append(f"### {name}")
                report.append(f"- Success Rate: {verifier.history.success_rate():.1%}")
                report.append(
                    f"- Avg Duration: {verifier.history.average_duration():.2f}ms"
                )
                report.append(f"- Trend: {trend['trend']}")
                report.append("")

        return "\n".join(report)


# Example usage patterns
def create_standard_verifiers() -> VerificationOrchestrator:
    """Create a standard set of verifiers for a web application"""
    orchestrator = VerificationOrchestrator()

    # Basic health check
    orchestrator.add_verifier(
        APIVerifier("health_check", "http://localhost:8000/health")
    )

    # Database connectivity
    orchestrator.add_verifier(
        CommandVerifier(
            "database_check",
            "psql -c 'SELECT 1' -h localhost -U myuser mydb",
            success_criteria=lambda out: "1" in out,
        ),
        depends_on=["health_check"],
    )

    # API performance
    orchestrator.add_verifier(
        PerformanceVerifier(
            "api_performance",
            "curl -w '%{time_total}' http://localhost:8000/api/items",
            threshold_ms=200,
        ),
        depends_on=["health_check", "database_check"],
    )

    # Frontend build
    orchestrator.add_verifier(
        CommandVerifier(
            "frontend_build",
            "npm run build",
            success_criteria=lambda out: "Build completed" in out,
        )
    )

    # Type checking
    orchestrator.add_verifier(
        CommandVerifier(
            "type_check",
            "npm run typecheck && python -m mypy src/",
            success_criteria=lambda out: "Success" in out or not out.strip(),
        )
    )

    return orchestrator


# Integration with Claude's workflow
class ClaudeVerificationHelper:
    """Helper to integrate verification into Claude's workflow"""

    def __init__(self, orchestrator: VerificationOrchestrator):
        self.orchestrator = orchestrator
        self.verification_log = Path("verification_log.json")

    async def verify_and_fix(self, max_iterations: int = 3) -> bool:
        """Run verification and attempt fixes until success or max iterations"""
        for iteration in range(max_iterations):
            print(f"\n🔄 Verification iteration {iteration + 1}/{max_iterations}")

            # Run verification suite
            results = await self.orchestrator.run_verification_suite(target=None)

            # Log results
            self._log_results(results)

            # Check if all passed
            all_passed = all(
                r.status == VerificationStatus.SUCCESS for r in results.values()
            )
            if all_passed:
                print("✅ All verifications passed!")
                return True

            # Generate fix suggestions
            fixes_needed = []
            for name, result in results.items():
                if result.status != VerificationStatus.SUCCESS:
                    fixes_needed.append(
                        {
                            "verifier": name,
                            "error": result.message,
                            "suggested_fix": result.suggested_fix,
                            "category": result.error_category.value
                            if result.error_category
                            else "unknown",
                        }
                    )

            print(f"\n❌ {len(fixes_needed)} verifications failed:")
            for fix in fixes_needed:
                print(f"  - {fix['verifier']}: {fix['error']}")
                if fix["suggested_fix"]:
                    print(f"    💡 {fix['suggested_fix']}")

            # Here Claude would read the fixes and attempt to resolve them
            # For now, we'll just continue to the next iteration

        print(f"\n⚠️ Max iterations reached. Some verifications still failing.")
        return False

    def _log_results(self, results: Dict[str, VerificationResult]):
        """Log verification results for analysis"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "results": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "metrics": result.metrics,
                }
                for name, result in results.items()
            },
        }

        # Append to log file
        logs = []
        if self.verification_log.exists():
            with open(self.verification_log, "r") as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(self.verification_log, "w") as f:
            json.dump(logs, f, indent=2)

    def analyze_verification_history(self) -> Dict[str, Any]:
        """Analyze historical verification data"""
        if not self.verification_log.exists():
            return {"error": "No verification history found"}

        with open(self.verification_log, "r") as f:
            logs = json.load(f)

        if not logs:
            return {"error": "Empty verification history"}

        # Analyze patterns
        total_runs = len(logs)
        verifier_stats = {}

        for log in logs:
            for name, result in log["results"].items():
                if name not in verifier_stats:
                    verifier_stats[name] = {
                        "total": 0,
                        "successes": 0,
                        "failures": 0,
                        "avg_duration_ms": [],
                    }

                stats = verifier_stats[name]
                stats["total"] += 1

                if result["status"] == "success":
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

                if result.get("duration_ms"):
                    stats["avg_duration_ms"].append(result["duration_ms"])

        # Calculate aggregates
        for name, stats in verifier_stats.items():
            stats["success_rate"] = (
                stats["successes"] / stats["total"] if stats["total"] > 0 else 0
            )
            stats["avg_duration_ms"] = (
                sum(stats["avg_duration_ms"]) / len(stats["avg_duration_ms"])
                if stats["avg_duration_ms"]
                else 0
            )

        return {
            "total_verification_runs": total_runs,
            "verifier_statistics": verifier_stats,
            "last_run": logs[-1]["timestamp"] if logs else None,
        }


# Example: Creating a custom verifier for research code
class ExperimentReproducibilityVerifier(BaseVerifier):
    """Verify that experiments are reproducible"""

    def __init__(self, experiment_script: str, seed: int = 42):
        super().__init__("reproducibility_check")
        self.experiment_script = experiment_script
        self.seed = seed

    async def verify(self, target: Any) -> VerificationResult:
        """Run experiment twice and compare results"""
        start_time = time.time()

        try:
            # Run experiment twice with same seed
            results = []
            for run in range(2):
                process = await asyncio.create_subprocess_shell(
                    f"python {self.experiment_script} --seed {self.seed}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    return VerificationResult(
                        status=VerificationStatus.FAILURE,
                        message=f"Experiment failed: {stderr.decode('utf-8')}",
                        error_category=ErrorCategory.PERMANENT,
                        duration_ms=(time.time() - start_time) * 1000,
                    )

                # Extract metrics from output (simplified)
                output = stdout.decode("utf-8")
                # In reality, would parse specific metrics
                results.append(output)

            # Compare results
            if results[0] == results[1]:
                return VerificationResult(
                    status=VerificationStatus.SUCCESS,
                    message="Experiment is reproducible",
                    metrics={"seed": self.seed},
                    duration_ms=(time.time() - start_time) * 1000,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    message="Results differ between runs",
                    error_category=ErrorCategory.PERMANENT,
                    suggested_fix="Check for uncontrolled randomness or system dependencies",
                    duration_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                message=f"Error running experiment: {str(e)}",
                error_category=ErrorCategory.PERMANENT,
                duration_ms=(time.time() - start_time) * 1000,
            )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create orchestrator with standard verifiers
        orchestrator = create_standard_verifiers()

        # Add custom research verifier
        orchestrator.add_verifier(
            ExperimentReproducibilityVerifier("experiments/train_model.py"),
            depends_on=["database_check"],
        )

        # Create helper
        helper = ClaudeVerificationHelper(orchestrator)

        # Run verification and fix loop
        success = await helper.verify_and_fix(max_iterations=3)

        # Generate report
        results = await orchestrator.run_verification_suite(target=None)
        report = orchestrator.generate_report(results)
        print("\n" + report)

        # Analyze history
        analysis = helper.analyze_verification_history()
        print("\n## Historical Analysis")
        print(json.dumps(analysis, indent=2))

    # Run example
    asyncio.run(main())
