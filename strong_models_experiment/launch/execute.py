"""Isolated local workers for resolved plans; no engine import in the parent."""

import asyncio
import hashlib
from importlib import metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .plan import validate_plan
from .providers import doctor, redact
from .schema import canonical, digest
from .store import (COMPLETE, atomic_json, create_root, current_attempt, finish, lock,
                    new_attempt, read_json, status_report, validate_complete, validate_result)


def provenance():
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for package in ("strong_models_experiment", "negotiation", "game_environments"):
        for path in sorted((root / package).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".md", ".json") and "__pycache__" not in path.parts:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {distribution.metadata["Name"]: distribution.version for distribution in metadata.distributions()}
    relevant = {name: metadata.version(name) for name in ("numpy", "scipy", "aiohttp")}
    compatibility = {"sources": sources, "dependencies": relevant, "python": sys.version}
    revision = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            revision = result.stdout.strip()
    return {"compatibility_hash": digest(compatibility), **compatibility,
            "all_installed_packages": versions, "git_revision": revision}


def worker_environment(credentials):
    # Do not inherit old prompt controls, endpoint overrides, key pools, queue
    # paths, Python import paths, or working-directory .env discovery.
    env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "TMPDIR") if name in os.environ}
    env.update(credentials)
    env.update({"PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "NEGOTIATION_TOKEN_ESTIMATOR": "chars-v1", "NEGOTIATION_DISABLE_CONTEXT_COMPACTION": "1",
                "OPENAI_TRANSPORT": "direct", "OPENROUTER_TRANSPORT": "direct",
                "OPENROUTER_PROVIDER_FALLBACK": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    return env


def execute(plan, root, credentials, *, resume=False, retry_failed=False, accept_unknown=False):
    validate_plan(plan)
    setup = doctor(plan, credentials)
    if not setup["ready_offline"]:
        raise ValueError("Setup incomplete: " + canonical(setup))
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("This release supports direct hosted APIs on network-connected hosts, not Slurm jobs")
    root = Path(root).expanduser().resolve()
    if not resume:
        create_root(root, plan)
    elif read_json(root / "plan.json") != plan:
        raise ValueError("Output directory belongs to another plan")
    code = provenance()
    with lock(root) as lock_fd:
        for run in plan["runs"]:
            previous = current_attempt(root, run)
            recovery = None
            if previous is not None:
                prior = read_json(previous / "status.json")
                if read_json(previous / "provenance.json")["compatibility_hash"] != code["compatibility_hash"]:
                    raise ValueError("Code or runtime versions changed; use a new output directory")
                if prior["state"] in COMPLETE:
                    validate_complete(previous, run)
                    continue
                requests = [read_json(p) for p in sorted((previous / "requests").glob("*.json"))]
                unknown = prior["state"] == "unknown_request_outcome" or any(
                    r["state"] in ("dispatching", "unknown_request_outcome") for r in requests)
                if not retry_failed:
                    raise ValueError("An incomplete attempt exists; use --retry-failed to start a new attempt")
                if unknown and not accept_unknown:
                    raise ValueError("A provider request outcome is unknown; inspect it before using --accept-unknown-outcome (possible duplicate billing)")
                recovery = {"previous_attempt": previous.name, "previous_state": prior["state"],
                            "action": "restart-whole-negotiation", "unknown_outcome_accepted": unknown and accept_unknown}
            attempt = new_attempt(root, run, code)
            if recovery is not None:
                atomic_json(attempt / "recovery.json", recovery)
            print(f"Running {run['preset']} ({run['effort'] or run['engine']['game_type']}): {attempt}", flush=True)
            # The installed console entry point and module entry point both use
            # this interpreter. No personal checkout path is embedded in code.
            command = [sys.executable, "-m", "strong_models_experiment.launch.execute", str(attempt)]
            with (attempt / "worker.log").open("x", encoding="utf-8") as log:
                try:
                    process = subprocess.Popen(command, cwd=attempt, env=worker_environment(credentials),
                                               stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,), start_new_session=True)
                except OSError as exc:
                    finish(attempt, run, "failed", f"Worker could not start: {exc}")
                    return 1
                try:
                    code_result = process.wait()
                except KeyboardInterrupt:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record_interruption(attempt, run)
                    raise
            record = read_json(attempt / "status.json")
            if record["state"] == "running":
                record_interruption(attempt, run)
                record = read_json(attempt / "status.json")
            if code_result != 0 or record["state"] not in COMPLETE:
                print(f"Stopped with {record['state']}: {attempt / 'status.json'}", flush=True)
                return 1
            validate_complete(attempt, run)
    print(canonical(status_report(root, plan)))
    return 0


def record_interruption(attempt, run):
    if read_json(attempt / "status.json")["state"] in COMPLETE:
        return
    pending = any(read_json(p)["state"] in ("dispatching", "unknown_request_outcome")
                  for p in (attempt / "requests").glob("*.json"))
    finish(attempt, run, "unknown_request_outcome" if pending else "canceled",
           "Worker ended before publishing a complete result")


async def run_worker(attempt):
    # Engine imports happen only after the parent's environment allowlist is set.
    from strong_models_experiment.experiment import StrongModelsExperiment
    from strong_models_experiment.utils.experiment_utils import FileManager
    from .providers import RequestJournal, UnknownRequestOutcome, load_credentials
    from .runtime import HostedAgentFactory, StrictPhaseHandler
    import logging

    os.umask(0o077)
    run = read_json(attempt / "resolved.json")
    credentials = load_credentials()
    journal = RequestJournal(attempt, credentials, run["timeout_seconds"])

    class SecretFilter(logging.Filter):
        def filter(self, record):
            record.msg = redact(record.getMessage(), credentials)
            record.args = ()
            return True

    def save_state(state, config, environment):
        journal.game_state = state
        journal.game_environment = environment
        atomic_json(attempt / "initial_state.json", state)
        atomic_json(attempt / "effective_engine_config.json", config)
        if run["preset"] == "ttc":
            root = attempt.parents[2]
            plan = read_json(root / "plan.json")
            for other in plan["runs"]:
                if other["run_id"] == run["run_id"]:
                    continue
                previous = current_attempt(root, other)
                if previous is not None and (previous / "initial_state.json").exists():
                    if read_json(previous / "initial_state.json") != state:
                        raise ValueError("TTC effort conditions generated different game instances")

    class AtomicResults(FileManager):
        def save_experiment_result(self, result, exp_dir, *args):
            if journal.failure is not None:
                raise journal.failure
            validate_result(result, run)
            atomic_json(attempt / "engine_result.json", redact(result, credentials))

    class RecordedExperiment(StrongModelsExperiment):
        def _setup_logging(self):
            logger = logging.getLogger("bargain.worker")
            logger.setLevel(logging.INFO)
            return logger

        def _save_interaction(self, agent_id, phase, prompt, response, round_num=None, token_usage=None, model_name=None, **kwargs):
            record = redact({"agent_id": agent_id, "phase": phase, "prompt": prompt,
                             "response": response, "round": round_num, "token_usage": token_usage,
                             "model_name": model_name, "timestamp": time.time()}, credentials)
            self.all_interactions.append(record)
            self.agent_interactions.setdefault(agent_id, []).append(record)
            with (attempt / "interactions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        def _stream_save_json(self, changed_agent_id=None):
            atomic_json(attempt / "interactions.json", self.all_interactions)

    try:
        logging.basicConfig(level=logging.INFO)
        if provenance()["compatibility_hash"] != read_json(attempt / "provenance.json")["compatibility_hash"]:
            raise ValueError("Code changed between planning execution and worker startup")
        experiment = RecordedExperiment(output_dir=attempt,
                                        agent_factory=HostedAgentFactory(run, journal),
                                        phase_handler_class=StrictPhaseHandler, game_state_callback=save_state)
        experiment.file_manager = AtomicResults(attempt)
        for logger in [logging.getLogger(), experiment.logger]:
            for handler in logger.handlers:
                handler.addFilter(SecretFilter())
        result = await experiment.run_single_experiment([s["name"] for s in run["seats"]], run["engine"])
        if journal.failure is not None:
            raise journal.failure
        payload = result.to_dict()
        outcome = validate_result(payload, run)
        atomic_json(attempt / "result.json", redact(payload, credentials))
        finish(attempt, run, outcome)
        return 0
    except Exception as exc:
        actual = journal.failure if journal.failure is not None else exc
        state = "unknown_request_outcome" if isinstance(actual, UnknownRequestOutcome) else "failed"
        finish(attempt, run, state, redact(f"{type(actual).__name__}: {actual}", credentials))
        return 1
    finally:
        await journal.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_worker(Path(sys.argv[1]).resolve())))
