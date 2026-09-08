"""Private attempt directories, durable records, and read-only status checks."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid

from .schema import canonical, finite, strict_json

COMPLETE = {"agreement", "disagreement"}


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    payload = canonical(value) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def lock(root):
    import fcntl

    path = Path(root) / ".launch.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this output directory") from exc
        yield fd
    finally:
        # Children inherit the descriptor so a surviving worker retains ownership.
        os.close(fd)


def create_root(root, plan):
    root = Path(root)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    atomic_json(root / "plan.json", plan)
    (root / "runs").mkdir(mode=0o700)
    return root


def new_attempt(root, run, provenance):
    run_dir = Path(root) / "runs" / run["run_id"]
    if (Path(root) / "runs").is_symlink() or run_dir.is_symlink():
        raise ValueError("Run directories must not be symlinks")
    run_dir.mkdir(exist_ok=True, mode=0o700)
    attempt_id = uuid.uuid4().hex
    attempt = run_dir / attempt_id
    attempt.mkdir(mode=0o700)
    atomic_json(attempt / "resolved.json", run)
    atomic_json(attempt / "provenance.json", provenance)
    atomic_json(attempt / "status.json", {"state": "running", "run_id": run["run_id"], "started_at": time.time()})
    atomic_json(run_dir / "current.json", {"attempt_id": attempt_id})
    return attempt


def current_attempt(root, run):
    run_dir = Path(root) / "runs" / run["run_id"]
    if (Path(root) / "runs").is_symlink() or run_dir.is_symlink():
        raise ValueError("Run directories must not be symlinks")
    pointer = run_dir / "current.json"
    if not pointer.exists():
        return None
    record = read_json(pointer)
    attempt_id = record.get("attempt_id")
    if not isinstance(attempt_id, str) or len(attempt_id) != 32 or any(c not in "0123456789abcdef" for c in attempt_id):
        raise ValueError("Invalid attempt pointer")
    attempt = run_dir / attempt_id
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("Attempt directory is missing or is a symlink")
    return attempt


def validate_result(result, run):
    utilities = result.get("final_utilities")
    expected = {seat["agent_id"] for seat in run["seats"]}
    if not isinstance(utilities, dict) or set(utilities) != expected or not all(finite(v) for v in utilities.values()):
        raise ValueError("Result must contain finite utilities for exactly the requested seats")
    if type(result.get("consensus_reached")) is not bool:
        raise ValueError("Result is missing the agreement outcome")
    final_round = result.get("final_round")
    if type(final_round) is not int or not 1 <= final_round <= run["engine"]["t_rounds"]:
        raise ValueError("Result has an invalid terminal round")
    if not result["consensus_reached"] and final_round != run["engine"]["t_rounds"]:
        raise ValueError("Disagreement must reach the requested round cap")
    if not result["consensus_reached"] and any(v != 0 for v in utilities.values()):
        raise ValueError("Completed disagreement must have rule-defined zero utility")
    return "agreement" if result["consensus_reached"] else "disagreement"


def finish(attempt, run, state, error=None):
    prior = read_json(attempt / "status.json")
    artifacts = {str(p.relative_to(attempt)): file_hash(p)
                 for p in sorted(attempt.rglob("*")) if p.is_file() and p.name != "status.json" and p.name != "worker.log"}
    atomic_json(attempt / "status.json", {**prior, "state": state, "finished_at": time.time(),
                                        "error": error, "run_id": run["run_id"], "artifacts": artifacts})


def validate_complete(attempt, run):
    status = read_json(attempt / "status.json")
    if status["state"] not in COMPLETE or status.get("run_id") != run["run_id"]:
        raise ValueError("Attempt is not a complete result for this run")
    artifacts = status.get("artifacts", {})
    for required in ("result.json", "resolved.json", "initial_state.json", "provenance.json", "interactions.jsonl"):
        if required not in artifacts:
            raise ValueError(f"Complete attempt lacks {required}")
    for relative, expected_hash in artifacts.items():
        path = attempt / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.resolve().is_relative_to(attempt.resolve()) or not path.is_file()):
            raise ValueError("Invalid artifact reference")
        if file_hash(path) != expected_hash:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    if read_json(attempt / "resolved.json") != run:
        raise ValueError("Attempt configuration differs from the saved plan")
    if validate_result(read_json(attempt / "result.json"), run) != status["state"]:
        raise ValueError("Status differs from the result outcome")
    return status


def status_report(root, plan):
    rows = []
    for run in plan["runs"]:
        attempt = current_attempt(root, run)
        if attempt is None:
            rows.append({"run_id": run["run_id"], "state": "pending"})
            continue
        record = read_json(attempt / "status.json")
        if record["state"] == "running" and not lock_is_held(root):
            record = {**record, "state": "interrupted", "error": "No process holds the run lock; inspect requests before restarting"}
        if record["state"] in COMPLETE:
            validate_complete(attempt, run)
        rows.append({"run_id": run["run_id"], "state": record["state"], "attempt": str(attempt),
                     "error": record.get("error")})
    return {"output": str(Path(root).resolve()), "requested": len(rows),
            "complete": sum(row["state"] in COMPLETE for row in rows), "runs": rows}


def lock_is_held(root):
    import fcntl

    path = Path(root) / ".launch.lock"
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)
