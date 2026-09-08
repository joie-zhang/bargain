"""Opt-in real OpenAI smoke tests with a shared spending guard.

This is test instrumentation, not an experiment launcher or research dataset.
It runs the public CLI and substitutes only the worker bootstrap so each real
HTTP request can reserve budget before dispatch. Payloads and answers are not
changed. Rates are standard USD per million tokens, checked on 2026-09-08 at
https://developers.openai.com/api/docs/pricing .
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from strong_models_experiment.launch.store import atomic_json, read_json


RATES = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "o3-mini-2025-01-31": (1.10, 4.40),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-2025-08-07": (1.25, 10.00),
    "gpt-5.4": (2.50, 15.00),
}


def ledger_lock(directory):
    # Production locks fail immediately; parallel test reservations can wait.
    from contextlib import contextmanager
    import fcntl

    @contextmanager
    def acquire():
        fd = os.open(directory / ".budget.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)
    return acquire()


def reserve(directory, case, model, payload):
    input_rate, output_rate = RATES[model]
    cap = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if type(cap) is not int or cap <= 0:
        raise ValueError("Smoke requests require a finite output cap")
    # UTF-8 bytes bound text BPE tokens; extra bytes cover role/message overhead.
    # No tools, images, or external operations are allowed in these requests.
    if any(k in payload for k in ("tools", "functions", "web_search_options")):
        raise ValueError("Smoke budget does not cover tools")
    input_bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024
    amount = math.ceil(1.10 * (input_bound * input_rate + cap * output_rate))
    with ledger_lock(directory):
        ledger = read_json(directory / "budget.json")
        if ledger.get("halt_reason"):
            raise RuntimeError("Spending guard is halted: " + ledger["halt_reason"])
        records = ledger["requests"]
        charged = sum(r["accounted_micro_usd"] for r in records.values())
        case_total = sum(r["accounted_micro_usd"] for r in records.values() if r["case"] == case)
        if charged + amount > ledger["limit_micro_usd"]:
            raise RuntimeError("Shared smoke-test budget exhausted before dispatch")
        if case_total + amount > ledger["cases"][case]["limit_micro_usd"]:
            raise RuntimeError("This smoke-test case has exhausted its assigned budget")
        ticket = uuid.uuid4().hex
        records[ticket] = {"case": case, "model": model, "state": "reserved",
                           "accounted_micro_usd": amount, "reserved_micro_usd": amount,
                           "created_at": time.time()}
        atomic_json(directory / "budget.json", ledger)
    return ticket


def settle(directory, ticket, data):
    usage = data.get("usage") if isinstance(data, dict) else None
    with ledger_lock(directory):
        ledger = read_json(directory / "budget.json")
        record = ledger["requests"][ticket]
        if (isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
                and type(usage.get("completion_tokens")) is int
                and usage["prompt_tokens"] >= 0 and usage["completion_tokens"] >= 0):
            input_rate, output_rate = RATES[record["model"]]
            # Charge all input at the uncached rate, a conservative cost estimate.
            cost = math.ceil(usage["prompt_tokens"] * input_rate + usage["completion_tokens"] * output_rate)
            record.update(state="usage_received", accounted_micro_usd=cost, usage=usage)
            exceeded = cost > record["reserved_micro_usd"]
            if exceeded:
                ledger["halt_reason"] = "Actual token cost exceeded a reservation"
        else:
            record["state"] = "unknown_cost_reservation_retained"
            exceeded = False
        atomic_json(directory / "budget.json", ledger)
        if exceeded:
            raise RuntimeError("Actual token cost exceeded the conservative reservation")


def install_guard(directory, case):
    from strong_models_experiment.launch.providers import RequestJournal
    original = RequestJournal.post

    async def guarded(self, seat, payload, context_record):
        if seat["provider"] != "openai" or seat["endpoint"] != "https://api.openai.com/v1/chat/completions":
            raise ValueError("These smoke tests authorize direct OpenAI only")
        ticket = reserve(directory, case, seat["model_id"], payload)
        try:
            response, reference = await original(self, seat, payload, context_record)
        except BaseException:
            # A failure keeps the whole reservation, including unknown billing.
            settle(directory, ticket, None)
            raise
        settle(directory, ticket, response)
        return response, reference

    RequestJournal.post = guarded


def run_case(directory, case, retry_failed=False, credential_env="OPENAI_API_KEY"):
    from strong_models_experiment.cli import main
    from strong_models_experiment.launch.execute import worker_environment
    from strong_models_experiment.launch.plan import build_plan
    from strong_models_experiment.launch.schema import LaunchRequest

    with ledger_lock(directory):
        settings = read_json(directory / "budget.json")["cases"][case]
    request = LaunchRequest(**settings["request"])
    plan = build_plan(request)
    if any(s["provider"] != "openai" for r in plan["runs"] for s in r["seats"]):
        raise ValueError("Case would use a provider outside the user's authorization")
    key = os.environ.get(credential_env)
    if not key:
        raise ValueError("The explicitly selected OpenAI credential is unavailable")
    clean = worker_environment({"OPENAI_API_KEY": key})
    os.environ.clear()
    os.environ.update(clean)
    output = directory / case
    argv = ["run", request.preset, "--output", str(output)]
    for field, value in settings["request"].items():
        if field != "preset" and value is not None:
            argv.extend(["--" + field.replace("_", "-"), str(value)])
    if retry_failed:
        argv = ["resume", str(output), "--retry-failed"]

    original_popen = subprocess.Popen
    script = str(Path(__file__).resolve())

    def instrumented_popen(command, *args, **kwargs):
        if isinstance(command, list) and command[1:3] == ["-m", "strong_models_experiment.launch.execute"]:
            command = [command[0], script, "worker", str(directory), case, command[3]]
        return original_popen(command, *args, **kwargs)

    # Keep the public parent, subprocess isolation, saved plan, and engine intact.
    subprocess.Popen = instrumented_popen
    try:
        started = time.time()
        rc = main(argv)
        summary = {"case": case, "cli_argv": argv, "returncode": rc,
                   "started_at": started, "finished_at": time.time(),
                   "output": str(output), "kind": "live-openai-smoke-not-research",
                   "explicit_failed_restart": retry_failed,
                   "worker_instrumentation": "shared-pre-dispatch-spending-guard"}
        if (output / "plan.json").exists():
            from strong_models_experiment.launch.store import status_report
            saved = read_json(output / "plan.json")
            summary["status"] = status_report(output, saved)
            if rc == 0:
                before = sum(r["case"] == case for r in read_json(directory / "budget.json")["requests"].values())
                summary["resume_returncode"] = main(["resume", str(output)])
                after = sum(r["case"] == case for r in read_json(directory / "budget.json")["requests"].values())
                summary["resume_additional_requests"] = after - before
                summary["verified_complete_resume"] = summary["resume_returncode"] == 0 and after == before
        records = read_json(directory / "budget.json")["requests"]
        own = [r for r in records.values() if r["case"] == case]
        summary["request_count"] = len(own)
        summary["accounted_usd_upper_estimate"] = sum(r["accounted_micro_usd"] for r in own) / 1_000_000
        summary["unknown_cost_count"] = sum(r["state"] != "usage_received" for r in own)
        suffix = "_retry_summary.json" if retry_failed else "_summary.json"
        atomic_json(directory / (case + suffix), summary)
        print(json.dumps(summary, indent=2))
        return rc
    finally:
        subprocess.Popen = original_popen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("case", "worker"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("case")
    parser.add_argument("attempt", nargs="?", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--credential-env", default="OPENAI_API_KEY",
                        help="Name of the environment variable containing the OpenAI key")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == "worker":
        if args.attempt is None:
            parser.error("worker requires an attempt directory")
        install_guard(directory, args.case)
        from strong_models_experiment.launch.execute import run_worker
        return asyncio.run(run_worker(args.attempt.resolve()))
    if not args.execute:
        parser.error("live tests require --execute")
    return run_case(directory, args.case, args.retry_failed, args.credential_env)


if __name__ == "__main__":
    raise SystemExit(main())
