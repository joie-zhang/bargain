"""The public CLI keeps help, planning, and setup checks offline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from .launch.schema import FAMILIES, GAMES, LaunchRequest


def parser():
    root = argparse.ArgumentParser(prog="bargain", description="Run small, new negotiation experiments with explicit hosted API routes.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List frozen model aliases and provider routes (offline)")
    for command in ("plan", "doctor", "run"):
        sub = commands.add_parser(command, help={"plan": "Resolve inputs without API calls", "doctor": "Check local setup without API calls", "run": "Run one preset (makes paid API calls)"}[command])
        sub.add_argument("preset", choices=FAMILIES)
        sub.add_argument("--game", choices=GAMES, default="game1")
        sub.add_argument("--adversary")
        sub.add_argument("--model")
        sub.add_argument("--agents", type=int)
        sub.add_argument("--family", choices=("gpt5", "claude", "gemini"))
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--rounds", type=int, default=10)
        sub.add_argument("--discussion-turns", type=int, default=2)
        sub.add_argument("--discount", type=float, default=0.9)
        sub.add_argument("--position", choices=("first", "last"), default="first")
        sub.add_argument("--competition", type=float, help="Game 1 requested preference cosine (legacy parameter name)")
        for parameter in ("rho", "theta", "alpha", "sigma"):
            sub.add_argument("--" + parameter, type=float)
        sub.add_argument("--stratum", type=int, help="Heterogeneous Elo-SD stratum, 0..4 (default 2)")
        sub.add_argument("--max-tokens", type=int, help="Explicit per-phase output cap, up to each model's declared launch cap")
        sub.add_argument("--timeout", type=float, default=300.0)
        if command == "plan":
            sub.add_argument("--save-plan", type=Path, help="Save JSON to a new file instead of printing it")
        else:
            sub.add_argument("--env-file", type=Path, help="Private KEY=value file; no shell execution or implicit .env loading")
        if command == "run":
            sub.add_argument("--output", type=Path, help="New output directory (must not exist)")
    sub = commands.add_parser("execute", help="Execute an offline JSON plan (requires --execute)")
    sub.add_argument("plan", type=Path)
    sub.add_argument("--execute", action="store_true", required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--env-file", type=Path)
    sub = commands.add_parser("resume", help="Resume pending cells; incomplete attempts require explicit restart")
    sub.add_argument("output", type=Path)
    sub.add_argument("--env-file", type=Path)
    sub.add_argument("--retry-failed", action="store_true")
    sub.add_argument("--accept-unknown-outcome", action="store_true", help="Permit a new attempt despite an unknown remote outcome; may incur duplicate charges")
    for name in ("status", "summarize"):
        sub = commands.add_parser(name, help="Read and verify saved results without executing experiments")
        sub.add_argument("output", type=Path)
    return root


def default_output(preset):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / "bargain-runs" / f"{preset}-{timestamp}-{uuid.uuid4().hex[:8]}"


def main(argv=None):
    args = parser().parse_args(argv)
    from .launch.plan import build_plan, catalog, validate_plan
    from .launch.providers import doctor, load_credentials
    from .launch.store import read_json, status_report, current_attempt, COMPLETE

    try:
        if args.command == "models":
            for name, model in sorted(catalog()["models"].items()):
                print(f"{name}\t{model['api_type']}\t{model['model_id']}")
            return 0
        if args.command in ("plan", "doctor", "run"):
            fields = LaunchRequest.__dataclass_fields__
            request = LaunchRequest(**{key: value for key, value in vars(args).items() if key in fields})
            plan = build_plan(request)
            if args.command == "plan":
                payload = json.dumps(plan, indent=2, allow_nan=False) + "\n"
                if args.save_plan:
                    path = args.save_plan.expanduser().resolve()
                    with path.open("x", encoding="utf-8") as stream:
                        stream.write(payload)
                    print(path)
                else:
                    print(payload, end="")
                return 0
            credentials = load_credentials(args.env_file)
            if args.command == "doctor":
                report = doctor(plan, credentials)
                print(json.dumps(report, indent=2))
                return 0 if report["ready_offline"] else 1
            from .launch.execute import execute
            return execute(plan, args.output or default_output(args.preset), credentials)
        if args.command == "execute":
            from .launch.execute import execute
            return execute(validate_plan(read_json(args.plan.expanduser().resolve())), args.output, load_credentials(args.env_file))
        output = args.output.expanduser().resolve()
        plan = validate_plan(read_json(output / "plan.json"))
        if args.command == "resume":
            from .launch.execute import execute
            if args.accept_unknown_outcome and not args.retry_failed:
                raise ValueError("--accept-unknown-outcome requires --retry-failed")
            return execute(plan, output, load_credentials(args.env_file), resume=True,
                           retry_failed=args.retry_failed, accept_unknown=args.accept_unknown_outcome)
        report = status_report(output, plan)
        if args.command == "summarize":
            for row, run in zip(report["runs"], plan["runs"]):
                if row["state"] in COMPLETE:
                    result = read_json(current_attempt(output, run) / "result.json")
                    row.update(final_utilities=result["final_utilities"], final_round=result["final_round"],
                               effort=run["effort"], models={s["agent_id"]: s["name"] for s in run["seats"]})
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] == report["requested"] else 1
    except (ValueError, TypeError, OSError, ImportError) as exc:
        print(f"bargain: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("bargain: interrupted; inspect status before restarting", file=sys.stderr)
        return 130
