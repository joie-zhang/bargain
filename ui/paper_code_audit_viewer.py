#!/usr/bin/env python3
"""Read-only audit browser. Serves only indexed audit reports, never source files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import importlib.util
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "docs/analysis/paper_code_audit_20260907"
ASSETS = Path(__file__).resolve().parent
STATUSES = {"needed", "protected-other-git-worktree", "conditional-retirement-candidate", "unresolved-do-not-delete"}


def short_id(value: str) -> str:
    match = re.match(r"^(e\d{2})(?:_|$)", value)
    return match[1] if match else value


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AuditCatalog:
    def __init__(self, directory: Path = DEFAULT_AUDIT):
        self.directory = directory.resolve()
        self.index = read_csv(self.directory / "audit_index.csv")
        self.code = read_csv(self.directory / "code_status.csv")
        self.needed = read_csv(self.directory / "needed_files.csv")
        self.coverage = read_csv(self.directory / "paper_coverage.csv")
        self.candidates = json.loads((self.directory / "cleanup_candidates.json").read_text())
        self.selections = json.loads((self.directory / "directory_selections.json").read_text())
        self.previous = json.loads((self.directory / "validation.json").read_text())
        self.inventory = json.loads((self.directory / "e43_code_filename_inventory.json").read_text())
        self.audits = {}
        self.evidence = defaultdict(list)
        self.documents = {}
        for row in self.index:
            report = Path(row["report_path"]).resolve()
            dependency = Path(row["dependency_report_path"]).resolve()
            for path in (report, dependency):
                if path.parent != self.directory:
                    raise ValueError(f"Report path escapes the audit directory: {path}")
            data = json.loads(dependency.read_text())
            key = short_id(data["task_id"])
            if key in self.audits:
                raise ValueError(f"Duplicate audit ID: {key}")
            self.audits[key] = {**data, "id": key, "report": report.read_text(), "report_name": report.name}
            self.documents[report.name] = report.read_bytes()
            self.documents[dependency.name] = dependency.read_bytes()
            for entry in data["needed"]:
                self.evidence[entry["path"]].append({"audit": key, **entry})
        for name in ("report.md", "needed_files.csv", "code_status.csv", "audit_index.csv", "paper_coverage.csv", "cleanup_candidates.json", "directory_selections.json", "unresolved_questions.json", "validation.json", "e43_code_filename_inventory.json"):
            self.documents[name] = (self.directory / name).read_bytes()
        self.code_by_path = {row["path"]: row for row in self.code}
        self.needed_by_path = {row["path"]: row for row in self.needed}
        self.candidate_by_path = {row["path"]: row for row in self.candidates}
        self.review = self.validate()
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        self.fingerprints = {}
        for name in sorted(self.documents):
            path = self.directory / name
            stat = path.stat()
            self.fingerprints[name] = (stat.st_size, stat.st_mtime_ns)
        self.digest = hashlib.sha256(b"".join(name.encode() + self.documents[name] for name in sorted(self.documents))).hexdigest()

    def validate(self) -> dict:
        checks = []

        def check(name: str, issues: list, scope: str):
            checks.append({"name": name, "passed": not issues, "issues": issues, "scope": scope})

        agents = {key for key in self.audits if re.fullmatch(r"e\d{2}", key)}
        check("All 45 audit assignments have reports", sorted(agents ^ {f"e{i:02}" for i in range(1, 46)}), "Assignment coverage, not proof that every source line was reviewed.")
        check("Keep paths are unique", [p for p, n in Counter(r["path"] for r in self.needed).items() if n != 1], "Exact path strings in the consolidated CSV.")
        concrete = {p for p in self.evidence if Path(p).is_file()}
        check("Keep list matches the detailed reports", sorted(concrete ^ set(self.needed_by_path)), "Union of concrete files; directory selections remain separate.")
        check("All reported dependency paths exist", [p for p in self.evidence if not Path(p).exists()], "File existence does not prove historical byte equality or scientific validity.")
        check("Every code path has one valid status", [r for r in self.code if r["status"] not in STATUSES] + [p for p, n in Counter(r["path"] for r in self.code).items() if n != 1], "No absent status is treated as unused.")
        check("Code inventory matches status rows", sorted(set(self.inventory["paths"]) ^ set(self.code_by_path)), "The original 1,204-file inventory is a dated, extension-limited snapshot.")
        check("Needed code rows have positive evidence", [r["path"] for r in self.code if r["status"] == "needed" and r["path"] not in concrete], "Evidence can describe runtime, reproduction, provenance or support.")
        conflicts = []
        resolved_keep = {str(Path(p).resolve()) for p in self.needed_by_path}
        resolved_directories = {Path(s["path"]).resolve() for s in self.selections}
        for candidate in self.candidates:
            path = Path(candidate["path"]).resolve()
            if str(path) in resolved_keep:
                conflicts.append(candidate["path"])
            if any(p in path.parents for p in resolved_directories):
                conflicts.append(candidate["path"])
        check("Candidates do not overlap keep entries", sorted(set(conflicts)), "Checks resolved paths and ancestor directory selections; this does not approve deletion.")
        check("Candidate paths still exist", [r["path"] for r in self.candidates if not Path(r["path"]).is_file()], "Candidates are suggestions, never executable actions.")
        cache_issues = []
        for row in self.candidates:
            if row["confidence"] != "safe-generated":
                continue
            path = Path(row["path"])
            match = re.fullmatch(r"(.+)\.cpython-\d+(?:-pytest-[\d.]+)?\.pyc", path.name)
            if not match or not path.is_file() or path.stat().st_size < 16:
                cache_issues.append(str(path))
                continue
            source = path.parent.parent / (match[1] + ".py")
            magic, mode, mtime, size = struct.unpack("<4sIII", path.read_bytes()[:16])
            if not source.is_file() or magic != importlib.util.MAGIC_NUMBER or mode != 0 or mtime != (int(source.stat().st_mtime) & 0xffffffff) or size != (source.stat().st_size & 0xffffffff):
                cache_issues.append(str(path))
        check("Bytecode candidates have matching source counterparts", cache_issues, "Checks CPython/pytest headers, not a mathematical proof of semantic identity.")
        coverage_issues = []
        for row in self.coverage:
            for audit in row["audit_ids"].split(";"):
                if audit not in self.audits:
                    coverage_issues.append(f"Unknown audit {audit}")
            for path in [row["source_path"], *filter(None, row["graphic_inputs"].split(";"))]:
                if not Path(path).is_file():
                    coverage_issues.append(path)
        check("Paper mappings resolve to reports and assets", coverage_issues, "Checks mapped sources and assets, not a fresh PDF compilation.")
        figures = sum(r["kind"] == "figure" for r in self.coverage)
        tables = sum(r["kind"] == "table" for r in self.coverage)
        check("Coverage contains 30 figures and 10 tables", [] if (figures, tables) == (30, 10) else [(figures, tables)], "Counts refer to the audited paper snapshot.")
        observed_labels = []
        for source in dict.fromkeys(row["source_path"] for row in self.coverage):
            text = re.sub(r"(?<!\\)%[^\n]*", "", Path(source).read_text())
            for match in re.finditer(r"\\begin\{(figure\*?|wrapfigure|table\*?)\}(.*?)\\end\{\1\}", text, re.S):
                observed_labels.append(";".join(re.findall(r"\\label\{([^}]+)\}", match[2])))
        check("Mapped labels match the current source floats", [] if Counter(observed_labels) == Counter(r["labels"] for r in self.coverage) else [{"observed": observed_labels}], "Detects added, removed or relabelled figures and tables in the mapped source files.")
        status_counts = dict(Counter(r["status"] for r in self.code))
        expected = self.previous["code_status_counts"]
        check("Headline counts agree with the saved validation", [] if status_counts == expected and len(self.needed) == self.previous["needed_files"] else [{"actual": status_counts, "saved": expected}], "Consistency check only; it is not a completeness certificate.")
        refs = {key for row in self.needed for key in row["audit_ids"].split(";")}
        check("Keep-list audit references resolve", sorted(refs - self.audits.keys()), "Each keep entry leads to a detailed report.")
        return {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": checks, "passed": all(c["passed"] for c in checks), "limits": ["664 inventoried code paths remain unresolved.", "43 directory paths use subset-selection rules, not complete per-file manifests.", "Historical code versions and some dynamic dependencies remain unresolved.", "This review checks the audit records; it does not rerun all experiments or certify every scientific claim.", "New viewer and test files are outside the original September 7 code inventory."]}

    def summary(self) -> dict:
        changed = []
        for name, expected in self.fingerprints.items():
            path = self.directory / name
            if not path.exists() or (path.stat().st_size, path.stat().st_mtime_ns) != expected:
                changed.append(name)
        return {"read_only": True, "loaded_at": self.loaded_at, "input_digest": self.digest, "changed_inputs_since_load": changed, "audit_directory": str(self.directory), "agent_audits": sum(bool(re.fullmatch(r"e\d{2}", k)) for k in self.audits), "root_supplements": sum(not bool(re.fullmatch(r"e\d{2}", k)) for k in self.audits), "needed_files": len(self.needed), "code_files": len(self.code), "status_counts": dict(Counter(r["status"] for r in self.code)), "directory_paths": len({r["path"] for r in self.selections}), "source_candidates": sum(r["confidence"] == "candidate-only" for r in self.candidates), "cache_candidates": sum(r["confidence"] == "safe-generated" for r in self.candidates), "review_passed": self.review["passed"], "figures": sum(r["kind"] == "figure" for r in self.coverage), "tables": sum(r["kind"] == "table" for r in self.coverage)}

    def rows(self, dataset: str, query: str = "", status: str = "", audit: str = "", offset: int = 0, limit: int = 100) -> dict:
        if dataset not in {"code", "needed"} or offset < 0 or not 1 <= limit <= 250:
            raise ValueError("Invalid dataset or pagination")
        source = self.code if dataset == "code" else self.needed
        terms = query.casefold().split()
        rows = [r for r in source if all(t in r["path"].casefold() for t in terms) and (not status or r.get("status") == status) and (not audit or audit in r["audit_ids"].split(";"))]
        return {"total": len(rows), "offset": offset, "limit": limit, "rows": rows[offset:offset + limit]}

    def detail(self, path: str) -> dict:
        if path not in self.needed_by_path and path not in self.code_by_path and path not in self.candidate_by_path:
            raise KeyError("Path is not in the audit inventory")
        return {"path": path, "code": self.code_by_path.get(path), "keep": self.needed_by_path.get(path), "candidate": self.candidate_by_path.get(path), "evidence": self.evidence.get(path, []), "exists_now": Path(path).exists()}


def make_handler(catalog: AuditCatalog):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body: bytes, mime: str, status: int = 200, filename: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host", "").split(":")[0] not in {"127.0.0.1", "localhost"}:
                self.respond(b"Loopback Host required", "text/plain", 403)
                return
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            arg = lambda key, default="": params.get(key, [default])[0]
            try:
                if parsed.path in {"/", "/app.js"}:
                    name, mime = ("paper_code_audit_viewer.html", "text/html; charset=utf-8") if parsed.path == "/" else ("paper_code_audit_viewer.js", "text/javascript; charset=utf-8")
                    self.respond((ASSETS / name).read_bytes(), mime)
                    return
                if parsed.path == "/download":
                    name = arg("name")
                    if name not in catalog.documents:
                        raise KeyError("Unknown report")
                    self.respond(catalog.documents[name], "application/octet-stream", filename=name)
                    return
                if parsed.path == "/api/status":
                    data = catalog.summary()
                elif parsed.path == "/api/review":
                    data = catalog.review
                elif parsed.path == "/api/audits":
                    data = [{"id": k, "result": d["result"], "summary": d["summary"], "needed_entries": len(d["needed"]), "open_questions": len(d["unresolved"])} for k, d in catalog.audits.items()]
                elif parsed.path == "/api/audit":
                    data = catalog.audits[arg("id")]
                elif parsed.path == "/api/files":
                    data = catalog.rows(arg("dataset", "code"), arg("q"), arg("status"), arg("audit"), int(arg("offset", "0")), int(arg("limit", "100")))
                elif parsed.path == "/api/file":
                    data = catalog.detail(arg("path"))
                elif parsed.path == "/api/candidates":
                    data = catalog.candidates
                elif parsed.path == "/api/coverage":
                    data = catalog.coverage
                elif parsed.path == "/api/selections":
                    data = catalog.selections
                elif parsed.path == "/api/report":
                    name = arg("name", "report.md")
                    if name not in catalog.documents or not name.endswith(".md"):
                        raise KeyError("Unknown report")
                    data = {"name": name, "text": catalog.documents[name].decode()}
                else:
                    raise KeyError("Unknown endpoint")
                self.respond(json.dumps(data).encode(), "application/json; charset=utf-8")
            except (KeyError, ValueError) as error:
                self.respond(json.dumps({"error": str(error)}).encode(), "application/json", 400 if isinstance(error, ValueError) else 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    catalog = AuditCatalog(args.audit_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    print(json.dumps({"url": f"http://{args.host}:{args.port}", **catalog.summary()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
