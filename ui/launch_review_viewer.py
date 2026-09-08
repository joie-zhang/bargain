#!/usr/bin/env python3
"""Read-only browser for the launch-interface review and implementation proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "launch_review_assets"
DEFAULT_REVIEW = ROOT / "docs/analysis/public_launch_review_20260907"


class ReviewCatalog:
    def __init__(self, review_dir: Path = DEFAULT_REVIEW):
        self.review_dir = review_dir.resolve(strict=True)
        index = ASSETS / "review.json"
        self.data = json.loads(index.read_text(encoding="utf-8"))
        self.documents = {}
        index_stat = index.stat()
        self.fingerprints = {index: (index_stat.st_size, index_stat.st_mtime_ns)}
        for record in self.data["reports"]:
            path = (self.review_dir / record["filename"]).resolve(strict=True)
            if path.parent != self.review_dir or path.suffix != ".md":
                raise ValueError("Report must be a Markdown file directly inside the review directory")
            if record["id"] in self.documents:
                raise ValueError("Duplicate report ID")
            content = path.read_text(encoding="utf-8")
            self.documents[record["id"]] = {**record, "path": str(path), "text": content}
            stat = path.stat()
            self.fingerprints[path] = (stat.st_size, stat.st_mtime_ns)
        self.sources = {}
        for entry in self.data["sources"]:
            if entry["id"] in self.sources:
                raise ValueError("Duplicate source ID")
            path = (ROOT / entry["file"]).resolve(strict=True)
            if not path.is_relative_to(ROOT) or any(part.startswith(".") for part in path.relative_to(ROOT).parts):
                raise ValueError("Source is outside the explicit public source boundary")
            if path.suffix not in {".py", ".sh", ".md", ".tex", ".txt", ".ini"}:
                raise ValueError("Unsupported source type")
            if not path.is_file() or not isinstance(entry["line"], int) or entry["line"] < 1:
                raise ValueError("Invalid source reference")
            lines = path.read_text(encoding="utf-8").splitlines()
            if entry["line"] > len(lines):
                raise ValueError("Source line is beyond the end of the file")
            start = max(1, entry["line"] - 6)
            end = min(len(lines), entry["line"] + 38)
            self.sources[entry["id"]] = {
                **entry, "path": str(path), "start": start, "end": end,
                "text": "\n".join(lines[start - 1:end]),
            }
            stat = path.stat()
            self.fingerprints[path] = (stat.st_size, stat.st_mtime_ns)
        families = self.data["families"]
        if len(families) != 7 or len({item["id"] for item in families}) != 7:
            raise ValueError("The review must contain seven distinct experiment families")
        if sum(item["paper_runs"] for item in families) != 7160:
            raise ValueError("Paper run counts do not sum to 7,160")
        if len(self.documents) != 15:
            raise ValueError("Expected two summary documents and thirteen agent reports")
        for item in [*families, *self.data["findings"]]:
            if item["report"] not in self.documents:
                raise ValueError("Unknown report reference")
            if any(key not in self.sources for key in item["sources"]):
                raise ValueError("Unknown source reference")
        for stage in self.data["patch_stages"]:
            if any(key not in self.sources for key in stage["sources"]):
                raise ValueError("Unknown patch-stage source reference")
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "index": self.data,
            "reports": self.documents,
            "sources": self.sources,
        }
        self.digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()

    def status(self):
        changed = []
        for path, fingerprint in self.fingerprints.items():
            try:
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != fingerprint:
                    changed.append(str(path))
            except FileNotFoundError:
                changed.append(str(path))
        return {
            "ok": True, "read_only": True, "experiment_execution_enabled": False,
            "families": len(self.data["families"]), "agent_reports": len(self.documents) - 2,
            "paper_runs": sum(item["paper_runs"] for item in self.data["families"]),
            "loaded_at": self.loaded_at, "digest": self.digest,
            "changed_inputs_since_load": changed, "review_directory": str(self.review_dir),
        }

    def public_data(self):
        data = {**self.data, "status": self.status(), "root": str(ROOT)}
        data["reports"] = [{key: value for key, value in item.items() if key != "text"} for item in self.documents.values()]
        data["sources"] = [{key: value for key, value in item.items() if key != "text"} for item in self.sources.values()]
        return data


def make_handler(catalog: ReviewCatalog):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body: bytes, mime: str, status: int = 200, filename: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, data, status=200):
            self.respond(json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
            except ValueError:
                hostname = None
            if hostname not in {"127.0.0.1", "localhost"}:
                self.json_response({"error": "Loopback Host required"}, 403)
                return
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            key = params.get("id", [""])[0]
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if parsed.path in assets:
                name, mime = assets[parsed.path]
                self.respond((ASSETS / name).read_bytes(), mime)
            elif parsed.path == "/api/status":
                self.json_response(catalog.status())
            elif parsed.path == "/api/review":
                self.json_response(catalog.public_data())
            elif parsed.path == "/api/report" and key in catalog.documents:
                self.json_response(catalog.documents[key])
            elif parsed.path == "/api/source" and key in catalog.sources:
                self.json_response(catalog.sources[key])
            elif parsed.path == "/download" and key in catalog.documents:
                doc = catalog.documents[key]
                self.respond(doc["text"].encode(), "text/markdown; charset=utf-8", filename=doc["filename"])
            else:
                self.json_response({"error": "Unknown read-only resource"}, 404)

        def do_POST(self):
            self.json_response({"error": "This UI is read-only; experiments and file changes are disabled"}, 405)

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    catalog = ReviewCatalog(args.review_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    print(json.dumps({"url": f"http://{args.host}:{args.port}", **catalog.status()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
