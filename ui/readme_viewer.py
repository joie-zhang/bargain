#!/usr/bin/env python3
"""Read-only, loopback-only viewer for the experiment launch guide."""

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "README2.md"
ASSETS = Path(__file__).resolve().parent / "readme_assets"


def make_handler(document=DOCUMENT):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, mime, status=200, download=False):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="README2.md"')
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, value, status=200):
            self.respond(json.dumps(value).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            except ValueError:
                host = None
            if host not in {"127.0.0.1", "localhost"}:
                self.json_response({"error": "Loopback Host required"}, 403)
                return
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8"),
                      "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            try:
                if path in assets:
                    name, mime = assets[path]
                    self.respond((ASSETS / name).read_bytes(), mime)
                elif path in {"/api/document", "/api/status", "/download"}:
                    # Read on each request, so Refresh never serves an old snapshot.
                    raw = document.read_bytes()
                    data = {"ok": True, "read_only": True, "path": str(document.resolve()),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "modified_at": document.stat().st_mtime}
                    if path == "/download":
                        self.respond(raw, "text/markdown; charset=utf-8", download=True)
                    else:
                        if path == "/api/document":
                            data["markdown"] = raw.decode("utf-8")
                        self.json_response(data)
                else:
                    self.json_response({"error": "Unknown read-only resource"}, 404)
            except (OSError, UnicodeError):
                self.json_response({"error": "The guide or a viewer asset could not be read"}, 503)

        def do_POST(self):
            self.json_response({"error": "Read-only viewer; command execution and file changes are disabled"}, 405)

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    DOCUMENT.read_text(encoding="utf-8")
    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"Read-only guide at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
