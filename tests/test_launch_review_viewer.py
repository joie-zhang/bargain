"""Validate saved launch-review records through a real loopback HTTP server."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ui.launch_review_viewer import ROOT, ReviewCatalog, make_handler


@pytest.fixture(scope="module")
def catalog():
    return ReviewCatalog()


@pytest.fixture(scope="module")
def server(catalog):
    service = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(catalog))
    worker = threading.Thread(target=service.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{service.server_port}"
    service.shutdown()
    service.server_close()
    worker.join(timeout=5)


def test_actual_review_counts(catalog):
    status = catalog.status()
    assert status["read_only"]
    assert not status["experiment_execution_enabled"]
    assert status["paper_runs"] == 7160
    assert status["families"] == 7
    assert status["agent_reports"] == 13
    assert len(catalog.documents) == 15
    assert len(catalog.sources) == 29
    assert not status["changed_inputs_since_load"]


def test_excerpts_are_exact_and_explicit(catalog):
    for source in catalog.sources.values():
        lines = (ROOT / source["file"]).read_text(encoding="utf-8").splitlines()
        assert source["text"] == "\n".join(lines[source["start"] - 1:source["end"]])
        assert source["start"] <= source["line"] <= source["end"]
    assert all("text" not in item for item in catalog.public_data()["sources"])


def test_no_missing_report_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReviewCatalog(tmp_path)


@pytest.mark.parametrize("endpoint", [
    "/", "/app.js", "/style.css", "/api/status", "/api/review",
    "/api/report?id=overview", "/api/report?id=implementation",
    "/api/source?id=runner", "/download?id=implementation",
])
def test_read_only_http_resources(server, endpoint):
    with urlopen(server + endpoint, timeout=10) as response:
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        body = response.read()
        assert body
        if endpoint.startswith("/api/"):
            json.loads(body)
        if endpoint.startswith("/download"):
            assert "implementation_proposal.md" in response.headers["Content-Disposition"]


@pytest.mark.parametrize("endpoint", [
    "/.env", "/review.json", "/api/source?id=../../.env",
    "/api/source?id=not-indexed", "/api/report?id=../AGENTS.md",
    "/download?id=../../.env", "/api/execute", "/../../AGENTS.md",
])
def test_unlisted_resources_are_unavailable(server, endpoint):
    with pytest.raises(HTTPError) as exc:
        urlopen(server + endpoint, timeout=10)
    assert exc.value.code == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_writes_are_rejected(server, method):
    request = Request(server + "/api/execute", method=method, data=b"{}")
    with pytest.raises(HTTPError) as exc:
        urlopen(request, timeout=10)
    assert exc.value.code == 405


def test_non_loopback_host_is_rejected(server):
    request = Request(server + "/api/review", headers={"Host": "untrusted.example"})
    with pytest.raises(HTTPError) as exc:
        urlopen(request, timeout=10)
    assert exc.value.code == 403
