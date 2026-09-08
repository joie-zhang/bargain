"""Checks against the real audit records and a real loopback HTTP server."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from ui.paper_code_audit_viewer import AuditCatalog, make_handler, short_id


@pytest.fixture(scope="module")
def catalog():
    return AuditCatalog()


@pytest.fixture(scope="module")
def server(catalog):
    service = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(catalog))
    worker = threading.Thread(target=service.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{service.server_port}"
    service.shutdown()
    service.server_close()
    worker.join(timeout=5)


def test_all_reports_resolve_including_order_audit(catalog):
    assert catalog.summary()["agent_audits"] == 45
    assert catalog.summary()["root_supplements"] == 2
    assert catalog.audits["e34"]["task_id"] == "e34_order"
    assert catalog.review["passed"], catalog.review
    assert short_id("root_import_closure") == "root_import_closure"


def test_live_consistency_counts(catalog):
    summary = catalog.summary()
    assert summary["needed_files"] == 12078
    assert summary["code_files"] == 1204
    assert summary["status_counts"]["unresolved-do-not-delete"] == 664
    assert summary["source_candidates"] == 9
    assert summary["cache_candidates"] == 25
    assert not summary["changed_inputs_since_load"]


def test_search_filters_and_pagination(catalog):
    rows = catalog.rows("code", "utils", "conditional-retirement-candidate")
    assert rows["total"] == 7
    assert all("/utils/" in r["path"] for r in rows["rows"])
    first = catalog.rows("needed", audit="e34", limit=3)
    following = catalog.rows("needed", audit="e34", offset=3, limit=3)
    assert first["total"] == 27
    assert len(first["rows"]) == 3
    assert {r["path"] for r in first["rows"]}.isdisjoint(r["path"] for r in following["rows"])


def test_exact_evidence_and_unresolved_are_distinct(catalog):
    path = next(r["path"] for r in catalog.code if r["status"] == "needed")
    assert catalog.detail(path)["evidence"]
    path = next(r["path"] for r in catalog.code if r["status"] == "unresolved-do-not-delete")
    assert catalog.detail(path)["code"]["status"] == "unresolved-do-not-delete"
    assert catalog.detail(path)["candidate"] is None


@pytest.mark.parametrize("args", [("unknown",), ("code", "", "", "", -1), ("code", "", "", "", 0, 1000)])
def test_invalid_queries_fail(catalog, args):
    with pytest.raises(ValueError):
        catalog.rows(*args)


def test_missing_audit_does_not_load_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        AuditCatalog(tmp_path)


@pytest.mark.parametrize("endpoint", ["/", "/app.js", "/api/status", "/api/review", "/api/audits", "/api/audit?id=e34", "/api/files?dataset=code", "/api/candidates", "/api/coverage", "/api/selections", "/api/report", "/download?name=needed_files.csv"])
def test_real_http_endpoints(server, endpoint):
    with urlopen(server + endpoint, timeout=20) as response:
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.read()
        assert body
        if endpoint.startswith("/api/"):
            json.loads(body)


def test_file_endpoint_only_returns_evidence(server, catalog):
    path = catalog.needed[0]["path"]
    with urlopen(server + "/api/file?" + urlencode({"path": path}), timeout=20) as response:
        data = json.load(response)
        assert data["path"] == path
        assert "contents" not in data


@pytest.mark.parametrize("endpoint", ["/download?name=../../AGENTS.md", "/download?name=.env", "/api/report?name=e34_order.json", "/api/file?path=/etc/passwd", "/etc/passwd"])
def test_unindexed_files_are_not_served(server, endpoint):
    with pytest.raises(HTTPError) as error:
        urlopen(server + endpoint, timeout=20)
    assert error.value.code == 404


def test_cross_host_is_rejected(server):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(server + "/api/status", headers={"Host": "outside.example"}), timeout=20)
    assert error.value.code == 403


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_no_write_endpoints(server, method):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(server + "/api/files", data=b"{}", method=method), timeout=20)
    assert error.value.code == 501
