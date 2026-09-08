import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from ui.readme_viewer import DOCUMENT, make_handler


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(server, path, method="GET", headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_document_and_download_are_exact(server):
    status, headers, body = request(server, "/api/document")
    assert status == 200
    assert json.loads(body)["markdown"] == DOCUMENT.read_text()
    assert json.loads(body)["read_only"] is True
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    status, headers, body = request(server, "/download")
    assert status == 200
    assert body == DOCUMENT.read_bytes()
    assert headers["Content-Disposition"] == 'attachment; filename="README2.md"'


@pytest.mark.parametrize("path", ["/", "/app.js", "/style.css", "/api/status", "/favicon.svg"])
def test_assets_and_health(server, path):
    status, _, body = request(server, path)
    assert status == 200
    assert body


@pytest.mark.parametrize("path", ["/.env", "/README.md", "/../.env", "/%2e%2e/.env", "/api/execute"])
def test_no_arbitrary_files_or_execution_endpoint(server, path):
    assert request(server, path)[0] == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_read_only(server, method):
    assert request(server, "/api/document", method)[0] == 405


def test_host_boundary(server):
    assert request(server, "/", headers={"Host": "external.invalid"})[0] == 403


def test_missing_document_is_explicit():
    class Missing:
        def read_bytes(self):
            raise FileNotFoundError

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Missing()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        assert request(httpd, "/api/document")[0] == 503
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
