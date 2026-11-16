import asyncio
import json
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.middleware import BODY_LIMIT
from app.schemas import TextRequest


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(artifact_dir=tmp_path), raise_server_exceptions=False) as instance:
        yield instance


def assert_error(response, status, code):
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == code
    assert str(UUID(error["request_id"])) == error["request_id"]
    assert response.headers["x-request-id"] == error["request_id"]
    assert response.headers["cache-control"] == "no-store"


def test_health_distinguishes_process_from_model(client):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "spam-screening-api"}
    assert_error(client.get("/api/v1/health/ready"), 503, "MODEL_UNAVAILABLE")


@pytest.mark.parametrize("endpoint,code", [
    ("model", "MODEL_UNAVAILABLE"),
    ("evaluation", "EVALUATION_UNAVAILABLE"),
    ("dataset-summary", "DATA_UNAVAILABLE"),
    ("clusters", "CLUSTERS_UNAVAILABLE"),
])
def test_absent_artifacts_are_not_fabricated(client, endpoint, code):
    response = client.get(f"/api/v1/{endpoint}")
    assert_error(response, 503, code)
    assert set(response.json()) == {"error"}


@pytest.mark.parametrize("endpoint,code", [
    ("predictions", "MODEL_UNAVAILABLE"), ("comparisons", "COMPARISON_UNAVAILABLE")
])
def test_valid_text_reaches_explicit_unavailable_state(client, endpoint, code):
    response = client.post(f"/api/v1/{endpoint}", json={"text": "Private text"})
    assert_error(response, 503, code)
    assert "Private text" not in response.text


@pytest.mark.parametrize("payload,code", [
    ({}, "INVALID_TEXT"),
    ({"text": ""}, "INVALID_TEXT"),
    ({"text": " \n\t "}, "INVALID_TEXT"),
    ({"text": None}, "INVALID_TEXT"),
    ({"text": 123}, "INVALID_TEXT"),
    ({"text": True}, "INVALID_TEXT"),
    ({"text": ["Private text"]}, "INVALID_TEXT"),
    ({"text": {"private": "secret"}}, "INVALID_TEXT"),
    ({"text": "a" * 2001}, "INVALID_TEXT"),
    ({"text": "Private text", "threshold": 0.1}, "INVALID_REQUEST"),
    (["Private text"], "INVALID_REQUEST"),
])
@pytest.mark.parametrize("endpoint", ["predictions", "comparisons"])
def test_strict_schema_and_private_validation(client, payload, code, endpoint):
    response = client.post(f"/api/v1/{endpoint}", json=payload)
    assert_error(response, 422, code)
    assert "Private text" not in response.text
    assert "secret" not in response.text
    assert "input" not in response.json()["error"]


@pytest.mark.parametrize("text", ["x", "a" * 2000, "😀" * 2000, "  valid  "], ids=["single", "maximum", "unicode", "trimmed"])
def test_valid_unicode_boundaries(client, text):
    response = client.post("/api/v1/predictions", json={"text": text})
    assert_error(response, 503, "MODEL_UNAVAILABLE")


def test_trimming_happens_before_length_validation():
    assert TextRequest(text=" \t" + "😀" * 2000 + "\n ").text == "😀" * 2000


def test_unicode_overflow_is_rejected(client):
    assert_error(client.post("/api/v1/predictions", json={"text": "😀" * 2001}), 422, "INVALID_TEXT")


def test_json_escaping_counts_towards_transport_limit(client):
    body = json.dumps({"text": "😀" * 2000}, ensure_ascii=True)
    assert_error(client.post("/api/v1/predictions", content=body,
                             headers={"Content-Type": "application/json"}), 413, "BODY_TOO_LARGE")


@pytest.mark.parametrize("content_type", [None, "text/plain", "multipart/form-data"])
def test_non_json_is_rejected(client, content_type):
    headers = {"Content-Type": content_type} if content_type else {}
    assert_error(client.post("/api/v1/predictions", content='{"text":"private"}', headers=headers),
                 415, "UNSUPPORTED_MEDIA_TYPE")


def test_json_charset_parameter_is_supported(client):
    response = client.post("/api/v1/predictions", content='{"text":"valid"}',
                           headers={"Content-Type": "application/json; charset=utf-8"})
    assert_error(response, 503, "MODEL_UNAVAILABLE")


def test_malformed_json_does_not_echo_input(client):
    response = client.post("/api/v1/predictions", content='{"text":"Private text',
                           headers={"Content-Type": "application/json"})
    assert_error(response, 422, "INVALID_REQUEST")
    assert "Private text" not in response.text


@pytest.mark.parametrize("path", [
    "model?model_id=../../secret", "evaluation?model_id=other", "dataset-summary?source=secret",
    "dataset-summary?split=other", "dataset-summary?actual_class=other", "clusters?cluster_id=-1",
])
def test_artifact_filters_are_allowlisted(client, path):
    assert_error(client.get(f"/api/v1/{path}"), 422, "INVALID_REQUEST")


def test_request_ids_are_generated_and_unknown_api_is_not_spa(client):
    first = client.get("/api/v1/not-found", headers={"X-Request-ID": "untrusted"})
    second = client.get("/api/v1/not-found")
    assert_error(first, 404, "NOT_FOUND")
    assert first.headers["x-request-id"] != "untrusted"
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


def test_method_not_allowed_is_sanitized(client):
    assert_error(client.get("/api/v1/predictions"), 405, "METHOD_NOT_ALLOWED")


def test_docs_describe_current_contract(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "503" in response.json()["paths"]["/api/v1/predictions"]["post"]["responses"]
    assert client.get("/docs").status_code == 200


async def invoke_asgi(chunks, headers=()):
    app = create_app()
    sent = []
    events = iter([
        {"type": "http.request", "body": body, "more_body": index < len(chunks) - 1}
        for index, body in enumerate(chunks)
    ])

    async def receive():
        return next(events, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    await app({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/v1/predictions",
        "raw_path": b"/api/v1/predictions", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), *headers],
        "client": ("127.0.0.1", 1), "server": ("localhost", 8000),
    }, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent)
    return status, json.loads(body)


def test_chunked_limit_without_content_length():
    status, body = asyncio.run(invoke_asgi([b"x" * 8192, b"x" * 8193]))
    assert status == 413
    assert body["error"]["code"] == "BODY_TOO_LARGE"


def test_understated_content_length_cannot_bypass_limit():
    status, body = asyncio.run(invoke_asgi([b"x" * (BODY_LIMIT + 1)], [(b"content-length", b"1")]))
    assert status == 413
    assert body["error"]["code"] == "BODY_TOO_LARGE"


@pytest.mark.parametrize("length", [b"-1", b"abc", b"1,1", b"", b"1.0", b" 1"])
def test_malformed_lengths_fail_safely(length):
    status, body = asyncio.run(invoke_asgi([b"{}"], [(b"content-length", length)]))
    assert status == 422
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_extreme_length_does_not_crash_integer_parser():
    status, body = asyncio.run(invoke_asgi([b"{}"], [(b"content-length", b"9" * 5000)]))
    assert status == 413
    assert body["error"]["code"] == "BODY_TOO_LARGE"


def test_duplicate_content_length_is_rejected():
    status, body = asyncio.run(invoke_asgi([b"{}"], [(b"content-length", b"2"), (b"content-length", b"2")]))
    assert status == 422
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_content_length_mismatch_is_rejected():
    status, body = asyncio.run(invoke_asgi([b"{}"], [(b"content-length", b"1")]))
    assert status == 422
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_exact_transport_limit_is_accepted():
    payload = b'{"text":"valid"}'
    payload += b" " * (BODY_LIMIT - len(payload))
    status, body = asyncio.run(invoke_asgi([payload[:8192], payload[8192:]]))
    assert status == 503
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"


def test_packaged_release_serves_both_sources():
    from pathlib import Path
    artifacts = Path(__file__).resolve().parents[1] / "artifacts" / "current"
    with TestClient(create_app(artifact_dir=artifacts), raise_server_exceptions=False) as packaged:
        assert packaged.get("/api/v1/health/ready").status_code == 200
        overall = packaged.get("/api/v1/dataset-summary").json()
        assert set(overall["source_counts"]) == {"uci_sms", "mendeley_smishing"}
        assert sum(overall["source_counts"].values()) == overall["sample_count"]
        added = packaged.get("/api/v1/dataset-summary?source=mendeley_smishing").json()
        assert added["source_counts"] == {"mendeley_smishing": added["sample_count"]} and added["sample_count"] > 0
        assert packaged.get("/api/v1/clusters").json()["source_id"] == "combined"
        result = packaged.post("/api/v1/predictions", json={"text": "URGENT: your parcel is held, pay the customs fee at bit.ly/xyz"}).json()
        assert result["label"] == "spam"
