import json
from urllib.parse import parse_qs

import pytest
from fastapi.testclient import TestClient

from app import number_lookup
from app.errors import ServiceError
from app.main import create_app
from test_release import release_dir


PHONE = "+447700900123"
PROVIDER = {"status": "success", "phone_valid": True, "e164": PHONE,
            "country": "United Kingdom", "country_code": "GB", "carrier": "Example", "phone_type": "mobile"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VERIPHONE_API_KEY", "test-placeholder")
    monkeypatch.setattr(number_lookup, "verify_number", lambda *args: dict(PROVIDER))
    with TestClient(create_app(tmp_path)) as instance:
        yield instance


def test_lookup_contract_without_model(client):
    response = client.post("/api/v1/number-lookup", json={"numbers": [PHONE, "80086"]})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"
    assert body["provider"] == "veriphone"
    assert body["results"][0] == {"input": PHONE, "status": "looked_up", "valid": True,
        "e164": PHONE, "country_name": "United Kingdom", "country_code": "GB", "carrier": "Example",
        "line_type": "mobile", "reason_code": None}
    assert body["results"][1]["reason_code"] == "SHORT_CODE"
    assert body["results"][1]["valid"] is False


@pytest.mark.parametrize("payload", [{"numbers": []}, {"numbers": [PHONE] * 6}, {"numbers": ["1" * 21]},
    {"numbers": [123]}, {"numbers": [""]}, {"numbers": PHONE}, {"numbers": [PHONE], "extra": True}])
def test_invalid_request(client, payload):
    response = client.post("/api/v1/number-lookup", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_no_key(client, monkeypatch):
    monkeypatch.delenv("VERIPHONE_API_KEY")
    response = client.post("/api/v1/number-lookup", json={"numbers": [PHONE]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "NUMBER_LOOKUP_UNAVAILABLE"


def test_local_normalization(client, monkeypatch):
    calls = []
    def verify(number, key):
        calls.append(number)
        return PROVIDER
    monkeypatch.setattr(number_lookup, "verify_number", verify)
    results = client.post("/api/v1/number-lookup", json={"numbers": ["0044 7700 900123", "07700900123", "abc", "911"]}).json()["results"]
    assert calls == [PHONE]
    assert [r["reason_code"] for r in results] == [None, "COUNTRY_CODE_REQUIRED", "NOT_A_NUMBER", "SHORT_CODE"]
    assert results[1]["valid"] is None


@pytest.mark.parametrize("provider", [{}, {**PROVIDER, "phone_valid": "true"}, {**PROVIDER, "e164": "invalid"},
    {**PROVIDER, "carrier": []}, {"status": "error", "message": "secret"}])
def test_bad_provider_response(client, monkeypatch, provider):
    monkeypatch.setattr(number_lookup, "verify_number", lambda *args: provider)
    response = client.post("/api/v1/number-lookup", json={"numbers": [PHONE]})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "NUMBER_LOOKUP_FAILED"


def test_mid_batch_failure_is_atomic_and_private(client, monkeypatch, caplog):
    calls = 0
    def verify(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ServiceError("NUMBER_LOOKUP_UNAVAILABLE", "Unavailable")
        return PROVIDER
    monkeypatch.setattr(number_lookup, "verify_number", verify)
    response = client.post("/api/v1/number-lookup", json={"numbers": [PHONE, "+14169670000"]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "NUMBER_LOOKUP_FAILED"
    assert "results" not in response.json()
    assert PHONE not in response.text + caplog.text
    assert "test-placeholder" not in response.text + caplog.text


@pytest.mark.parametrize("status", [200, 302, 401, 402, 429, 500])
def test_transport_body_only_no_redirect(monkeypatch, status):
    calls = []
    class Response:
        def read(self, limit):
            assert limit == 65537
            return json.dumps(PROVIDER).encode()
    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.veriphone.io" and timeout == 5
        def request(self, method, path, body, headers):
            calls.append((method, path, body, headers))
        def getresponse(self):
            response = Response()
            response.status = status
            return response
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(number_lookup, "HTTPSConnection", Connection)
    if status == 200:
        assert number_lookup.verify_number(PHONE, "placeholder") == PROVIDER
    else:
        with pytest.raises(ServiceError):
            number_lookup.verify_number(PHONE, "placeholder")
    method, path, body, headers = calls[0]
    assert method == "POST" and path == "/v3/verify"
    assert parse_qs(body) == {"phone": [PHONE], "mode": ["static"], "record": ["false"]}
    assert headers["Authorization"] == "Bearer placeholder"
    assert calls[1:] == ["closed"]


def test_manifest_positive_label(release_dir):
    with TestClient(create_app(release_dir)) as client:
        for model in ("primary", "naive_bayes", "logistic_regression"):
            for path in ("evaluation", "model"):
                response = client.get(f"/api/v1/{path}", params={"model_id": model})
                assert response.status_code == 200
                assert response.json()["positive_label"] == "spam"


def test_external_reference_http(client, monkeypatch):
    from app import routes
    monkeypatch.setattr(routes, "classify_external_reference", lambda text: {
        "provider": "huggingface", "model_id": "example/reference", "source": "external_reference",
        "scores": [{"label": "LABEL_0", "score": 0.8}]})
    response = client.post("/api/v1/external-reference", json={"text": "A test message"})
    assert response.status_code == 200
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert response.json()["source"] == "external_reference"
    assert response.json()["scores"] == [{"label": "LABEL_0", "score": 0.8}]
    assert client.post("/api/v1/external-reference", json={"text": " "}).status_code == 422


def test_external_reference_missing_config(client, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    response = client.post("/api/v1/external-reference", json={"text": "A test message"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EXTERNAL_REFERENCE_UNAVAILABLE"


def test_startup_loads_only_backend_env_without_overrides(monkeypatch, tmp_path):
    from app import main
    calls = []
    monkeypatch.setattr(main, "load_dotenv", lambda path, **kwargs: calls.append((path, kwargs)))
    with TestClient(create_app(tmp_path)):
        pass
    assert calls == [(main.Path(main.__file__).resolve().parents[1] / ".env",
                      {"override": False, "interpolate": False})]
