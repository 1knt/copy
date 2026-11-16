import json
from unittest.mock import Mock

import pytest

from app import external_reference
from app.errors import ServiceError


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_private_token")
    monkeypatch.setenv("HF_MODEL_ID", "example/reference-model")
    connection = Mock()
    response = connection.getresponse.return_value
    response.status = 200
    response.read.return_value = b'[{"label":"LABEL_1","score":0.8},{"label":"LABEL_0","score":0.2}]'
    factory = Mock(return_value=connection)
    monkeypatch.setattr(external_reference, "HTTPSConnection", factory)
    return factory, connection, response


def assert_unavailable():
    with pytest.raises(ServiceError) as caught:
        external_reference.classify_external_reference("Private message")
    assert caught.value.code == "EXTERNAL_REFERENCE_UNAVAILABLE"
    assert caught.value.status_code == 503
    assert caught.value.message == "The external reference is unavailable."


@pytest.mark.parametrize("name,value", [
    ("HF_TOKEN", ""), ("HF_MODEL_ID", ""),
    ("HF_MODEL_ID", "https://evil.example/model"),
    ("HF_MODEL_ID", "example/../secret"),
    ("HF_MODEL_ID", "example/model?token=secret"),
])
def test_missing_or_invalid_configuration_never_connects(monkeypatch, transport, name, value):
    monkeypatch.setenv(name, value)
    assert_unavailable()
    transport[0].assert_not_called()


def test_sends_only_text_to_fixed_router_and_preserves_neutral_scores(transport):
    factory, connection, response = transport
    result = external_reference.classify_external_reference("Private message")
    factory.assert_called_once_with("router.huggingface.co", timeout=10)
    args, kwargs = connection.request.call_args
    assert args == ("POST", "/hf-inference/models/example/reference-model")
    assert json.loads(kwargs["body"]) == {"inputs": "Private message"}
    assert kwargs["headers"]["Authorization"] == "Bearer hf_private_token"
    response.read.assert_called_once_with(65537)
    connection.close.assert_called_once()
    assert result == {
        "provider": "huggingface", "model_id": "example/reference-model", "source": "external_reference",
        "scores": [{"label": "LABEL_1", "score": 0.8}, {"label": "LABEL_0", "score": 0.2}],
    }


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 429, 500, 503])
def test_provider_errors_and_redirects_are_not_followed(transport, status):
    _, connection, response = transport
    response.status = status
    assert_unavailable()
    connection.request.assert_called_once()
    response.read.assert_not_called()
    connection.close.assert_called_once()


@pytest.mark.parametrize("payload", [
    {}, [], [[{}, {}], [{}]], [{"label": "", "score": 0.4}],
    [{"label": "X", "score": True}], [{"label": "X", "score": -0.1}],
    [{"label": "X", "score": 1.1}], [{"label": "X", "score": float("nan")}],
    [{"label": "X", "score": float("inf")}], [{"label": "X", "score": "0.5"}],
    [{"label": "X", "score": 0.5, "text": "Private message"}],
    [{"label": "X", "score": 0.5}, {"label": "X", "score": 0.5}],
])
def test_invalid_response_shapes_fail_closed(transport, payload):
    transport[2].read.return_value = json.dumps(payload).encode()
    assert_unavailable()


def test_single_batch_response_is_supported(transport):
    transport[2].read.return_value = b'[[{"label":"neutral","score":0.4}]]'
    assert external_reference.classify_external_reference("message")["scores"] == [{"label": "neutral", "score": 0.4}]


@pytest.mark.parametrize("body", [b"not json", b"x" * 65537], ids=["invalid-json", "oversized"])
def test_invalid_or_oversized_response_is_sanitized(transport, body):
    transport[2].read.return_value = body
    assert_unavailable()


def test_timeout_is_sanitized_and_not_logged(transport, caplog, capsys):
    transport[1].getresponse.side_effect = TimeoutError("hf_private_token Private message")
    assert_unavailable()
    transport[1].close.assert_called_once()
    assert not caplog.text
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("text", ["", "  ", "x" * 2001, None], ids=["empty", "blank", "long", "null"])
def test_invalid_text_never_connects(transport, text):
    with pytest.raises(ServiceError) as caught:
        external_reference.classify_external_reference(text)
    assert caught.value.code == "INVALID_TEXT"
    transport[0].assert_not_called()
