import pytest


@pytest.fixture(autouse=True)
def isolate_provider_configuration(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in ("VERIPHONE_API_KEY", "HF_TOKEN", "HF_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
