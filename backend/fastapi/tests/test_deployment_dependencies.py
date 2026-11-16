import tomllib
from pathlib import Path


def test_vercel_dependencies_match_runtime_requirements():
    backend = Path(__file__).resolve().parents[1]
    project = tomllib.loads((backend / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        line.strip()
        for line in (backend / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert set(project["project"]["dependencies"]) == requirements
