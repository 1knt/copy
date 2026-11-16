import argparse
import zipfile
from pathlib import Path

from ai.data import SEED


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATTERNS = ("ai/*.py", "ai/tests/*.py", "ai/README.md", "ai/environment.yml", "ai/requirements.txt", "fastapi/requirements.txt", "fastapi/requirements-dev.txt")


def collect(release, prepared, reports):
    files = []
    for pattern in SOURCE_PATTERNS:
        files.extend(sorted(ROOT.glob(pattern)))
    for directory in (prepared, release, reports):
        files.extend(sorted(path for path in Path(directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts))
    return files


def build(output, release, prepared, reports):
    output = Path(output)
    if output.exists():
        raise ValueError("The submission archive already exists; choose a new name.")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect(release, prepared, reports)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "ai" / "README.md", "README.md")
        for path in files:
            archive.write(path, (Path("backend") / path.resolve().relative_to(ROOT)).as_posix())
    return output, len(files)


def main():
    parser = argparse.ArgumentParser(description="Package the Assignment 2 source code, processed datasets, final models and reports into one ZIP file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=ROOT / "fastapi" / "artifacts" / "current")
    parser.add_argument("--prepared", type=Path, default=ROOT / "ai" / "data" / "processed" / f"combined-sms-seed-{SEED}")
    parser.add_argument("--reports", type=Path, default=ROOT / "ai" / "reports")
    arguments = parser.parse_args()
    output, count = build(arguments.output, arguments.release, arguments.prepared, arguments.reports)
    print(f"Wrote {count} files to {output}")


if __name__ == "__main__":
    main()
