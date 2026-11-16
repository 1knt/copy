# Backend

The backend has two parts that share one Python environment:

| Folder | Assignment | Role |
|---|---|---|
| [`ai/`](ai/README.md) | Assignment 2 | Offline pipeline: downloads and merges the datasets, analyses them, trains and evaluates the models, and writes a frozen release |
| [`fastapi/`](fastapi/README.md) | Assignment 3 | FastAPI service that loads the frozen release at startup and serves predictions, evaluation, dataset summaries and clusters |

The pipeline writes a release into `fastapi/artifacts/<name>`. The API serves `fastapi/artifacts/current`. The API never trains models or downloads data.

```text
backend/
├── ai/
│   ├── data.py, sources.py        dataset download, parsing, merging and splitting
│   ├── experiments.py, clustering.py, metrics.py, train.py   model selection and frozen release
│   ├── analysis.py, extra_experiments.py   EDA, extra experiments and figures
│   ├── predict.py, package_submission.py   command-line prediction and submission ZIP
│   ├── data/                      raw and processed datasets (created on first run, not tracked)
│   ├── reports/                   EDA and experiment JSON plus figures
│   └── tests/
└── fastapi/
    ├── app/                       API application
    ├── artifacts/current/         frozen models and evaluation served by the API
    └── tests/
```

## Environment

Python 3.14 is required because the serialized models record the Python and scikit-learn versions they were trained with, and the API refuses a mismatch. `ai/requirements.txt` includes everything in `fastapi/requirements-dev.txt` plus pandas and matplotlib, so one environment runs both folders.

### Option 1: conda

```bash
cd backend/ai
conda env create -f environment.yml
conda activate hn1-8-ai
cd ..
```

The environment uses conda-forge only. If conda stops with a Terms of Service message for `repo.anaconda.com` channels, accept them with the printed `conda tos accept` commands or remove the default channels from your conda configuration.

### Option 2: virtual environment

```powershell
cd backend
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r ai/requirements.txt
```

On macOS/Linux use `.venv/bin/python`. The API alone needs only `fastapi/requirements.txt`.

## Run

All commands below run from `backend/` with the environment active (replace `python` with `.venv/Scripts/python.exe` when using the virtual environment).

| Task | Command |
|---|---|
| Start the API | `python -m uvicorn app.main:app --app-dir fastapi --host 127.0.0.1 --port 8000 --no-access-log` |
| Train a new release | `python -m ai.train --output fastapi/artifacts/my-release` |
| Data analysis figures | `python -m ai.analysis` |
| Extra experiments | `python -m ai.extra_experiments` |
| Predict from the command line | `python -m ai.predict "Your parcel is on hold, pay at bit.ly/x"` |
| Pipeline tests | `python -m pytest ai/tests` |
| API tests | `cd fastapi && python -m pytest` |

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before training so repeated runs are comparable.

## Switching the served release

A new training run does not replace the served models. To serve a new release, stop the API, move the old `fastapi/artifacts/current` out of the way, rename the new release folder to `current`, and start the API again. The API checks every file hash, the model types, the evaluation figures and the split membership at startup. It reports not ready (HTTP 503) instead of serving a release that fails any check.

## Optional configuration

`fastapi/.env` (ignored by Git) can set `INFERENCE_CONCURRENCY` (1 to 16, default 2), `VERIPHONE_API_KEY` for phone number metadata, and `HF_TOKEN` with `HF_MODEL_ID` for an optional external Hugging Face reference model. The application works without any of them; the related features then report that they are unavailable. Never put provider keys in frontend files.
