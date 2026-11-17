# Spam Screening API

FastAPI serves the frozen English SMS classifiers trained in Assignment 2, together with their evaluation, dataset summary and clustering artifacts. The models were trained on the UCI SMS Spam Collection and the Mendeley SMS Phishing Dataset. The API loads the packaged release once during process startup. It does not train models, fetch data, accept model uploads or retain submitted messages.

## Setup and launch

The packaged release requires Python 3.14 and the exact scikit-learn, NumPy, SciPy and joblib versions in the requirement files. The loader accepts the same Python major/minor version and verifies the exact four library versions recorded by training. The environment is shared with the Assignment 2 pipeline; see [the backend guide](../README.md).

From `backend/` on Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r fastapi/requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir fastapi --host 127.0.0.1 --port 8000 --no-access-log
```

On macOS/Linux substitute `.venv/bin/python`. Runtime-only installations can use `fastapi/requirements.txt`. These version pins are not hash-verified supply-chain lockfiles. Start the frontend separately from `frontend/` with `npm run dev`.

With conda, create the shared environment from `backend/ai/environment.yml` and start the API from `backend/`:

```powershell
cd backend/ai
conda env create -f environment.yml
conda activate hn1-8-ai
cd ..
python -m uvicorn app.main:app --app-dir fastapi --host 127.0.0.1 --port 8000 --no-access-log
```

Verified on 13 October 2026 (Windows 11): the environment built Python 3.14.8 from conda-forge, `pip check` found no broken requirements, and both the pipeline and API test suites passed inside it. `fastapi/environment.yml` remains for a serving-only environment created from `backend/fastapi`.

## Implemented HTTP endpoints

Open `http://127.0.0.1:8000/docs` for typed OpenAPI request and successful response schemas.

| Endpoint | Successful behavior |
|---|---|
| `GET /api/v1/health/live` | Process liveness, independent of model readiness |
| `GET /api/v1/health/ready` | Release ID and primary model version when the complete bundle is valid |
| `POST /api/v1/predictions` | Primary frozen classifier result, or explicit zero-feature abstention |
| `POST /api/v1/comparisons` | Exactly one result for each frozen NB and LR classifier on the same input |
| `GET /api/v1/model` | Sanitized model metadata, source attribution and limitations |
| `GET /api/v1/evaluation` | Measured test evaluation and operational abstention/coverage statistics |
| `GET /api/v1/dataset-summary` | Filtered counts, message-length bins and classifier score histograms |
| `GET /api/v1/clusters` | At most 1,000 deterministic test projection points and cluster sizes |

`model_id` allows `primary`, `naive_bayes` and `logistic_regression`; the primary alias resolves through the loaded registry. Dataset filters allow `source=all|uci_sms|mendeley_smishing` (a message found in both datasets is counted under `uci_sms`; `mendeley_smishing` covers only messages the additional dataset contributed), `split=all|train|validation|test` and `actual_class=all|ham|spam`. Unknown values receive 422. Cluster filters must exist in the frozen summaries, including valid clusters with no sampled points. Health endpoints do not fabricate a readiness success if any required artifact fails validation.

POST bodies require `application/json` and are capped at 16 KiB including JSON escaping. Text-classification endpoints accept exactly one `text` string. Text is trimmed before its 1-2,000 Unicode-code-point limit is checked. Number lookup uses the separate `numbers` contract below. Streaming requests and inconsistent, duplicate or malformed Content-Length values cannot bypass the boundary. Scores are validated as finite probabilities between zero and one; `spam_score >= threshold` means spam. A zero-feature vector returns `unable_to_assess` with null score and label.

Errors use `{ "error": { "code", "message", "request_id" } }`. IDs are generated on the server and returned in `X-Request-ID`; caller-supplied IDs are ignored. Responses and errors never echo input messages. Request-path failures are caught inside the application boundary before they can escape into Uvicorn exception logging. Keep `--no-access-log` and equivalent proxy query/body logging controls, because arbitrary caller query strings may contain sensitive data even though this API uses POST bodies for messages.

## Optional external services

At startup the API loads `backend/fastapi/.env` without overriding deployment environment variables. This file is ignored by Git. Set `VERIPHONE_API_KEY` for number metadata; set `HF_TOKEN` and `HF_MODEL_ID=owner/model` for an explicitly chosen Hugging Face text-classification model served by hf-inference. Restart after changing configuration. On Vercel, set these as server environment variables. Never put provider keys in frontend variables. Tests disable dotenv loading and mock provider connections; no real credentials or provider calls are required.

`POST /api/v1/number-lookup` accepts `{"numbers":["+447700900123","80086"]}`: one through five strings, each one through twenty characters. The response has `request_id`, `provider: "veriphone"`, and ordered `results` containing `input`, `status`, `valid`, `e164`, `country_name`, `country_code`, `carrier`, `line_type`, and `reason_code`. Status is `looked_up`, `not_a_number`, or `unavailable`. Metadata fields are nullable. Short local digit strings return `SHORT_CODE`; other national-format numbers return `COUNTRY_CODE_REQUIRED` with unknown validity instead of guessing a country. Formatting separators are removed, and `00` is normalized to `+`. Provider-confirmed invalid numbers retain `looked_up` with `valid: false`.

Missing credentials, provider authorization/quota failures or network unavailability return 503 `NUMBER_LOOKUP_UNAVAILABLE`. Malformed responses return 502 `NUMBER_LOOKUP_FAILED`. A failure after an earlier successful provider lookup returns 502/503 `NUMBER_LOOKUP_FAILED` without partial results. Each process allows two simultaneous lookup batches; excess requests return 503. Calls use a five-second socket timeout and a 64 KiB response cap. No automatic retry occurs. Veriphone receives a POST form body, bearer authorization, static mode and `record=false`; numbers never enter the outbound URL. Static carrier metadata is numbering-plan information, not reputation, current reachability or a spam verdict. See [Veriphone's API contract](https://veriphone.io/docs/v3).

`POST /api/v1/external-reference` accepts the same `{"text":"..."}` body as prediction. It returns `request_id`, `provider: "huggingface"`, `source: "external_reference"`, configured `model_id`, and `scores: [{"label":"...","score":0.8}]`. Labels retain provider meanings without an assumed ham/spam mapping. This is an external reference, separate from the team's frozen classifiers and measured evaluation. Missing configuration, unauthorized tokens, unsupported models, invalid responses and provider failures return 503 `EXTERNAL_REFERENCE_UNAVAILABLE`; no substitute result is generated. The provider receives the submitted text only through a POST JSON body with bearer authorization. The connection has a ten-second socket timeout, no redirects and a 64 KiB response cap. See [Hugging Face's classification contract](https://huggingface.co/docs/inference-providers/tasks/text-classification). Review the chosen provider's data policy before submitting sensitive content.

Both external endpoints use the existing error envelope, request ID, JSON content type and body-size boundaries. The API does not persist or log submitted numbers, messages, credentials or provider response bodies. Number-lookup success intentionally echoes each input in its response. Disable proxy/access query and body logging as described above.

Before scoring, `app/emoji_text.py` (a copy of `ai/emoji_text.py`) replaces each emoji with its English name, so `🎁` is read as `wrapped gift` instead of being dropped by the tokenizer. Responses never echo the message or its emoji.

`GET /api/v1/model` and `GET /api/v1/evaluation` expose nullable `positive_label`. The loader fills it from the integrity-checked model manifest rather than inferring it from class order or rewriting evaluation artifacts.

## Release validation and lifecycle

The default directory is `backend/fastapi/artifacts/current`. `create_app(artifact_dir=Path(...))` supports isolated tests and operator-controlled deployments, not HTTP requests. The loader permits fixed filenames only, rejects symlink escape and oversized files, checks SHA-256 hashes before any deserialization, checks release/manifest/evaluation/data identities, validates split membership, and checks evaluation matrices, metrics and abstention accounting. Referenced projection and selection artifacts must also match their hashes. Unknown artifact registry entries are rejected.

Only trusted locally trained pipelines may be installed. Joblib uses pickle semantics: matching hashes detect accidental corruption relative to the registry, but do not make an attacker-supplied registry or pipeline safe. Keep the directory writable only by the deployment operator. No public file path, upload, callback or model-install endpoint exists.

The classifier must be exactly the expected scikit-learn Pipeline with a fitted TF-IDF vectorizer and NB/LR estimator. The loader checks fitted classes, vocabulary, feature configuration, finite learned parameters and probability output. Clustering and projection binaries are integrity-checked but are not deserialized during serving; the API reads their frozen sanitized projection artifact.

The active release is held in memory for the lifetime of the process. Replacing files has no live effect; restart to load a reviewed replacement. Missing or invalid bundles keep liveness available while readiness and dependent endpoints return sanitized 503 errors. Both classifiers and their supporting artifacts must pass together.

## Concurrency and aggregate interpretation

Inference endpoints run in worker threads, outside the async event loop. `INFERENCE_CONCURRENCY` defaults to `2` and must be an integer from 1 through 16. A nonblocking semaphore caps active inference per process; excess requests receive 503 `SERVICE_BUSY`. A comparison uses one slot for both classifiers. Multiple worker processes have independent slots and model copies. There is no public queue or automatic request retry.

Histogram intervals are left-inclusive/right-exclusive except the final score interval includes 1.0. The final length bin covers lengths of 2,000 or more within the validated artifact limit, and its `overflow_count` is reported separately. Each all-score count equals ham plus spam; zero-feature rows are counted as abstentions and excluded from score bins. `sample_count` counts rows matching all requested filters, while `excluded_count` counts retained release rows excluded by those filters; it does not represent malformed source rows removed during preparation.

Clustering points contain only stable sample IDs, projection coordinates and cluster IDs. Sampling chooses the lexicographically first 1,000 test IDs before any cluster filter. `sample_count` is returned points; `total_count` is the full test population for the selected clusters. Projections are exploratory and are not probabilities or classification labels.

## Verification and module boundaries

```powershell
cd backend/fastapi
../.venv/Scripts/python.exe -m pytest
```

The backend suite includes independently trained tiny fixture pipelines, direct-inference parity, class-order/threshold boundaries, malformed and tampered artifacts, bounded inference, privacy, Unicode/framing and histogram/cluster filtering. The packaged-release suite checks the actual frozen release. Windows may skip the filesystem symlink-creation test when the host denies that privilege; the same test runs where symlinks are permitted.

- `app/main.py`: application construction, startup loading and configuration.
- `app/artifacts.py`: trusted artifact reading, validation and release assembly.
- `app/service.py`: inference decisions, concurrency and aggregate calculations.
- `app/routes.py`: HTTP endpoint contracts.
- `app/schemas.py`: request and public response types.
- `app/middleware.py`: request boundaries, privacy headers and exception containment.
- `app/errors.py`: stable error envelopes.
- `tests/`: security, artifact, inference and HTTP regression checks.

The service is a course project evaluated on English SMS data from 2011 and 2022. It is not a malware scanner or a guarantee of safety.
