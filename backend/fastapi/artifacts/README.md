# Serving artifacts

`current/` contains the frozen release exported by the offline `backend/ai` pipeline. The backend reads this directory once at startup. Do not put visitor messages or raw training data here.

The current release is `sms-b5d453f438-2226ac7c49-714333`. It was trained on 6,104 deduplicated messages from the UCI SMS Spam Collection and the Mendeley SMS Phishing Dataset (dataset ID `uci_sms_plus_mendeley_smishing`).

The serving release contains:

- `release.json`: release registry, primary classifier, dataset/split identities and expected hashes.
- `naive_bayes/` and `logistic_regression/`: each model's `pipeline.joblib`, `model-manifest.json` and `evaluation.json`.
- `dataset-summary.json`: sanitized per-sample labels, split, source ID (`uci_sms` or `mendeley_smishing`), lengths and model scores; public endpoints aggregate this file and never expose its rows.
- `cluster-visualization.json`: deterministic test projection points and cluster counts.
- `clustering.joblib` and `projection.joblib`: offline fitted clustering/projection binaries; verified but not deserialized by the serving API.
- `split-manifest.json`: frozen membership checked against summary records.
- `selection-decision.json`: validation-only candidate/threshold/primary selection evidence, referenced by its hash.
- `source-provenance.json`, `quality-report.json`, `training-source.json`, `dummy-evaluation.json`: dataset sources and licences, cleaning counts, training code hashes and the majority-class baseline.

Provenance and research evidence are not automatically exposed as API responses. See the [API guide](../README.md) for startup checks and the [pipeline guide](../../ai/README.md) for how a release is produced.

Only install trusted operator-produced artifacts. A checksum inside a locally editable registry is not a digital signature and cannot establish the safety of an untrusted joblib file. The backend has no model upload/install endpoint. Install a complete reviewed release and restart; do not mutate a running process's models or regenerate evaluation metrics during requests.
