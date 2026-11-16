# Offline Machine Learning Pipeline (Assignment 2)

This package builds the SMS spam screening models used by the web application. It downloads two open datasets, merges and deduplicates them, splits the result into training, validation and test partitions, analyses the data, selects two classifiers and a clustering pipeline on the validation partition, and evaluates the frozen models once on the held-out test partition. It runs offline; the web API never trains on visitor input.


## Intended users and problem

The intended users are adult smartphone users in Vietnam and other English-speaking SMS markets who receive unsolicited messages and are unsure whether a message is a scam, especially people with limited security experience such as older adults. They need a quick, explainable second opinion before they tap a link, call a number or reply. Rule-based carrier filters miss new templates, and smishing messages imitate legitimate delivery, banking and prize notices, so a model that learns lexical patterns from labelled SMS (Almeida, Gómez Hidalgo and Yamakami, 2011) is a suitable screening aid. It does not replace link or malware scanning.

## Requirements

Functional requirements describe what the pipeline does. Non-functional requirements describe how well it must do it. Technical requirements fix the environment it runs in.

### Functional

| ID | Requirement |
|---|---|
| FR-1 | Download both datasets and verify each archive against a pinned SHA-256 hash before parsing. |
| FR-2 | Parse, validate and merge both sources into one binary ham/spam dataset, keeping the smishing subtype. |
| FR-3 | Remove cross-source duplicates and quarantine duplicate groups with conflicting labels before splitting. |
| FR-4 | Split the data 70/15/15 into training, validation and test partitions, stratified by label and origin. |
| FR-5 | Produce exploratory statistics and figures from the processed data. |
| FR-6 | Train and compare classification candidates and select each model and its decision threshold on validation data. |
| FR-7 | Train a clustering pipeline and select the number of clusters on validation data. |
| FR-8 | Evaluate the frozen classifiers once on the test partition and save all metrics and figures. |
| FR-9 | Predict ham or spam for new messages from the command line with a saved model. |
| FR-10 | Package the source code, processed datasets, final models and reports into one submission ZIP. |

### Non-functional

| ID | Category | Requirement | Evidence |
|---|---|---|---|
| NFR-1 | Effectiveness | Spam precision and recall on the test partition are each at least 0.90. | Results table below |
| NFR-2 | Validity | No message appears in more than one partition, and no learned step is fitted on validation or test data. | `ai/tests/test_data.py`, `ai/tests/test_sources.py` |
| NFR-3 | Reproducibility | The same seed and data produce the same split and selections. | Seed 42, lineage and `selection-decision.json` |
| NFR-4 | Integrity | Model files are checked against their manifest hash before loading. | `ai/tests/test_predict.py` |
| NFR-5 | Usability | Every command runs from `backend/` with the documented arguments and prints readable results. | Commands in sections 1 to 3 |
| NFR-6 | Privacy | Reports and figures contain counts and frequent tokens only, never full messages. | Files in `ai/reports/` |

### Technical

| ID | Requirement |
|---|---|
| TR-1 | Python 3.14 in the conda environment `hn1-8-ai`, built from the conda-forge channel only. |
| TR-2 | pandas, scikit-learn, NumPy, SciPy, matplotlib and joblib at the exact versions pinned in `requirements.txt` and `../fastapi/requirements.txt`. |
| TR-3 | Models are saved as joblib pipelines with a JSON manifest; datasets as UTF-8 CSV. |
| TR-4 | The pipeline runs offline on a laptop CPU; no GPU or paid service is needed. |

## Datasets

| Source | Citation | Licence | Role |
|---|---|---|---|
| UCI SMS Spam Collection | Almeida and Gómez Hidalgo (2011) | CC BY 4.0 | Base dataset: 5,574 English SMS labelled ham or spam |
| Mendeley SMS Phishing Dataset v1 | Mishra and Soni (2022) | CC BY 4.0 | Additional dataset: 5,971 SMS labelled ham, spam or smishing, with URL, email and phone attributes |

Both archives are downloaded on first use into `ai/data/raw/` and checked against SHA-256 hashes. The Mendeley archive hash is pinned in `ai/sources.py`.

5,025 of the 5,971 Mendeley messages are exact normalized copies of UCI messages, so the sources are merged before splitting. Otherwise the same message could appear in training and test data. After merging, 6,104 unique messages remain: 5,041 ham and 1,063 spam. The additional dataset contributes 945 new messages and a smishing/spam subtype for 963 spam messages. Its messages were converted to text from images of real messages (Mishra and Soni, 2023), so it adds newer smishing templates that the 2011 collection lacks.

## Environment setup with conda

Run these commands from `backend/ai`:

```bash
conda env create -f environment.yml
conda activate hn1-8-ai
cd ..
```

All later commands run from `backend/`. The environment uses conda-forge only. If conda stops with a Terms of Service message for `repo.anaconda.com` channels, either accept those terms with the `conda tos accept` commands it prints, or remove the default channels from your conda configuration; the project does not need them. To update an existing environment use `conda env update -f ai/environment.yml --prune`. Set single-threaded numerical libraries so repeated runs are comparable:

```bash
# Windows PowerShell
$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
# macOS / Linux
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Check the installation:

```bash
python -m pytest ai/tests -q
```

## 1. Data processing and model training

```bash
python -m ai.train --output fastapi/artifacts/my-release
```

This one command:

1. downloads or reuses both raw archives and verifies their checksums;
2. parses each source, rejects malformed rows, maps Mendeley `spam` and `smishing` to the binary `spam` label and keeps the subtype;
3. deduplicates across sources using Unicode NFKC, casefolding and collapsed whitespace, and quarantines groups whose labels disagree;
4. splits 70/15/15 with a fixed seed (42), stratified by label and source origin;
5. writes the processed datasets to `ai/data/processed/combined-sms-seed-42/` (`train.csv`, `validation.csv`, `test.csv`, quality report, lineage, provenance);
6. tries 4 Naive Bayes and 6 logistic regression candidates on TF-IDF features (Manning, Raghavan and Schütze, 2008) built with scikit-learn (Pedregosa et al., 2011), selects each family's best candidate and decision threshold on validation spam F1, then selects KMeans clustering (k = 2, 4, 6) by validation silhouette (Rousseeuw, 1987);
7. freezes all selections in `selection-decision.json`, and only then evaluates both classifiers once on the test partition.

The output directory must not already exist. Releases are immutable, so a new run never overwrites a previous result. Use `--seed` for a different split and `--data-dir` for a different data folder.

## 2. Further analysis of the processed data

```bash
python -m ai.analysis
python -m ai.extra_experiments --release fastapi/artifacts/my-release
```

`ai.analysis` reads the processed CSV files with pandas (McKinney, 2010), draws figures with matplotlib (Hunter, 2007) and writes statistics to `ai/reports/eda-summary.json` and figures to `ai/reports/figures/eda-*.png`:

- retained messages by origin and label;
- message length distributions, with a Mann-Whitney U test (Mann and Whitney, 1947);
- URL, phone, currency and email indicator rates by subtype, with chi-square tests of independence;
- the most frequent tokens for ham, spam and smishing;
- agreement between the published Mendeley attributes and this project's regular-expression extractors.

`ai.extra_experiments` writes `ai/reports/experiments.json` and the `exp-*`, `cluster-*` and `eval-*` figures:

- eight classifiers compared on the validation partition: Naive Bayes, logistic regression, linear SVM, random forest, decision tree, k-nearest neighbours, engineered features only, and TF-IDF plus engineered features;
- cross-source generalisation: models trained on UCI rows only versus both datasets, evaluated on UCI and Mendeley-only validation messages;
- a second classification task, smishing versus generic spam, that only the additional dataset makes possible;
- KMeans elbow and silhouette for k = 2 to 10, and DBSCAN (Ester et al., 1996) with a k-distance plot, both checked against the true labels with adjusted Rand index (Hubert and Arabie, 1985) and purity;
- ROC curves, precision-recall curves (more informative than ROC for the minority spam class; Saito and Rehmsmeier, 2015), confusion matrices and threshold-search curves for the frozen release models on the test partition.

Exploratory experiments only use the training and validation partitions. The test partition is reserved for the frozen release models.

## 3. Prediction with a trained model

```bash
python -m ai.predict "URGENT! Your parcel is on hold, pay the fee at bit.ly/xyz" "Are we still meeting at 6?"
python -m ai.predict --release fastapi/artifacts/my-release --model logistic_regression "Claim your free prize now"
```

Each message prints one JSON line with the label, spam score and decision threshold. Before scoring, each emoji is replaced by its English name (`🎁` becomes `wrapped gift`, see `emoji_text.py`), so emoji count as words the model already knows instead of being dropped. A message with no known vocabulary returns `unable_to_assess` instead of a guess. The default release is `fastapi/artifacts/current`; `--model` accepts `primary`, `naive_bayes` or `logistic_regression`. The pipeline file is checked against its manifest hash before loading.

## Modules

| Module | Responsibility |
|---|---|
| `data.py` | UCI download and parsing, deduplication helpers, stratified split, frozen CSV/JSON exports |
| `sources.py` | Mendeley download and parsing, cross-source merge, engineered text features |
| `experiments.py` | Naive Bayes and logistic regression candidates and threshold search |
| `clustering.py` | TF-IDF, SVD and KMeans candidates, silhouette and stability selection, 2D projection |
| `metrics.py` | Spam-positive metrics, undefined denominators, abstention reporting |
| `train.py` | Release orchestration, frozen decisions, final evaluation and API artifacts |
| `analysis.py` | Exploratory data analysis figures and statistics |
| `extra_experiments.py` | Additional experiments and evaluation figures |
| `predict.py` | Command-line prediction with a frozen release |
| `package_submission.py` | Builds the submission ZIP: source code, processed datasets, final models and reports |

To build the submission archive: `python -m ai.package_submission --output ../../a2-submission/hn1-group-8-assignment2-code.zip`.

## Results of the current release

Release `sms-b5d453f438-2226ac7c49-714333`, trained on 4,272 messages, validated on 916 and tested on 916:

| Model | Threshold | Test precision | Test recall | Test F1 | ROC AUC |
|---|---|---|---|---|---|
| Naive Bayes (primary) | 0.25 | 0.939 | 0.962 | 0.950 | 0.996 |
| Logistic regression | 0.35 | 0.974 | 0.956 | 0.965 | 0.997 |
| Majority-class baseline | - | undefined | 0.000 | undefined | - |

Training on both datasets raised Naive Bayes spam recall on unseen Mendeley-only messages from 0.587 to 0.841, and logistic regression from 0.635 to 0.810, with no loss on UCI messages. Clustering is exploratory: validation silhouettes stay below 0.05, and DBSCAN labels most messages as noise.

## Limits

- Exact deduplication does not remove near-duplicate templates.
- The Mendeley text was converted from images and contains replacement characters in 145 messages.
- The data is English SMS from 2011 and 2022. Other languages, email and malware files are out of scope.
- Scores are uncalibrated and are not a safety guarantee.
- Timings and binary hashes can differ between machines and library versions.

## References

Almeida, T.A. and Gómez Hidalgo, J.M. (2011) *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. Available at: https://doi.org/10.24432/C5CC84.

Almeida, T.A., Gómez Hidalgo, J.M. and Yamakami, A. (2011) 'Contributions to the study of SMS spam filtering: new collection and results', in *Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11)*. Mountain View, CA: ACM, pp. 259–262.

Ester, M., Kriegel, H.-P., Sander, J. and Xu, X. (1996) 'A density-based algorithm for discovering clusters in large spatial databases with noise', in *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*. Portland, OR: AAAI Press, pp. 226–231.

Hubert, L. and Arabie, P. (1985) 'Comparing partitions', *Journal of Classification*, 2(1), pp. 193–218.

Hunter, J.D. (2007) 'Matplotlib: a 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90–95.

Mann, H.B. and Whitney, D.R. (1947) 'On a test of whether one of two random variables is stochastically larger than the other', *The Annals of Mathematical Statistics*, 18(1), pp. 50–60.

Manning, C.D., Raghavan, P. and Schütze, H. (2008) *Introduction to information retrieval*. Cambridge: Cambridge University Press.

McKinney, W. (2010) 'Data structures for statistical computing in Python', in *Proceedings of the 9th Python in Science Conference*. Austin, TX, pp. 56–61.

Mishra, S. and Soni, D. (2022) *SMS phishing dataset for machine learning and pattern recognition* [Dataset]. Mendeley Data, V1. Available at: https://doi.org/10.17632/f45bkkt8pr.1.

Mishra, S. and Soni, D. (2023) 'DSmishSMS: a system to detect smishing SMS', *Neural Computing and Applications*, 35(7), pp. 4975–4992. Available at: https://doi.org/10.1007/s00521-021-06305-y.

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Rousseeuw, P.J. (1987) 'Silhouettes: a graphical aid to the interpretation and validation of cluster analysis', *Journal of Computational and Applied Mathematics*, 20, pp. 53–65.

Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', *PLoS ONE*, 10(3), e0118432. Available at: https://doi.org/10.1371/journal.pone.0118432.
