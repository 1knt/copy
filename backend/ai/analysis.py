import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer

from ai.data import SEED, write_json
from ai.emoji_text import text_for_model
from ai.sources import parse_mendeley, read_mendeley_archive, text_features


ROOT = Path(__file__).resolve().parents[1]
INDICATORS = ["has_url", "has_phone", "has_currency", "has_email", "has_emoji"]
COLORS = {"ham": "#4c72b0", "spam": "#dd8452", "smishing": "#c44e52", "unspecified": "#8c8c8c"}


def load_prepared(directory):
    frames = [pd.read_csv(Path(directory) / f"{name}.csv", dtype=str, keep_default_na=False).assign(partition=name) for name in ("train", "validation", "test")]
    data = pd.concat(frames, ignore_index=True)
    features = pd.DataFrame([text_features(text) for text in data["text"]])
    data = pd.concat([data, features], axis=1)
    data["model_text"] = data["text"].map(text_for_model)
    return data


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_sources(data, figures):
    origin = data["sources"].map({"uci_sms": "UCI only", "mendeley_smishing": "Mendeley only", "uci_sms|mendeley_smishing": "Both sources"})
    table = pd.crosstab(origin, data["subtype"]).reindex(columns=[c for c in COLORS if c in set(data["subtype"])])
    figure, axis = plt.subplots(figsize=(7, 4))
    table.plot(kind="bar", stacked=True, ax=axis, color=[COLORS[c] for c in table.columns])
    axis.set_xlabel("Origin of retained message")
    axis.set_ylabel("Messages after deduplication")
    axis.set_title("Retained messages by origin and label")
    axis.tick_params(axis="x", rotation=0)
    save(figure, figures / "eda-origin-by-label.png")
    return {origin_name: {k: int(v) for k, v in row.items()} for origin_name, row in table.iterrows()}


def plot_lengths(train, figures):
    figure, axis = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 330, 10)
    for label in ("ham", "spam"):
        axis.hist(train.loc[train["label"] == label, "length"].clip(upper=320), bins=bins, alpha=0.6, label=label, color=COLORS[label], density=True)
    axis.set_xlabel("Message length in characters (clipped at 320)")
    axis.set_ylabel("Density")
    axis.set_title("Message length distribution, training partition")
    axis.legend()
    save(figure, figures / "eda-length-distribution.png")
    ham = train.loc[train["label"] == "ham", "length"]
    spam = train.loc[train["label"] == "spam", "length"]
    test = mannwhitneyu(spam, ham, alternative="greater")
    return {"median_length": {"ham": float(ham.median()), "spam": float(spam.median())}, "mean_length": {"ham": float(ham.mean()), "spam": float(spam.mean())}, "mann_whitney_spam_longer": {"statistic": float(test.statistic), "p_value": float(test.pvalue)}}


def plot_indicators(train, figures):
    groups = [g for g in ("ham", "spam", "smishing") if g in set(train["subtype"])]
    rates = train[train["subtype"].isin(groups)].groupby("subtype")[INDICATORS].mean().reindex(groups)
    figure, axis = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(groups)
    positions = np.arange(len(INDICATORS))
    for index, group in enumerate(groups):
        axis.bar(positions + index * width, rates.loc[group] * 100, width, label=group, color=COLORS[group])
    axis.set_xticks(positions + width * (len(groups) - 1) / 2, ["URL", "Phone number", "Currency", "Email", "Emoji"])
    axis.set_ylabel("Messages containing indicator (%)")
    axis.set_title("Engineered indicator rates by subtype, training partition")
    axis.legend()
    save(figure, figures / "eda-indicator-rates.png")
    tests = {}
    for indicator in INDICATORS:
        table = pd.crosstab(train["label"], train[indicator])
        if table.shape == (2, 2):
            chi2, p_value, _, _ = chi2_contingency(table)
            tests[indicator] = {"chi2": float(chi2), "p_value": float(p_value)}
    return {"rates_percent": {g: {k: round(float(v) * 100, 2) for k, v in rates.loc[g].items()} for g in groups}, "chi_square_label_independence": tests}


def plot_tokens(train, figures):
    result = {}
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, group in zip(axes, ("ham", "spam", "smishing")):
        texts = train.loc[train["subtype"] == group, "text"]
        vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", lowercase=True)
        counts = np.asarray(vectorizer.fit_transform(texts).sum(axis=0)).ravel()
        order = counts.argsort()[::-1][:12]
        words = vectorizer.get_feature_names_out()[order]
        axis.barh(words[::-1], counts[order][::-1], color=COLORS[group])
        axis.set_title(f"{group} (n={len(texts)})")
        axis.set_xlabel("Token count")
        result[group] = {str(w): int(c) for w, c in zip(words, counts[order])}
    figure.suptitle("Most frequent tokens by subtype, training partition")
    save(figure, figures / "eda-top-tokens.png")
    return result


def extractor_agreement(raw_directory):
    archive = Path(raw_directory) / "mendeley-sms-phishing.zip"
    if not archive.is_file():
        return None
    rows, _, _ = parse_mendeley(read_mendeley_archive(archive.read_bytes()))
    result = {}
    for flag, feature in (("url", "has_url"), ("email", "has_email"), ("phone", "has_phone")):
        published = np.array([row["source_flags"][flag] for row in rows])
        detected = np.array([text_features(row["text"])[feature] for row in rows])
        result[flag] = {"published_positive": int(published.sum()), "detected_positive": int(detected.sum()), "agreement": float((published == detected).mean()), "both_positive": int((published & detected).sum())}
    return {"rows": len(rows), "flags": result, "note": "Agreement between the published Mendeley URL/EMAIL/PHONE attributes and this project's regular-expression extractors."}


def run(prepared, output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = load_prepared(prepared)
    train = data[data["partition"] == "train"]
    summary = {
        "schema_version": 1,
        "seed": SEED,
        "prepared_directory": Path(prepared).name,
        "row_counts": {name: int(count) for name, count in data["partition"].value_counts().items()},
        "label_counts": {name: int(count) for name, count in data["label"].value_counts().items()},
        "subtype_counts": {name: int(count) for name, count in data["subtype"].value_counts().items()},
        "missing_text": int((data["text"].str.strip() == "").sum()),
        "replacement_character_rows": int((data["replacement_characters"] > 0).sum()),
        "emoji_rows": {name: int(count) for name, count in data.loc[data["has_emoji"], "subtype"].value_counts().items()},
        "emoji_total": int(data["emoji_count"].sum()),
        "origin_by_subtype": plot_sources(data, figures),
        "length": plot_lengths(train, figures),
        "indicators": plot_indicators(train, figures),
        "top_tokens": plot_tokens(train, figures),
        "extractor_agreement": extractor_agreement(Path(prepared).parents[1] / "raw"),
    }
    write_json(output / "eda-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis figures and statistics for the prepared combined SMS dataset.")
    parser.add_argument("--prepared", type=Path, default=ROOT / "ai" / "data" / "processed" / f"combined-sms-seed-{SEED}")
    parser.add_argument("--output", type=Path, default=ROOT / "ai" / "reports")
    arguments = parser.parse_args()
    summary = run(arguments.prepared, arguments.output)
    print(f"EDA written for {sum(summary['row_counts'].values())} rows to {arguments.output}")


if __name__ == "__main__":
    main()
