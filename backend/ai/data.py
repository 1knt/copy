import csv
import hashlib
import io
import json
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split


SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
SEED = 42


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    serialized = json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    Path(path).write_bytes(serialized.encode("utf-8"))


def duplicate_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_archive(archive):
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        matches = [item for item in bundle.infolist() if item.filename == "SMSSpamCollection"]
        if len(matches) != 1 or matches[0].file_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("The source archive does not contain one bounded expected dataset member.")
        payload = bundle.read(matches[0])
    return payload


def download_source(raw_directory):
    raw_directory = Path(raw_directory)
    raw_directory.mkdir(parents=True, exist_ok=True)
    archive_path = raw_directory / "uci-sms-spam-collection.zip"
    provenance_path = raw_directory / "source-provenance.json"
    if archive_path.exists():
        if not provenance_path.is_file():
            raise ValueError("Cached source lacks provenance; use a fresh data directory.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        archive = archive_path.read_bytes()
        if sha256_bytes(archive) != provenance["archive_sha256"]:
            raise ValueError("Cached archive checksum differs from recorded provenance.")
    else:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "HN1-8-SMS-Research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(archive) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Source download exceeds the documented size bound.")
        payload = read_archive(archive)
        provenance = {
            "source_id": "uci_sms",
            "dataset_id": "uci_sms_spam_collection",
            "source_url": SOURCE_URL,
            "source_page": SOURCE_PAGE,
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84",
            "retrieved_at": timestamp(),
            "archive_sha256": sha256_bytes(archive),
            "dataset_sha256": sha256_bytes(payload),
            "archive_member": "SMSSpamCollection",
            "parser_version": "uci-tab-v1",
        }
        archive_path.write_bytes(archive)
        write_json(provenance_path, provenance)
    payload = read_archive(archive)
    if sha256_bytes(payload) != provenance["dataset_sha256"]:
        raise ValueError("Dataset member checksum differs from recorded provenance.")
    return payload, provenance


def prepare_records(payload):
    dataset_hash = sha256_bytes(payload)
    groups = defaultdict(list)
    rejected = []
    lines = payload.decode("utf-8", errors="strict").splitlines()
    for position, line in enumerate(lines, start=1):
        row_id = sha256_bytes(f"{dataset_hash}:{position}".encode())[:24]
        fields = line.split("\t", 1)
        if len(fields) != 2 or fields[0] not in {"ham", "spam"} or not fields[1].strip():
            rejected.append({"row_id": row_id, "line_number": position, "issue_code": "INVALID_ROW"})
            continue
        label, text = fields
        group_id = sha256_bytes(duplicate_key(text).encode("utf-8"))
        groups[group_id].append({"sample_id": row_id, "line_number": position, "label": label, "text": text.strip(), "group_id": group_id, "source_id": "uci_sms"})
    records = []
    mapping = []
    conflicting_groups = 0
    conflicting_rows = 0
    removed_duplicates = 0
    for group_id in sorted(groups):
        members = groups[group_id]
        if len({item["label"] for item in members}) != 1:
            conflicting_groups += 1
            conflicting_rows += len(members)
            for member in members:
                mapping.append({"row_id": member["sample_id"], "group_id": group_id, "representative_id": None, "decision": "quarantined_conflicting_labels"})
            continue
        representative = members[0]
        records.append(representative)
        removed_duplicates += len(members) - 1
        for index, member in enumerate(members):
            mapping.append({"row_id": member["sample_id"], "group_id": group_id, "representative_id": representative["sample_id"], "decision": "retained" if index == 0 else "removed_duplicate"})
    quality = {
        "raw_count": len(lines),
        "invalid_count": len(rejected),
        "parsed_count": len(lines) - len(rejected),
        "conflicting_group_count": conflicting_groups,
        "conflicting_row_count": conflicting_rows,
        "duplicate_rows_removed": removed_duplicates,
        "retained_count": len(records),
        "class_counts": dict(sorted(Counter(item["label"] for item in records).items())),
        "duplicate_normalization": "Unicode NFKC, casefold, collapsed whitespace; model text preserves punctuation and case",
        "residual_risk": "Exact normalized deduplication does not remove all semantic or templated near-duplicates.",
        "rejected_rows": rejected,
    }
    if quality["raw_count"] != len(records) + removed_duplicates + conflicting_rows + len(rejected):
        raise ValueError("Data-quality counts do not reconcile.")
    return records, mapping, quality


def stratum(row):
    return f"{row['label']}|{row.get('sources', row['source_id'])}"


def split_records(records, seed=SEED):
    train, remainder = train_test_split(records, test_size=0.30, stratify=[stratum(row) for row in records], random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.50, stratify=[stratum(row) for row in remainder], random_state=seed)
    partitions = {"train": train, "validation": validation, "test": test}
    return {name: sorted(rows, key=lambda row: row["sample_id"]) for name, rows in partitions.items()}


def save_prepared_data(directory, partitions, provenance, quality, mapping, seed=SEED):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assignments = [{"sample_id": row["sample_id"], "group_id": row["group_id"], "source_id": row["source_id"], "partition": name} for name, rows in partitions.items() for row in rows]
    manifest = {"schema_version": 1, "dataset_sha256": provenance["dataset_sha256"], "seed": seed, "requested_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15}, "counts": {name: len(rows) for name, rows in partitions.items()}, "assignments": assignments}
    manifest_path = directory / "split-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Prepared directory already contains a different frozen split; choose a new data directory.")
    write_json(manifest_path, manifest)
    write_json(directory / "quality-report.json", quality)
    write_json(directory / "duplicate-lineage.json", mapping)
    write_json(directory / "source-provenance.json", provenance)
    fieldnames = ["sample_id", "group_id", "source_id", "sources", "line_number", "label", "subtype", "text"]
    for name, rows in partitions.items():
        with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return manifest, sha256_file(manifest_path)
