import csv
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from ai.data import MAX_DOWNLOAD_BYTES, download_source, duplicate_key, sha256_bytes, timestamp, write_json
from ai.emoji_text import find_emojis


MENDELEY_URL = "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/edb361de-918d-469f-9106-e84823830665/file_downloaded"
MENDELEY_PAGE = "https://data.mendeley.com/datasets/f45bkkt8pr/1"
MENDELEY_ARCHIVE_SHA256 = "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3"
MENDELEY_MEMBER = "Dataset_5971.csv"
MENDELEY_LABELS = {"ham": ("ham", "ham"), "spam": ("spam", "spam"), "smishing": ("spam", "smishing")}
SOURCE_ORDER = ("uci_sms", "mendeley_smishing")
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|net|org|co\.uk|io|ly|me|info|biz|uk|us)(?:/\S*)?\b")
EMAIL_PATTERN = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s-]{4,}\d(?!\w)")
CURRENCY_PATTERN = re.compile(r"[£$€]|\b(?:gbp|usd|pounds?|dollars?)\b", re.IGNORECASE)


def read_mendeley_archive(archive):
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        matches = [item for item in bundle.infolist() if item.filename == MENDELEY_MEMBER]
        if len(matches) != 1 or matches[0].file_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("The Mendeley archive does not contain one bounded expected dataset member.")
        return bundle.read(matches[0])


def download_mendeley(raw_directory):
    raw_directory = Path(raw_directory)
    raw_directory.mkdir(parents=True, exist_ok=True)
    archive_path = raw_directory / "mendeley-sms-phishing.zip"
    provenance_path = raw_directory / "mendeley-provenance.json"
    if archive_path.exists():
        archive = archive_path.read_bytes()
    else:
        request = urllib.request.Request(MENDELEY_URL, headers={"User-Agent": "HN1-8-SMS-Research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(archive) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Mendeley download exceeds the documented size bound.")
    if sha256_bytes(archive) != MENDELEY_ARCHIVE_SHA256:
        raise ValueError("Mendeley archive checksum differs from the pinned published file.")
    payload = read_mendeley_archive(archive)
    if not archive_path.exists():
        archive_path.write_bytes(archive)
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        provenance = {
            "source_id": "mendeley_smishing",
            "dataset_id": "mendeley_sms_phishing_dataset_v1",
            "source_url": MENDELEY_URL,
            "source_page": MENDELEY_PAGE,
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Mishra, S. and Soni, D. (2022). SMS Phishing Dataset for Machine Learning and Pattern Recognition. Mendeley Data, V1. https://doi.org/10.17632/f45bkkt8pr.1",
            "retrieved_at": timestamp(),
            "archive_sha256": MENDELEY_ARCHIVE_SHA256,
            "dataset_sha256": sha256_bytes(payload),
            "archive_member": MENDELEY_MEMBER,
            "parser_version": "mendeley-csv-v1",
        }
        write_json(provenance_path, provenance)
    if sha256_bytes(payload) != provenance["dataset_sha256"]:
        raise ValueError("Mendeley dataset member checksum differs from recorded provenance.")
    return payload, provenance


def parse_uci(payload):
    dataset_hash = sha256_bytes(payload)
    rows, rejected = [], []
    lines = payload.decode("utf-8", errors="strict").splitlines()
    for position, line in enumerate(lines, start=1):
        row_id = sha256_bytes(f"{dataset_hash}:{position}".encode())[:24]
        fields = line.split("\t", 1)
        if len(fields) != 2 or fields[0] not in {"ham", "spam"} or not fields[1].strip():
            rejected.append({"row_id": row_id, "source_id": "uci_sms", "line_number": position, "issue_code": "INVALID_ROW"})
            continue
        label, text = fields
        rows.append({"sample_id": row_id, "source_id": "uci_sms", "line_number": position, "label": label, "subtype": "ham" if label == "ham" else None, "text": text.strip(), "source_flags": None})
    return rows, rejected, len(lines)


def parse_mendeley(payload):
    dataset_hash = sha256_bytes(payload)
    rows, rejected = [], []
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8", errors="strict")))
    if reader.fieldnames != ["LABEL", "TEXT", "URL", "EMAIL", "PHONE"]:
        raise ValueError("Unexpected Mendeley column layout.")
    total = 0
    for position, record in enumerate(reader, start=2):
        total += 1
        row_id = sha256_bytes(f"{dataset_hash}:{position}".encode())[:24]
        raw_label = (record["LABEL"] or "").strip().casefold()
        text = (record["TEXT"] or "").strip()
        if raw_label not in MENDELEY_LABELS or not text:
            rejected.append({"row_id": row_id, "source_id": "mendeley_smishing", "line_number": position, "issue_code": "INVALID_ROW"})
            continue
        label, subtype = MENDELEY_LABELS[raw_label]
        flags = {name.lower(): (record[name] or "").strip().casefold() == "yes" for name in ("URL", "EMAIL", "PHONE")}
        rows.append({"sample_id": row_id, "source_id": "mendeley_smishing", "line_number": position, "label": label, "subtype": subtype, "text": text, "source_flags": flags})
    return rows, rejected, total


def text_features(text):
    emoji_count = len(find_emojis(text))
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "digit_ratio": sum(character.isdigit() for character in text) / max(len(text), 1),
        "uppercase_ratio": sum(character.isupper() for character in text) / max(sum(character.isalpha() for character in text), 1),
        "has_url": bool(URL_PATTERN.search(text)),
        "has_email": bool(EMAIL_PATTERN.search(text)),
        "has_phone": bool(PHONE_PATTERN.search(text)),
        "has_currency": bool(CURRENCY_PATTERN.search(text)),
        "exclamation_count": text.count("!"),
        "has_emoji": emoji_count > 0,
        "emoji_count": emoji_count,
        "replacement_characters": text.count("�"),
    }


def group_subtype(members):
    subtypes = {item["subtype"] for item in members if item["subtype"] is not None}
    if subtypes == {"ham"}:
        return "ham"
    if subtypes == {"smishing"}:
        return "smishing"
    if subtypes == {"spam"}:
        return "spam"
    return "unspecified"


def merge_sources(parsed):
    groups = defaultdict(list)
    quality_sources = {}
    rejected = []
    for source_id in SOURCE_ORDER:
        rows, source_rejected, total = parsed[source_id]
        rejected.extend(source_rejected)
        quality_sources[source_id] = {"raw_count": total, "invalid_count": len(source_rejected), "parsed_count": len(rows), "label_counts": dict(sorted(Counter(row["subtype"] or row["label"] for row in rows).items()))}
        for row in rows:
            group_id = sha256_bytes(duplicate_key(row["text"]).encode("utf-8"))
            groups[group_id].append({**row, "group_id": group_id})
    records, mapping = [], []
    conflicts = {"label_groups": 0, "label_rows": 0, "subtype_groups": 0}
    removed = 0
    origin_counts = Counter()
    for group_id in sorted(groups):
        members = sorted(groups[group_id], key=lambda item: (SOURCE_ORDER.index(item["source_id"]), item["line_number"]))
        if len({item["label"] for item in members}) != 1:
            conflicts["label_groups"] += 1
            conflicts["label_rows"] += len(members)
            for member in members:
                mapping.append({"row_id": member["sample_id"], "source_id": member["source_id"], "group_id": group_id, "representative_id": None, "decision": "quarantined_conflicting_labels"})
            continue
        sources = sorted({item["source_id"] for item in members}, key=SOURCE_ORDER.index)
        subtype = group_subtype(members)
        if subtype == "unspecified" and any(item["source_id"] == "mendeley_smishing" for item in members):
            conflicts["subtype_groups"] += 1
        flags = next((item["source_flags"] for item in members if item["source_flags"] is not None), None)
        representative = {**members[0], "sources": "|".join(sources), "subtype": subtype, "source_flags": flags}
        records.append(representative)
        origin_counts["+".join(sources)] += 1
        removed += len(members) - 1
        for index, member in enumerate(members):
            mapping.append({"row_id": member["sample_id"], "source_id": member["source_id"], "group_id": group_id, "representative_id": representative["sample_id"], "decision": "retained" if index == 0 else ("removed_cross_source_duplicate" if member["source_id"] != representative["source_id"] else "removed_duplicate")})
    raw_total = sum(item["raw_count"] for item in quality_sources.values())
    quality = {
        "sources": quality_sources,
        "raw_count": raw_total,
        "invalid_count": len(rejected),
        "duplicate_rows_removed": removed,
        "cross_source_duplicates_removed": sum(item["decision"] == "removed_cross_source_duplicate" for item in mapping),
        "conflicting_group_count": conflicts["label_groups"],
        "conflicting_row_count": conflicts["label_rows"],
        "subtype_disagreement_groups": conflicts["subtype_groups"],
        "retained_count": len(records),
        "class_counts": dict(sorted(Counter(item["label"] for item in records).items())),
        "subtype_counts": dict(sorted(Counter(item["subtype"] for item in records).items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "duplicate_normalization": "Unicode NFKC, casefold, collapsed whitespace across both sources; UCI rows are preferred as representatives",
        "label_mapping": "Mendeley ham->ham; Mendeley spam and smishing->spam for the binary task; the smishing/spam subtype is retained separately",
        "residual_risk": "Exact normalized deduplication does not remove semantic or templated near-duplicates, and OCR-derived Mendeley text contains replacement characters.",
        "rejected_rows": rejected,
    }
    if raw_total != len(records) + removed + conflicts["label_rows"] + len(rejected):
        raise ValueError("Combined data-quality counts do not reconcile.")
    return records, mapping, quality


def load_combined(raw_directory):
    uci_payload, uci_provenance = download_source(raw_directory)
    mendeley_payload, mendeley_provenance = download_mendeley(raw_directory)
    parsed = {"uci_sms": parse_uci(uci_payload), "mendeley_smishing": parse_mendeley(mendeley_payload)}
    records, mapping, quality = merge_sources(parsed)
    combined_hash = sha256_bytes(f"{uci_provenance['dataset_sha256']}:{mendeley_provenance['dataset_sha256']}".encode())
    provenance = {
        "source_id": "combined",
        "dataset_id": "uci_sms_plus_mendeley_smishing",
        "dataset_sha256": combined_hash,
        "combination": "SHA-256 of the two member dataset hashes joined in source order",
        "sources": [uci_provenance, mendeley_provenance],
        "parser_version": "combined-v1",
    }
    return records, mapping, quality, provenance
