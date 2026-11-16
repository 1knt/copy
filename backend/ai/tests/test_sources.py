import io
import zipfile

import pytest

from ai.data import split_records
from ai.sources import merge_sources, parse_mendeley, parse_uci, read_mendeley_archive, text_features


MENDELEY = (
    "LABEL,TEXT,URL,EMAIL,PHONE\n"
    "ham,Are you coming home?,No,No,No\n"
    "Smishing,Your parcel is held: pay at bit.ly/x1,yes,No,No\n"
    "spam,WIN a FREE prize now,No,No,No\n"
    "smishing,WIN a free prize NOW,No,No,No\n"
    "unknown,bad label,No,No,No\n"
    "ham,Offer ends today,No,No,No\n"
)
UCI = "ham\tAre you  coming home?\nspam\tOffer ends today\nspam\tCall 09061701461 to claim\n"


def parsed():
    return {"uci_sms": parse_uci(UCI.encode()), "mendeley_smishing": parse_mendeley(MENDELEY.encode())}


def test_mendeley_labels_map_to_binary_and_subtype():
    rows, rejected, total = parse_mendeley(MENDELEY.encode())
    assert total == 6 and len(rejected) == 1
    assert [(row["label"], row["subtype"]) for row in rows] == [("ham", "ham"), ("spam", "smishing"), ("spam", "spam"), ("spam", "smishing"), ("ham", "ham")]
    assert rows[1]["source_flags"] == {"url": True, "email": False, "phone": False}


def test_merge_prefers_uci_quarantines_conflicts_and_reconciles():
    records, mapping, quality = merge_sources(parsed())
    by_text = {row["text"]: row for row in records}
    home = by_text["Are you  coming home?"]
    assert home["source_id"] == "uci_sms" and home["sources"] == "uci_sms|mendeley_smishing"
    assert "Offer ends today" not in by_text
    assert quality["conflicting_row_count"] == 2
    assert quality["cross_source_duplicates_removed"] == 1
    prize = by_text["WIN a FREE prize now"]
    assert prize["subtype"] == "unspecified" and quality["subtype_disagreement_groups"] == 1
    assert by_text["Your parcel is held: pay at bit.ly/x1"]["subtype"] == "smishing"
    assert by_text["Call 09061701461 to claim"]["subtype"] == "unspecified"
    assert quality["raw_count"] == len(records) + quality["duplicate_rows_removed"] + quality["conflicting_row_count"] + quality["invalid_count"]
    assert {item["decision"] for item in mapping} >= {"retained", "removed_cross_source_duplicate", "quarantined_conflicting_labels"}


def test_combined_split_keeps_sources_in_every_partition():
    uci = "\n".join(f"{'spam' if n % 4 == 0 else 'ham'}\tUCI message {n}" for n in range(200))
    mendeley = "LABEL,TEXT,URL,EMAIL,PHONE\n" + "\n".join(f"{'smishing' if n % 4 == 0 else 'ham'},Mendeley message {n},No,No,No" for n in range(80))
    records, _, _ = merge_sources({"uci_sms": parse_uci(uci.encode()), "mendeley_smishing": parse_mendeley(mendeley.encode())})
    partitions = split_records(records)
    for rows in partitions.values():
        assert {row["source_id"] for row in rows} == {"uci_sms", "mendeley_smishing"}
        assert {row["label"] for row in rows} == {"ham", "spam"}


def test_text_features_detect_indicators():
    features = text_features("URGENT! Claim £100 at www.example.com or call +44 7700 900123, mail me@x.org")
    assert features["has_url"] and features["has_email"] and features["has_phone"] and features["has_currency"]
    assert features["exclamation_count"] == 1
    plain = text_features("see you later")
    assert not (plain["has_url"] or plain["has_email"] or plain["has_phone"] or plain["has_currency"])


def test_mendeley_archive_requires_exact_member():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.csv", "x")
    with pytest.raises(ValueError, match="one bounded expected"):
        read_mendeley_archive(buffer.getvalue())
