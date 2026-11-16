import io
import json
import zipfile

import pytest

from ai.data import duplicate_key, prepare_records, read_archive, save_prepared_data, sha256_bytes, split_records


def test_duplicates_and_conflicts_reconcile():
    payload = "ham\tHello World\nham\tＨＥＬＬＯ  world\nspam\tOffer now\nham\tOFFER NOW\nspam\tWin prize\ninvalid\tbad label\nham\t \n".encode()
    rows, mapping, quality = prepare_records(payload)
    assert {row["text"] for row in rows} == {"Hello World", "Win prize"}
    assert quality["raw_count"] == 7
    assert quality["retained_count"] == 2
    assert quality["duplicate_rows_removed"] == 1
    assert quality["conflicting_row_count"] == 2
    assert quality["invalid_count"] == 2
    assert sum(item["decision"] == "quarantined_conflicting_labels" for item in mapping) == 2
    duplicate = next(item for item in mapping if item["decision"] == "removed_duplicate")
    assert duplicate["representative_id"] in {row["sample_id"] for row in rows}


def test_duplicate_key_is_unicode_aware():
    assert duplicate_key(" ＦＲＥＥ\tPrize ") == "free prize"
    assert duplicate_key("FREE!!!") != duplicate_key("FREE")


def test_splits_are_deterministic_disjoint_and_stratified():
    payload = "\n".join(f"{'spam' if number % 5 == 0 else 'ham'}\tUnique message number {number}" for number in range(200)).encode()
    rows, _, _ = prepare_records(payload)
    first = split_records(rows)
    assert first == split_records(rows)
    assert len(first["train"]) == 140
    assert len(first["validation"]) == len(first["test"]) == 30
    ids = [{row["sample_id"] for row in values} for values in first.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(set.union(*ids)) == len(rows)
    for partition in first.values():
        assert sum(row["label"] == "spam" for row in partition) / len(partition) == 0.2


def test_frozen_split_refuses_different_seed(tmp_path):
    payload = "\n".join(f"{'spam' if number % 5 == 0 else 'ham'}\tUnique message {number}" for number in range(100)).encode()
    rows, mapping, quality = prepare_records(payload)
    provenance = {"dataset_sha256": sha256_bytes(payload)}
    partitions = split_records(rows)
    manifest, digest = save_prepared_data(tmp_path, partitions, provenance, quality, mapping)
    assert len(digest) == 64
    assert json.loads((tmp_path / "split-manifest.json").read_text()) == manifest
    assert save_prepared_data(tmp_path, partitions, provenance, quality, mapping)[1] == digest
    with pytest.raises(ValueError, match="different frozen split"):
        save_prepared_data(tmp_path, split_records(rows, seed=1), provenance, quality, mapping, seed=1)


def test_archive_reads_only_exact_member_without_extracting_paths(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", "ham\thello")
        archive.writestr("../../injected.txt", "never extracted")
    assert read_archive(buffer.getvalue()) == b"ham\thello"
    assert not list(tmp_path.iterdir())


def test_archive_rejects_duplicate_expected_member():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", "ham\thello")
        with pytest.warns(UserWarning):
            archive.writestr("SMSSpamCollection", "spam\thello")
    with pytest.raises(ValueError, match="one bounded expected dataset"):
        read_archive(buffer.getvalue())
