"""Official path, label-set, and manifest tests."""

from pathlib import Path

from merfish60.io import TARGET_COL, load_dataset
from merfish60.official_contract import (
    CANDIDATE_SUBMISSION_DIR_REL,
    OFFICIAL_COUNTS_TEST_REL,
    OFFICIAL_COUNTS_TRAIN_REL,
    OFFICIAL_DATA_RELS,
    OFFICIAL_EXAMPLE_PRED_REL,
    OFFICIAL_FINAL_SUBMISSION_REL,
    OFFICIAL_META_TEST_REL,
    OFFICIAL_META_TRAIN_REL,
    SUBMISSION_COLUMNS,
    YW002_FORBIDDEN_FIELDS,
    YW002_METADATA_FIELDS,
    allowed_labels,
    expected_test_cell_ids,
    expected_test_row_count,
    load_official_manifest,
    official_counts_test_path,
    official_counts_train_path,
    official_example_prediction_path,
    official_final_submission_path,
    official_meta_test_path,
    official_meta_train_path,
    sha256_file,
    verify_official_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_official_input_paths_exist():
    assert official_counts_train_path(ROOT).is_file()
    assert official_counts_test_path(ROOT).is_file()
    assert official_meta_train_path(ROOT).is_file()
    assert official_meta_test_path(ROOT).is_file()
    assert official_example_prediction_path(ROOT).is_file()
    rels = {
        OFFICIAL_COUNTS_TRAIN_REL,
        OFFICIAL_COUNTS_TEST_REL,
        OFFICIAL_META_TRAIN_REL,
        OFFICIAL_META_TEST_REL,
        OFFICIAL_EXAMPLE_PRED_REL,
        OFFICIAL_FINAL_SUBMISSION_REL,
        CANDIDATE_SUBMISSION_DIR_REL,
    }
    assert len(rels) >= 6


def test_allowed_labels_from_train_only():
    labels = allowed_labels(ROOT)
    data = load_dataset(ROOT)
    expected = sorted(data.y_train.astype(str).unique().tolist())
    assert labels == expected
    assert len(labels) == 60
    assert TARGET_COL == "MERFISH_cell_type_annotation"


def test_expected_test_ids_match_meta_test_order():
    ids = expected_test_cell_ids(ROOT)
    data = load_dataset(ROOT)
    assert ids == [str(v) for v in data.meta_test.index.tolist()]
    assert expected_test_row_count(ROOT) == len(data.meta_test)
    assert len(ids) == len(set(ids))


def test_manifest_integrity():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in m for m in messages)
    manifest = load_official_manifest(ROOT)
    assert manifest["target_column"] == TARGET_COL
    assert manifest["n_train_labels"] == 60
    for rel in OFFICIAL_DATA_RELS:
        entry = manifest["files"][rel]
        path = ROOT / rel
        assert entry["sha256"] == sha256_file(path)
        assert entry["byte_size"] == path.stat().st_size
        assert entry["n_data_rows"] == 5000


def test_yw002_field_lists():
    assert YW002_METADATA_FIELDS == ["Region", "Excitatory_vs_Inhibitory", "Segment"]
    for forbidden in (
        "Datasets",
        "volume",
        "center_x",
        "center_y",
        "Gender",
        "Mouse_ID",
        "AP_position",
        "Section_ID",
    ):
        assert forbidden in YW002_FORBIDDEN_FIELDS
        assert forbidden not in YW002_METADATA_FIELDS


def test_submission_headers_constant():
    assert SUBMISSION_COLUMNS == ["Cell_ID", "MERFISH_cell_type_annotation.y"]
