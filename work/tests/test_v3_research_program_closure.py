"""Integrity tests for the WYH V3 research-program closure package.

Does not train a model, blend experts, create MODEL V3, or modify
prediction/prediction.csv. Verifies frozen closure invariants against
committed reports and machine-readable artifacts.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "experiments" / "evidence" / "v3"
MANIFEST = OUT / "v3_research_program_manifest.json"
METRICS = OUT / "v3_research_program_metrics.csv"
SUMMARY = ROOT / "work" / "reports" / "v3" / "v3_research_program_summary.md"
CONTRIB = ROOT / "work" / "docs" / "contributions" / "wyh_v3_contribution.md"
PRED = ROOT / "prediction" / "prediction.csv"
MODEL_V3_DOC = ROOT / "work" / "docs" / "versions" / "model_v3.md"
MODEL_V2_METRICS = ROOT / "work" / "experiments" / "evidence" / "metrics" / "model_v2_metrics.json"
E00T = OUT / "v3_e00t_metrics.json"
E07D = OUT / "v3_e07d_decision.json"
E06M_COMP = OUT / "v3_e06m_complementarity.json"

FORBIDDEN_ORACLE_AS_OOF = (
    "V3 achieved 87.86% OOF",
    "V3 reached 87.86% OOF",
    "MODEL V3 achieved 82.18%",
    "WYH achieved 87.86% OOF",
    "I achieved 87.86% accuracy",
    "our LZH model",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT)).decode()


def test_manifest_closure_invariants():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "COMPLETED_NO_MODEL_V3_PROMOTION"
    assert manifest["model_v3_created"] is False
    assert manifest["current_personal_deployable_model"] == "MODEL V2"
    assert manifest["current_personal_accuracy"] == 0.8212
    assert manifest["strongest_auditable_team_standalone"] == "LZH Prior-H"
    assert manifest["strongest_auditable_team_standalone_accuracy"] == 0.8266
    assert manifest["final_diagnostic_oracle"] == 0.8786
    assert manifest["oracle_correct"] == 4393
    assert manifest["oracle_total"] == 5000
    assert manifest["oracle_is_not_deployable_accuracy"] is True
    assert manifest["personal_version_decision"] == "MODEL V3 PROMOTION NOT JUSTIFIED"
    ids = [row["experiment_id"] for row in manifest["experiments"]]
    assert ids == [
        "V3-E00T",
        "V3-E02D",
        "V3-E03A",
        "V3-E04S",
        "V3-E05A",
        "V3-E06M",
        "V3-E07D",
    ]
    commits = {row["experiment_id"]: row["commit"] for row in manifest["experiments"]}
    assert commits["V3-E00T"] == "f117b29"
    assert commits["V3-E07D"] == "e7f94fd"
    assert manifest["experiments"][-1]["classification"] == (
        "MODEL V3 PROMOTION NOT JUSTIFIED"
    )


def test_metrics_table_matches_frozen_identities():
    rows = list(csv.DictReader(METRICS.open()))
    by_key = {(row["stage"], row["candidate"], row["metric"]): row for row in rows}

    v2 = by_key[("MODEL-V2", "MODEL V2 / V2-B-REFONLY", "accuracy")]
    assert float(v2["value"]) == 0.8212
    assert v2["deployable_or_diagnostic"] == "deployable"

    lzh = by_key[("TEAM-COMPARATOR", "LZH Prior-H", "accuracy")]
    assert float(lzh["value"]) == 0.8266
    assert "not WYH-owned" in lzh["interpretation"]

    oracle = by_key[
        ("V3-E06M", "five-expert pool including M2", "five_expert_diagnostic_oracle_accuracy")
    ]
    assert float(oracle["value"]) == 0.8786
    assert oracle["deployable_or_diagnostic"] == "diagnostic"

    m2 = by_key[("V3-E06M", "M2 source-balanced 0.5/0.5", "external_validation_accuracy")]
    assert float(m2["value"]) == 0.8218
    assert "not MODEL V3" in m2["interpretation"]

    net = by_key[("V3-E06M", "M2 vs M0", "net_corrections")]
    assert float(net["value"]) == 3.0

    sni_u = by_key[("V3-E04S", "SNI unique recoveries", "sni_new_unique_recoveries")]
    assert float(sni_u["value"]) == 53.0

    s0_u = by_key[("V3-E03A", "S0 unique recoveries", "unique_recoveries_beyond_LZH_WYH")]
    assert float(s0_u["value"]) == 96.0

    m2_u = by_key[("V3-E06M", "M2 new unique recoveries", "new_unique_beyond_four_expert_pool")]
    assert float(m2_u["value"]) == 29.0

    decision = by_key[("V3-E07D", "personal version decision", "model_v3_promotion")]
    assert float(decision["value"]) == 0.0
    assert "MODEL V3 PROMOTION NOT JUSTIFIED" in decision["interpretation"]


def test_frozen_json_identities_agree_with_closure_numbers():
    v2 = json.loads(MODEL_V2_METRICS.read_text())
    assert v2["oof_accuracy"] == 0.8212
    assert v2["correct"] == 4106

    e00t = json.loads(E00T.read_text())
    assert e00t["individual"]["lzh_prior_h"]["overall"]["accuracy"] == 0.8266
    assert e00t["individual"]["lzh_prior_h"]["overall"]["correct"] == 4133
    assert e00t["multi_expert_oracle"]["overall"]["oracle_correct"] == 4215
    assert e00t["multi_expert_oracle"]["overall"]["oracle_accuracy"] == 0.843

    e06m = json.loads(E06M_COMP.read_text())
    assert e06m["m2_five_expert_oracle_correct"] == 4393
    assert e06m["m2_new_unique_recoveries"] == 29
    assert e06m["sni_unique_recoveries"] == 53
    assert e06m["deltas"]["m2_vs_m0"]["overall"]["net_correction"] == 3

    e07d = json.loads(E07D.read_text())
    assert e07d["created_model_v3"] is False
    assert e07d["integrity"]["m0_correct"] == 4106
    assert e07d["integrity"]["m2_correct"] == 4109
    assert e07d["integrity"]["lzh_correct"] == 4133
    assert e07d["integrity"]["five_expert_oracle"] == 4393
    assert e07d["decisions"]["personal_version"] == "MODEL V3 PROMOTION NOT JUSTIFIED"
    assert abs(e07d["m2_vs_m0"]["mcnemar"]["pvalue"] - 0.8895) < 0.0001


def test_no_model_v3_document_or_tag():
    assert not MODEL_V3_DOC.is_file()
    tags = _git("tag", "--list", "model-v3").strip()
    assert tags == ""


def test_summary_does_not_misstate_oracle_as_oof():
    text = SUMMARY.read_text()
    for phrase in FORBIDDEN_ORACLE_AS_OOF:
        assert phrase not in text
    assert "MODEL V3 PROMOTION NOT JUSTIFIED" in text
    assert "NOT CREATED" in text
    assert "diagnostic oracle coverage ceiling" in text
    assert re.search(r"0\.8786", text)
    assert "87.86% OOF" not in text
    assert "V3 reached 87.86%" not in text


def test_contribution_file_states_lzh_boundary():
    text = CONTRIB.read_text()
    assert "LZH Prior-H" in text
    assert "frozen team comparator" in text
    assert "I did not develop this path" in text
    assert "Does not claim ownership of LZH Prior-H" in text or (
        "does not claim ownership of LZH Prior-H" in text
    )
    for phrase in FORBIDDEN_ORACLE_AS_OOF:
        assert phrase not in text
    assert "I achieved 87.86% accuracy" not in text


def test_prediction_csv_unchanged():
    status = _git("diff", "--", "prediction/prediction.csv")
    assert status == ""
    assert PRED.is_file()


def test_frozen_experiment_artifacts_unchanged_by_closure():
    frozen = _git("diff", "--", "prediction/prediction.csv")
    assert frozen == ""
    branch = _git("branch", "--show-current").strip()
    assert (
        branch == "ywan/ml-pipeline"
        or branch == "wyh"
        or branch.startswith("wyh-delivery-closure-")
    ), f"Unexpected branch for V3 integrity audit: {branch}"
