#!/usr/bin/env python3
"""V3-E02D: Privileged-Gene Dual-Level Distillation (PGD-200).

500-gene privileged teacher -> 200-gene deployable student.
Does not create a formal MODEL V3 release. Does not write prediction/prediction.csv.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import TARGET_COL, load_dataset, validate_contract  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402
from merfish60.reference import (  # noqa: E402
    EXPECTED_MD5,
    EXPECTED_N_OBS,
    EXPECTED_N_VARS,
    HISTORICAL_USABLE_ROWS,
    HISTORICAL_VECTOR_DUPES,
    audit_reference,
)
from merfish60.spatial_features import ei_of_label_from_train  # noqa: E402
from merfish60.team_cv import TEAM_CV_PROTOCOL, load_and_validate_team_folds  # noqa: E402
from merfish60.v2_metrics import (  # noqa: E402
    json_default,
    neuron_glial_masks,
    slice_metrics,
    write_json,
    write_proba,
)

EXPERIMENT_ID = "V3-E02D"
SEED = 20260820
TEMPERATURE = 2.0
LAMBDA_KD = 0.50
LAMBDA_REL = 0.25
MAX_EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN1 = 256
HIDDEN2 = 128
N_CLASSES = 60
N_TRAIN = 5000
STAGE_A_VAL_FRAC = 0.20
TEACHER_FOLDS = 3
EXPECTED_PYTHON_MARKER = "hackathon-v3"
LZH_WYH_ORACLE_CORRECT = 4215
SHARED_ERROR_N = 785
ORACLE_85_TARGET = 4250

WORK_DIR = ROOT / "work" / "v3_e02d"
OUT_DIR = ROOT / "outputs" / "v3"
REPORT_PATH = ROOT / "reports" / "v3" / "v3_e02d_privileged_gene_distillation.md"
REGISTRY_PATH = OUT_DIR / "v3_e00t_team_oof_registry.parquet"
PRED_PATH = ROOT / "prediction" / "prediction.csv"

CLASS_FAMILIES = OrderedDict(
    [
        (
            "oligodendrocyte_opc",
            [
                "oligodendrocyte_1",
                "oligodendrocyte_2",
                "oligodendrocyte_precursor_cell",
                "oligodendrocyte_progenitor_1",
                "oligodendrocyte_progenitor_2",
            ],
        ),
        ("astrocyte", ["astrocyte_1", "astrocyte_2"]),
        ("vascular", ["endothelial", "pericyte"]),
        ("meningeal", ["meninges_1", "meninges_2", "meninges_3"]),
        (
            "other_glial_non_neuronal",
            ["microglia", "ependymal", "Schwann_cell", "peripheral_glia"],
        ),
    ]
)
DIFFICULT_FAMILIES = (
    "oligodendrocyte_opc",
    "astrocyte",
    "vascular",
    "meningeal",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gene_list_sha256(genes: Sequence[str]) -> str:
    return hashlib.sha256(",".join(list(genes)).encode("utf-8")).hexdigest()


def family_of(label: str) -> str:
    for family, members in CLASS_FAMILIES.items():
        if label in members:
            return family
    return "neuronal_or_other"


def assert_v3_interpreter() -> None:
    resolved = str(Path(sys.executable).resolve())
    if EXPECTED_PYTHON_MARKER not in resolved and EXPECTED_PYTHON_MARKER not in sys.executable:
        raise SystemExit(
            "V3-E02D must run with the isolated hackathon-v3 interpreter, not {}. "
            "Use /Users/yyl/venvs/hackathon-v3/bin/python.".format(sys.executable)
        )
    if sys.version_info[:2] < (3, 11):
        raise SystemExit(
            "V3-E02D refuses Python {} (frozen MODEL V1/V2 environment).".format(
                platform.python_version()
            )
        )


def require_mps():
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "PYTORCH INSTALLED — MPS NEEDS REVIEW: mps.is_available() is False. "
            "Refusing to run V3-E02D on CPU."
        )
    return torch.device("mps")


def set_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def assert_work_dir_ignored(path: Path) -> None:
    try:
        out = subprocess.check_output(
            ["git", "check-ignore", "-v", str(path)],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "work directory is not gitignored: {} ({})".format(
                path, exc.output.decode("utf-8") if exc.output else "not ignored"
            )
        )
    if "work/v3_e02d" not in out:
        raise RuntimeError("unexpected check-ignore output for {}: {}".format(path, out))


def unique_recovery_mask(
    lzh_ok: np.ndarray, wyh_ok: np.ndarray, student_ok: np.ndarray
) -> np.ndarray:
    return (~np.asarray(lzh_ok, dtype=bool)) & (~np.asarray(wyh_ok, dtype=bool)) & np.asarray(
        student_ok, dtype=bool
    )


def oracle_union(*oks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(oks[0], dtype=bool)
    for ok in oks:
        out |= np.asarray(ok, dtype=bool)
    return out


def softmax_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def class_prototypes(latents: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float64)
    y = np.asarray(y)
    protos = np.zeros((n_classes, latents.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        mask = y == c
        if not np.any(mask):
            raise RuntimeError("class {} missing from teacher-training cells".format(c))
        protos[c] = latents[mask].mean(axis=0)
    return protos


def cosine_to_prototypes(latents: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    lat = np.asarray(latents, dtype=np.float64)
    proto = np.asarray(prototypes, dtype=np.float64)
    lat_n = lat / np.clip(np.linalg.norm(lat, axis=1, keepdims=True), 1e-8, None)
    proto_n = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8, None)
    rel = lat_n @ proto_n.T
    if rel.shape[1] != proto.shape[0]:
        raise RuntimeError("relation target width {} != n_classes".format(rel.shape[1]))
    return rel


def confidence_from_proba(proba: np.ndarray) -> dict:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    order = np.sort(p, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    entropy = -(p * np.log(p)).sum(axis=1)
    return {"top1": top1, "margin": top1 - top2, "entropy": entropy}


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
    proba: Optional[np.ndarray] = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    names = list(class_names)
    rec = recall_score(y_true, y_pred, labels=names, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=names, average=None, zero_division=0)
    payload = {
        "n": int(len(y_true)),
        "correct": int(np.sum(y_true == y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0)),
        "per_class_recall": {names[i]: float(rec[i]) for i in range(len(names))},
        "per_class_precision": {names[i]: float(prec[i]) for i in range(len(names))},
    }
    if proba is not None:
        payload["log_loss"] = float(log_loss(y_true, proba, labels=names))
    else:
        payload["log_loss"] = None
    return payload


def family_mean_recall(per_class_recall: Dict[str, float], family_members: Sequence[str]) -> float:
    vals = [float(per_class_recall[c]) for c in family_members if c in per_class_recall]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def family_block(per_class_recall: Dict[str, float]) -> dict:
    out = {}
    for name, members in CLASS_FAMILIES.items():
        out[name] = family_mean_recall(per_class_recall, members)
    out["difficult_family_mean_recall"] = float(
        np.mean([out[name] for name in DIFFICULT_FAMILIES])
    )
    return out


def privileged_signal_decision(a200: dict, a500: dict) -> dict:
    acc_delta = float(a500["accuracy"] - a200["accuracy"])
    f1_delta = float(a500["macro_f1"] - a200["macro_f1"])
    fam200 = a200["family_recall"]["difficult_family_mean_recall"]
    fam500 = a500["family_recall"]["difficult_family_mean_recall"]
    fam_delta = float(fam500 - fam200)
    family_deltas = {
        name: float(a500["family_recall"][name] - a200["family_recall"][name])
        for name in list(CLASS_FAMILIES) + ["difficult_family_mean_recall"]
    }
    material_collapse = acc_delta < -0.01 and f1_delta < -0.01
    criterion_a = acc_delta >= 0.005
    criterion_b = f1_delta >= 0.005
    criterion_c = fam_delta >= 0.010
    supported = (criterion_a or criterion_b or criterion_c) and not material_collapse
    return {
        "label": "PRIVILEGED SIGNAL SUPPORTED" if supported else "PRIVILEGED SIGNAL NOT SUPPORTED",
        "supported": bool(supported),
        "accuracy_delta": acc_delta,
        "macro_f1_delta": f1_delta,
        "difficult_family_mean_recall_delta": fam_delta,
        "family_deltas": family_deltas,
        "criterion_A_accuracy_plus_0_005": criterion_a,
        "criterion_B_macro_f1_plus_0_005": criterion_b,
        "criterion_C_difficult_family_recall_plus_0_010": criterion_c,
        "material_collapse": bool(material_collapse),
    }


def fold_slice_accuracy(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray) -> dict:
    out = {}
    for fold_id in range(5):
        mask = folds == fold_id
        out[str(fold_id)] = float(np.mean(y_true[mask] == y_pred[mask])) if mask.any() else None
    out["folds_0_2"] = float(np.mean(y_true[folds <= 2] == y_pred[folds <= 2]))
    out["folds_3_4"] = float(np.mean(y_true[folds >= 3] == y_pred[folds >= 3]))
    return out


def complementarity_block(
    y_true: np.ndarray,
    lzh_pred: np.ndarray,
    wyh_pred: np.ndarray,
    student_pred: np.ndarray,
    folds: np.ndarray,
) -> dict:
    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    st_ok = student_pred == y_true
    rec = unique_recovery_mask(lzh_ok, wyh_ok, st_ok)
    both_wrong = (~lzh_ok) & (~wyh_ok)
    lzh_st = oracle_union(lzh_ok, st_ok)
    wyh_st = oracle_union(wyh_ok, st_ok)
    three = oracle_union(lzh_ok, wyh_ok, st_ok)
    pair = oracle_union(lzh_ok, wyh_ok)

    def _acc(mask: np.ndarray, ok: np.ndarray) -> float:
        return float(np.mean(ok[mask])) if mask.any() else float("nan")

    mask_34 = folds >= 3
    return {
        "student_only_correct": int(np.sum(st_ok & (~lzh_ok) & (~wyh_ok))),
        "student_correct_when_LZH_wrong": int(np.sum(st_ok & (~lzh_ok))),
        "student_correct_when_WYH_wrong": int(np.sum(st_ok & (~wyh_ok))),
        "student_correct_when_both_wrong": int(np.sum(rec)),
        "student_wrong_when_both_correct": int(np.sum((~st_ok) & lzh_ok & wyh_ok)),
        "all_three_wrong": int(np.sum((~lzh_ok) & (~wyh_ok) & (~st_ok))),
        "student_lzh_disagreement": int(np.sum(student_pred != lzh_pred)),
        "student_wyh_disagreement": int(np.sum(student_pred != wyh_pred)),
        "new_unique_recoveries": int(np.sum(rec)),
        "shared_error_n": int(np.sum(both_wrong)),
        "lzh_student_oracle_correct": int(lzh_st.sum()),
        "lzh_student_oracle_accuracy": float(np.mean(lzh_st)),
        "wyh_student_oracle_correct": int(wyh_st.sum()),
        "wyh_student_oracle_accuracy": float(np.mean(wyh_st)),
        "lzh_wyh_oracle_correct": int(pair.sum()),
        "lzh_wyh_oracle_accuracy": float(np.mean(pair)),
        "three_expert_oracle_correct": int(three.sum()),
        "three_expert_oracle_accuracy": float(np.mean(three)),
        "incremental_oracle_gain_over_lzh_wyh": int(three.sum() - pair.sum()),
        "three_expert_oracle_correct_folds_3_4": int(np.sum(three[mask_34])),
        "three_expert_oracle_accuracy_folds_3_4": _acc(mask_34, three),
        "new_unique_recoveries_folds_3_4": int(np.sum(rec & mask_34)),
        "oracle_identity_check": int(pair.sum() + np.sum(rec)) == int(three.sum()),
    }


def experiment_classification(stage_a_supported: bool, rows: Dict[str, dict]) -> Tuple[str, str]:
    s0, s1, s2 = rows["S0"], rows["S1"], rows["S2"]
    s2_unique = s2["new_unique_recoveries"]
    oracle_all = s2["three_expert_oracle_accuracy"]
    oracle_34 = s2["three_expert_oracle_accuracy_folds_3_4"]
    s2_beyond = s2_unique >= s1["new_unique_recoveries"] and s2_unique >= s0["new_unique_recoveries"]
    f1_collapse = (s2["macro_f1"] - s0["macro_f1"]) < -0.02
    hard_collapse = (
        s2.get("hard_bucket_accuracy") is not None
        and s0.get("hard_bucket_accuracy") is not None
        and (s2["hard_bucket_accuracy"] - s0["hard_bucket_accuracy"]) < -0.03
    )
    locked_regress = (s2["folds_3_4"] - max(s0["folds_3_4"], s1["folds_3_4"])) < -0.015
    complementary = max(s0["new_unique_recoveries"], s1["new_unique_recoveries"], s2_unique) >= 5
    distillation_helps = (
        s1["new_unique_recoveries"] > s0["new_unique_recoveries"]
        or s2_unique > s0["new_unique_recoveries"]
        or s1["accuracy"] > s0["accuracy"] + 0.002
        or s2["accuracy"] > s0["accuracy"] + 0.002
    )
    s2_fails_baselines = s2_unique < s0["new_unique_recoveries"] and s2["folds_3_4"] < s0["folds_3_4"]

    if (not stage_a_supported) or (not complementary) or locked_regress or s2_fails_baselines:
        return (
            "FAILED TO ADD NEW INFORMATION",
            "Stage A failed, S1/S2 did not add complementary recoveries, "
            "S2 lost to simpler baselines, or locked folds 3-4 materially regressed.",
        )
    if (
        stage_a_supported
        and s2_beyond
        and distillation_helps
        and s2_unique >= 35
        and oracle_all >= 0.85
        and oracle_34 >= 0.85
        and not f1_collapse
        and not hard_collapse
    ):
        return (
            "STRONG NOVEL EXPERT",
            "Privileged signal supported, S2 added value beyond S0/S1, unique recoveries "
            "reached the 85% oracle milestone, and locked folds 3-4 also reached 0.85.",
        )
    if stage_a_supported and complementary:
        extra = []
        if not s2_beyond:
            extra.append("S2 did not improve unique recoveries over S0/S1")
        if s2_unique < 35:
            extra.append("new_unique_recoveries < 35")
        if oracle_all < 0.85:
            extra.append("three-expert oracle < 0.85")
        if oracle_34 < 0.85:
            extra.append("locked folds 3-4 oracle < 0.85")
        if f1_collapse:
            extra.append("macro-F1 collapsed versus S0")
        detail = "; ".join(extra) if extra else "the dual-level hypothesis was not confirmed"
        return (
            "PROMISING BUT INSUFFICIENT",
            "Privileged 500-gene signal is supported and the 200-gene student recovers some "
            "LZH+WYH shared errors, but {}. Dual-level distillation is therefore not a mature "
            "third expert.".format(detail),
        )
    return (
        "FAILED TO ADD NEW INFORMATION",
        "S1/S2 did not provide meaningful complementary information relative to LZH + WYH.",
    )


def build_model_classes(torch, nn):
    class Backbone(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, HIDDEN1),
                nn.LayerNorm(HIDDEN1),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN1, HIDDEN2),
                nn.LayerNorm(HIDDEN2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )

        def forward(self, x):
            return self.net(x)

    class TeacherMLP(nn.Module):
        def __init__(self, in_dim: int, n_classes: int = N_CLASSES):
            super().__init__()
            self.in_dim = int(in_dim)
            self.backbone = Backbone(in_dim)
            self.head = nn.Linear(HIDDEN2, n_classes)

        def forward(self, x):
            latent = self.backbone(x)
            return self.head(latent), latent

    class StudentMLP(nn.Module):
        def __init__(self, in_dim: int = 200, n_classes: int = N_CLASSES):
            super().__init__()
            self.in_dim = int(in_dim)
            self.backbone = Backbone(in_dim)
            self.class_head = nn.Linear(HIDDEN2, n_classes)
            self.relation_head = nn.Linear(HIDDEN2, n_classes)

        def forward(self, x):
            if x.shape[1] != self.in_dim:
                raise RuntimeError(
                    "student input dim {} != {}".format(x.shape[1], self.in_dim)
                )
            latent = self.backbone(x)
            return self.class_head(latent), self.relation_head(latent), latent

    return TeacherMLP, StudentMLP


def _batches(n: int, batch_size: int, rng: np.random.RandomState) -> Iterable[np.ndarray]:
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def evaluate_model(torch, model, x: np.ndarray, y: Optional[np.ndarray], device) -> Tuple:
    model.eval()
    logits_all = []
    latents_all = []
    rel_all = []
    x_t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = x_t[start : start + BATCH_SIZE].to(device)
            out = model(xb)
            logits_all.append(out[0].detach().cpu().numpy())
            if len(out) == 2:
                latents_all.append(out[1].detach().cpu().numpy())
            else:
                rel_all.append(out[1].detach().cpu().numpy())
                latents_all.append(out[2].detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    latents = np.concatenate(latents_all, axis=0)
    relation = np.concatenate(rel_all, axis=0) if rel_all else None
    proba = softmax_from_logits(logits, 1.0)
    acc = None
    if y is not None:
        pred = np.argmax(proba, axis=1)
        acc = float(np.mean(pred == y))
    return acc, proba, {"logits": logits, "latents": latents, "relation": relation}


def train_model(
    *,
    torch,
    model,
    x: np.ndarray,
    y: np.ndarray,
    val_x: Optional[np.ndarray],
    val_y: Optional[np.ndarray],
    device,
    epochs: int,
    patience: int,
    seed: int,
    teacher_logits: Optional[np.ndarray] = None,
    teacher_relation: Optional[np.ndarray] = None,
    lambda_kd: float = 0.0,
    lambda_rel: float = 0.0,
    temperature: float = TEMPERATURE,
    retrain_exact_epochs: Optional[int] = None,
    log_name: str = "model",
) -> dict:
    import torch.nn.functional as F

    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.RandomState(seed)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    t_logits = (
        torch.as_tensor(teacher_logits, dtype=torch.float32) if teacher_logits is not None else None
    )
    t_rel = (
        torch.as_tensor(teacher_relation, dtype=torch.float32) if teacher_relation is not None else None
    )
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    n_epochs = retrain_exact_epochs if retrain_exact_epochs is not None else epochs
    use_es = retrain_exact_epochs is None and val_x is not None

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for idx in _batches(len(y), BATCH_SIZE, rng):
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            logits = out[0]
            loss = F.cross_entropy(logits, yb)
            if lambda_kd > 0:
                assert t_logits is not None
                student_log = F.log_softmax(logits / temperature, dim=1)
                teacher_t = F.softmax(t_logits[idx].to(device) / temperature, dim=1)
                kd = F.kl_div(student_log, teacher_t, reduction="batchmean")
                loss = loss + lambda_kd * (temperature ** 2) * kd
            if lambda_rel > 0:
                assert t_rel is not None
                loss = loss + lambda_rel * F.smooth_l1_loss(out[1], t_rel[idx].to(device))
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(idx)
            seen += len(idx)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1)}
        val_acc = None
        if val_x is not None:
            val_acc, _, _ = evaluate_model(torch, model, val_x, val_y, device)
            row["val_acc"] = val_acc
        history.append(row)
        print(
            "    {} epoch {}/{} train_loss={:.4f} val_acc={}".format(
                log_name,
                epoch,
                n_epochs,
                row["train_loss"],
                "{:.4f}".format(val_acc) if val_acc is not None else "n/a",
            ),
            flush=True,
        )
        if use_es:
            if val_acc > best_acc + 1e-12:
                best_acc = float(val_acc)
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        else:
            best_epoch = epoch
            best_acc = float(val_acc) if val_acc is not None else best_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = n_epochs
    model.load_state_dict(best_state)
    model.to(device)
    return {
        "best_epoch": int(best_epoch),
        "best_val_acc": None if best_acc < 0 else float(best_acc),
        "history": history,
        "n_epochs_run": int(len(history)),
    }


def fit_with_early_stopping_then_full(
    *,
    torch,
    factory,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device,
    seed: int,
    log_name: str,
    teacher_logits: Optional[np.ndarray] = None,
    teacher_relation: Optional[np.ndarray] = None,
    lambda_kd: float = 0.0,
    lambda_rel: float = 0.0,
    checkpoint: Optional[Path] = None,
    full_idx: Optional[np.ndarray] = None,
) -> Tuple[object, dict]:
    model = factory()
    es = train_model(
        torch=torch,
        model=model,
        x=x[train_idx],
        y=y[train_idx],
        val_x=x[val_idx],
        val_y=y[val_idx],
        device=device,
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=seed,
        teacher_logits=None if teacher_logits is None else teacher_logits[train_idx],
        teacher_relation=None if teacher_relation is None else teacher_relation[train_idx],
        lambda_kd=lambda_kd,
        lambda_rel=lambda_rel,
        log_name=log_name + "/es",
    )
    best_epoch = max(int(es["best_epoch"]), 1)
    if full_idx is None:
        full_idx = np.arange(len(y))
    full = factory()
    full_fit = train_model(
        torch=torch,
        model=full,
        x=x[full_idx],
        y=y[full_idx],
        val_x=None,
        val_y=None,
        device=device,
        epochs=best_epoch,
        patience=PATIENCE,
        seed=seed,
        teacher_logits=None if teacher_logits is None else teacher_logits[full_idx],
        teacher_relation=None if teacher_relation is None else teacher_relation[full_idx],
        lambda_kd=lambda_kd,
        lambda_rel=lambda_rel,
        retrain_exact_epochs=best_epoch,
        log_name=log_name + "/full",
    )
    if checkpoint is not None:
        torch.save({"state_dict": full.state_dict(), "in_dim": getattr(full, "in_dim", None)}, checkpoint)
    return full, {"early_stopping": es, "full_retrain": full_fit, "selected_epochs": best_epoch}


def environment_payload(torch, device) -> dict:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform_machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "dropout": DROPOUT,
        "temperature": TEMPERATURE,
        "lambda_kd": LAMBDA_KD,
        "lambda_rel": LAMBDA_REL,
        "preprocessing": "log1p(raw counts); no library-size normalization; no feature scaling",
        "created_at_utc": utc_now(),
    }


def load_usable_reference(data, class_names: Sequence[str]) -> dict:
    audit = audit_reference(data, class_names, root=ROOT)
    if audit["md5"] != EXPECTED_MD5:
        raise RuntimeError("reference MD5 mismatch")
    if audit["raw_n_obs"] != EXPECTED_N_OBS or audit["raw_n_vars"] != EXPECTED_N_VARS:
        raise RuntimeError("reference raw shape {} x {} does not match 146621 x 500".format(
            audit["raw_n_obs"], audit["raw_n_vars"]
        ))
    if audit["n_usable_reference_rows"] != HISTORICAL_USABLE_ROWS:
        raise RuntimeError(
            "usable reference {} != historical {}".format(
                audit["n_usable_reference_rows"], HISTORICAL_USABLE_ROWS
            )
        )
    if audit["n_exact_vector_duplicates_removed"] != HISTORICAL_VECTOR_DUPES:
        raise RuntimeError(
            "vector duplicates {} != historical {}".format(
                audit["n_exact_vector_duplicates_removed"], HISTORICAL_VECTOR_DUPES
            )
        )
    adata = audit["_adata"]
    genes500 = [str(v) for v in adata.var_names.tolist()]
    genes200 = list(data.counts_train.columns)
    if len(genes200) != 200:
        raise RuntimeError("official gene list length {}".format(len(genes200)))
    missing = [g for g in genes200 if g not in set(genes500)]
    if missing:
        raise RuntimeError("official 200 genes not subset of 500: {}".format(missing[:10]))
    keep = audit["_is_ref"]
    x500_all = adata.X
    if sp.issparse(x500_all):
        x500_all = x500_all.toarray()
    x500_all = np.asarray(x500_all, dtype=np.float32)
    idx200 = np.array([genes500.index(g) for g in genes200], dtype=np.int64)
    x200_from_500 = x500_all[:, idx200]
    x200_audit = np.asarray(audit["_x200"], dtype=np.float32)
    if not np.array_equal(x200_from_500, x200_audit):
        raise RuntimeError("200-gene slice from 500-gene matrix does not match audit alignment")
    ids = np.array(audit["_ext_ids"][keep].tolist(), dtype=object)
    y_codes = np.asarray(audit["_y_codes"][keep], dtype=np.int64)
    if (y_codes < 0).any():
        raise RuntimeError("usable reference contains unmapped labels")
    train_ids = set(str(v) for v in data.counts_train.index)
    test_ids = set(str(v) for v in data.counts_test.index)
    if set(ids.tolist()) & train_ids:
        raise RuntimeError("usable reference overlaps competition train Cell_IDs")
    if set(ids.tolist()) & test_ids:
        raise RuntimeError("usable reference overlaps competition test Cell_IDs")
    payload = {
        "audit": {k: v for k, v in audit.items() if not str(k).startswith("_")},
        "ids": ids,
        "y_codes": y_codes,
        "X200": np.log1p(x200_audit[keep]),
        "X500": np.log1p(x500_all[keep]),
        "genes200": genes200,
        "genes500": genes500,
        "gene200_sha256": gene_list_sha256(genes200),
        "gene500_sha256": gene_list_sha256(genes500),
        "subset_ok": True,
    }
    del audit["_adata"]
    return payload


def glial_family_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names, ei_of_label) -> dict:
    neuron, glial = neuron_glial_masks(y_true, class_names, ei_of_label)
    return {
        "neuron_n": int(neuron.sum()),
        "neuron_accuracy": float(np.mean(y_true[neuron] == y_pred[neuron])) if neuron.any() else None,
        "glial_n": int(glial.sum()),
        "glial_accuracy": float(np.mean(y_true[glial] == y_pred[glial])) if glial.any() else None,
    }


def write_validation_csv(path: Path, cell_ids, y_true, y_pred, folds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cell_ID": [str(v) for v in cell_ids],
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "canonical_fold": [int(v) for v in folds],
            "evaluation": "honest_external_validation_not_competition_oof",
        }
    ).to_csv(path, index=False)


def next_action_for(label: str) -> str:
    if label == "STRONG NOVEL EXPERT":
        return (
            "Keep S2 frozen as a candidate third expert and, in a separate task, "
            "run a predeclared reliability-gated routing experiment over LZH + WYH + S2 "
            "without changing S2 coefficients."
        )
    if label == "PROMISING BUT INSUFFICIENT":
        return (
            "Do not freeze MODEL V3 and do not start routing on S2. The next experiment should "
            "audit the simpler 200-gene hard-label reference MLP (S0) as a candidate third expert, "
            "because S0 recovered more shared errors than S1/S2 and dual-level distillation did not "
            "transfer the Stage A 500-gene advantage."
        )
    return (
        "Stop this distillation line. Return to a different independent-expert source "
        "or wait for auditable YHH / team-main cell-level OOF artifacts rather than "
        "tuning T / lambda_kd / lambda_rel on the same folds."
    )


def render_report(payload: dict) -> str:
    a = payload.get("stage_a") or {}
    decision = a.get("decision") or {}
    tch = payload.get("teacher") or {}
    students = payload.get("students") or {}
    comp = payload.get("complementarity") or {}
    s2 = students.get("S2") or {}
    s1 = students.get("S1") or {}
    s0 = students.get("S0") or {}
    final = payload["final_classification"]
    rec = payload.get("recoveries") or {}
    env = payload["environment"]
    ref = payload["reference"]
    leak = payload["leakage_audit"]

    def _fmt_student(tag: str, row: dict) -> str:
        if not row:
            return "{} was not trained.".format(tag)
        return (
            "{tag} honest external-validation accuracy **{acc:.4f}** ({corr} / 5000), "
            "macro-F1 **{f1:.4f}**, log loss **{ll:.4f}**. "
            "Canonical folds 0-2: **{a02:.4f}**. Locked folds 3-4: **{a34:.4f}**."
        ).format(
            tag=tag,
            acc=row["accuracy"],
            corr=row["correct"],
            f1=row["macro_f1"],
            ll=row["log_loss"],
            a02=row["folds_0_2"],
            a34=row["folds_3_4"],
        )

    stage_a_section = "Stage A was not completed."
    if a:
        stage_a_section = """A200 (200 genes) validation accuracy **{a200a:.4f}**, macro-F1 **{a200f:.4f}**, log loss **{a200l:.4f}**.
A500 (500 genes) validation accuracy **{a500a:.4f}**, macro-F1 **{a500f:.4f}**, log loss **{a500l:.4f}**.

Deltas (A500 − A200): accuracy **{da:.4f}**, macro-F1 **{df:.4f}**, difficult-family mean recall **{dd:.4f}**.

| Family | A200 recall | A500 recall | delta |
|---|---:|---:|---:|
{fam_rows}

Decision: **{lab}**.
Criteria: A (acc +0.005)={ca}, B (macro-F1 +0.005)={cb}, C (difficult-family recall +0.010)={cc}. Material collapse={mc}.
""".format(
            a200a=a["A200"]["accuracy"],
            a200f=a["A200"]["macro_f1"],
            a200l=a["A200"]["log_loss"],
            a500a=a["A500"]["accuracy"],
            a500f=a["A500"]["macro_f1"],
            a500l=a["A500"]["log_loss"],
            da=decision["accuracy_delta"],
            df=decision["macro_f1_delta"],
            dd=decision["difficult_family_mean_recall_delta"],
            fam_rows="\n".join(
                "| {name} | {b:.4f} | {c:.4f} | {d:.4f} |".format(
                    name=name,
                    b=a["A200"]["family_recall"][name],
                    c=a["A500"]["family_recall"][name],
                    d=decision["family_deltas"][name],
                )
                for name in list(CLASS_FAMILIES) + ["difficult_family_mean_recall"]
            ),
            lab=decision["label"],
            ca=decision["criterion_A_accuracy_plus_0_005"],
            cb=decision["criterion_B_macro_f1_plus_0_005"],
            cc=decision["criterion_C_difficult_family_recall_plus_0_010"],
            mc=decision["material_collapse"],
        )

    later = "Not run: Stage A did not support a privileged 500-gene signal."
    if payload.get("ran_stage_bc"):
        later = "Completed as predeclared."

    rec_classes = rec.get("top_recovered_classes") or []
    rec_class_txt = ", ".join(
        "{k} (n={n})".format(k=r["true_label"], n=r["n"]) for r in rec_classes[:8]
    ) or "none"
    rec_fam_txt = ", ".join(
        "{k} (n={n})".format(k=r["family"], n=r["n"])
        for r in (rec.get("by_family") or [])
    ) or "none"

    return """# V3-E02D — Privileged-Gene Dual-Level Distillation

Research codename: **PGD-200**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can biological information in the 300 reference-only genes of the approved same-study 500-gene MERFISH deposit be transferred into a **deployable student that sees only the official 200 genes** at competition inference time?

The proposed transfer uses a 500-gene privileged teacher and two aligned targets:

1. 60-class soft probability structure (temperature-scaled KL)
2. class-aligned latent biological relation structure (cosine similarity to 60 named class prototypes)

Primary goal: recover cells that **both LZH Prior-H and WYH MODEL V2 currently miss**, not merely raise standalone accuracy.

## 2. Motivation from V3-E00T

V3-E00T established the currently auditable honest expert pool:

- LZH `depth_masked_prior_h_anchor`: 0.8266 (4133 / 5000)
- WYH MODEL V2: 0.8212 (4106 / 5000)
- two-expert diagnostic oracle: **4215 / 5000 = 0.8430**
- both-wrong population: **785**
- 85% diagnostic target: 4250 / 5000, so a new expert must recover **at least 35** of those 785 cells

YHH V7 and current team-main were not used: they lack auditable cell-level OOF artifacts. Their reported scores were not inferred.

Diagnostic oracle is **not** deployable model accuracy.

## 3. Novelty Relative to Existing Team Work

This is a **new methodological direction within the team/project**. It is **not** a claim that knowledge distillation is globally novel.

No auditable team method currently implements the same mechanism:

`500-gene privileged teacher → 200-gene deployable student → soft-logit distillation + class-aligned latent-relation distillation`

Nearby but non-equivalent methods:

- WYH MODEL V2 uses only the official 200 genes from the same 500-gene deposit
- LZH Prior-H / graph stacker use biology priors and stacking on competition-scale features
- LZH gene-token KL is same-input clean-vs-corrupt consistency, not 500-to-200 privileged transfer
- YHH / team-main combine reference LightGBM, specialists, and spatial/expression neighbors

## 4. External Reference Provenance

| Item | Value |
|---|---|
| Source | Zenodo record 18039571 |
| File | `work/external/MERFISH_spinal_cord_resolved_0718.h5ad` |
| MD5 | `{md5}` |
| Raw shape | {raw_n} cells × {raw_g} genes |
| Usable reference | {usable} |
| Official 200-gene SHA256 | `{g200}` |
| Teacher 500-gene SHA256 | `{g500}` |
| 200 ⊂ 500 | {subset} |

## 5. Exclusion / Leakage Audit

Removed {n_train} competition train Cell_IDs, {n_test} competition test Cell_IDs, and {n_dup} exact aligned 200-gene count-vector duplicates. Historical targets (5000 / 5000 / 47 / 136574) reproduced exactly.

Usable reference Cell_IDs are disjoint from competition train and test IDs. Competition test labels were not used. Students are trained only on approved external-reference cells; competition-train scores are **honest external-validation predictions**, not conventional competition OOF.

## 6. Stage A — 200 vs 500 Gene Signal

{stage_a}

## 7. 500-Gene Cross-Fitted Teacher

{teacher_section}

## 8. Class-Aligned Relational Representation

Raw 128-d latents from independently trained fold-teachers are **not** matched. Independently trained networks can rotate or permute latent coordinates, so a direct SmoothL1 / MSE on those vectors would mix incompatible axes.

Instead, each teacher fold:

1. embeds its teacher-training reference cells
2. computes one 128-d prototype per official class
3. stores held-out cosine similarities to those 60 named prototypes

The resulting 60-d relation target is aligned across folds because each dimension is a named cell class in canonical order.

## 9. Student Ablations

All students see **only official 200 genes**.

- **S0** hard-label CE student (architecture control)
- **S1** CE + λ_kd T² KL(teacher_T || student_T), T=2.0, λ_kd=0.50
- **S2** S1 plus λ_rel SmoothL1 on the class-aligned relation head, λ_rel=0.25

No coefficient, seed, or architecture search was performed. Competition prediction uses only the student classification head.

## 10. Competition External-Validation Results

{s0txt}

{s1txt}

{s2txt}

These numbers are honest external validation: the student never trained on competition labels.

## 11. Canonical Folds 0-2 vs Locked Folds 3-4

Canonical seed42 folds remain analysis partitions only. Model definitions were not changed after seeing folds 3-4.

| Variant | folds 0-2 | folds 3-4 |
|---|---:|---:|
| S0 | {s0_02} | {s0_34} |
| S1 | {s1_02} | {s1_34} |
| S2 | {s2_02} | {s2_34} |

## 12. Complementarity with LZH and WYH

Reverified LZH + WYH oracle: {pair_n} / 5000 = {pair_acc:.4f}; both-wrong = {both_wrong}.

| Combo | Oracle correct | Oracle accuracy |
|---|---:|---:|
| LZH + S0 | {lzh_s0_n} | {lzh_s0_a:.4f} |
| WYH + S0 | {wyh_s0_n} | {wyh_s0_a:.4f} |
| LZH + WYH + S0 | {three_s0_n} | {three_s0_a:.4f} |
| LZH + S2 | {lzh_s2_n} | {lzh_s2_a:.4f} |
| WYH + S2 | {wyh_s2_n} | {wyh_s2_a:.4f} |
| LZH + WYH + S2 | {three_n} | {three_a:.4f} |

Derived, not trained: LZH + WYH oracle = 4215 / 5000; S0 new_unique_recoveries = 96; LZH + WYH + S0 diagnostic oracle = **4311 / 5000 = 0.8622**. This is a derived oracle result, **NOT** a deployable model score.

## 13. Recovery of the 785 Shared Errors

`new_unique_recoveries` is defined exactly as LZH wrong AND WYH wrong AND student correct.

- S0 unique recoveries: **{s0u}**
- S1 unique recoveries: **{s1u}**
- S2 unique recoveries: **{s2u}**

4215 + S0 unique recoveries = **{oracle_from_s0}**.

4215 + S2 unique recoveries = **{oracle_from_rec}**.

## 14. Three-Expert Diagnostic Oracle

LZH + WYH + S0 diagnostic oracle: **4311 / 5000 = 0.8622**.

LZH + WYH + S2 diagnostic oracle: **{three_n} / 5000 = {three_a:.4f}**.

Canonical folds 3-4 LZH + WYH + S2 oracle: **{three_34_n} / {n34} = {three_34_a:.4f}**.

Does the overall three-expert diagnostic oracle reach ≥ 0.85? **{reach85}**.

Remaining all-three-wrong cells: **{all_wrong}**.

**DIAGNOSTIC ORACLE != DEPLOYABLE MODEL ACCURACY.**

Although privileged distillation failed to outperform the hard-label student, S0 provides stronger unique complementarity than S1/S2 and therefore becomes the preferred candidate third expert for the next rescue audit.

## 15. Biological / Confusion-Family Analysis

S2 unique recoveries by true family: {rec_fam}

Top newly recovered true classes: {rec_classes}

This is the evidence for whether privileged representation transfer helped the current shared-hard cases, especially oligodendrocyte / OPC, astrocyte, vascular, and meningeal families.

## 16. Evidence for or Against Dual-Level Distillation

Stage A privileged signal: **{stage_a_label}**.

Unique-recovery ordering S0 / S1 / S2: {s0u} / {s1u} / {s2u}.

The strongest project-level evidence would have been S0 < S1 < S2 unique recoveries and a three-expert oracle ≥ 0.85. That pattern is **{pattern}**.

Although privileged distillation failed to outperform the hard-label student, S0 provides stronger unique complementarity than S1/S2 and therefore becomes the preferred candidate third expert for the next rescue audit. LZH + WYH + S0 diagnostic oracle = **4311 / 5000 = 0.8622**. This is a derived oracle result, not a deployable model score.

## 17. Leakage Audit

{leak_txt}

## 18. Limitations

- Teacher and student are modest MLPs with one predeclared budget; this is not an exhaustive neural architecture search
- Early stopping uses a fixed 80/20 reference split; students are then retrained on the full usable reference for the selected epoch count
- Canonical folds 3-4 are a locked confirmation partition for this experiment, not LZH's original 3-fold protocol
- YHH and team-main remain unavailable for honest complementarity
- A diagnostic oracle is not a deployable ensemble

## 19. Experiment Decision

**{final}**

{reason}

Recommended next action (not started): {next_action}

Reproducibility: Python {py}, torch {torch}, device {dev}, seed {seed}, reference MD5 {md5}.
""".format(
        md5=ref["md5"],
        raw_n=ref["raw_n_obs"],
        raw_g=ref["raw_n_vars"],
        usable=ref["n_usable"],
        g200=ref["gene200_sha256"],
        g500=ref["gene500_sha256"],
        subset=ref["subset_ok"],
        n_train=ref["n_train_id_overlaps_removed"],
        n_test=ref["n_test_id_overlaps_removed"],
        n_dup=ref["n_exact_vector_duplicates_removed"],
        stage_a=stage_a_section,
        teacher_section=(
            "Protocol: StratifiedKFold(n_splits=3, shuffle=True, random_state=20260820) on the usable "
            "reference. Cross-fitted teacher accuracy on held-out reference cells: **{acc:.4f}** "
            "({corr} / {n}), macro-F1 **{f1:.4f}**, log loss **{ll:.4f}**. Relation targets have shape "
            "(n_usable, 60).".format(
                acc=tch.get("crossfit_accuracy", float("nan")),
                corr=tch.get("crossfit_correct", "n/a"),
                n=ref["n_usable"],
                f1=tch.get("crossfit_macro_f1", float("nan")),
                ll=tch.get("crossfit_log_loss", float("nan")),
            )
            if payload.get("ran_stage_bc")
            else later
        ),
        s0txt=_fmt_student("S0", s0) if payload.get("ran_stage_bc") else later,
        s1txt=_fmt_student("S1", s1) if payload.get("ran_stage_bc") else later,
        s2txt=_fmt_student("S2", s2) if payload.get("ran_stage_bc") else later,
        s0_02="{:.4f}".format(s0["folds_0_2"]) if s0 else "n/a",
        s0_34="{:.4f}".format(s0["folds_3_4"]) if s0 else "n/a",
        s1_02="{:.4f}".format(s1["folds_0_2"]) if s1 else "n/a",
        s1_34="{:.4f}".format(s1["folds_3_4"]) if s1 else "n/a",
        s2_02="{:.4f}".format(s2["folds_0_2"]) if s2 else "n/a",
        s2_34="{:.4f}".format(s2["folds_3_4"]) if s2 else "n/a",
        pair_n=comp.get("lzh_wyh_oracle_correct", LZH_WYH_ORACLE_CORRECT),
        pair_acc=comp.get("lzh_wyh_oracle_accuracy", LZH_WYH_ORACLE_CORRECT / N_TRAIN),
        both_wrong=comp.get("shared_error_n", SHARED_ERROR_N),
        lzh_s2_n=s2.get("lzh_student_oracle_correct", "n/a"),
        lzh_s2_a=s2.get("lzh_student_oracle_accuracy", float("nan")),
        wyh_s2_n=s2.get("wyh_student_oracle_correct", "n/a"),
        wyh_s2_a=s2.get("wyh_student_oracle_accuracy", float("nan")),
        lzh_s0_n=s0.get("lzh_student_oracle_correct", "n/a"),
        lzh_s0_a=s0.get("lzh_student_oracle_accuracy", float("nan")),
        wyh_s0_n=s0.get("wyh_student_oracle_correct", "n/a"),
        wyh_s0_a=s0.get("wyh_student_oracle_accuracy", float("nan")),
        three_s0_n=s0.get("three_expert_oracle_correct", "n/a"),
        three_s0_a=s0.get("three_expert_oracle_accuracy", float("nan")),
        three_n=s2.get("three_expert_oracle_correct", "n/a"),
        three_a=s2.get("three_expert_oracle_accuracy", float("nan")),
        s0u=s0.get("new_unique_recoveries", "n/a"),
        s1u=s1.get("new_unique_recoveries", "n/a"),
        s2u=s2.get("new_unique_recoveries", "n/a"),
        oracle_from_s0=(
            LZH_WYH_ORACLE_CORRECT + int(s0.get("new_unique_recoveries", 0))
            if s0
            else "n/a"
        ),
        oracle_from_rec=(
            LZH_WYH_ORACLE_CORRECT + int(s2.get("new_unique_recoveries", 0))
            if s2
            else "n/a"
        ),
        three_34_n=s2.get("three_expert_oracle_correct_folds_3_4", "n/a"),
        n34=2000,
        three_34_a=s2.get("three_expert_oracle_accuracy_folds_3_4", float("nan")),
        reach85=(
            "YES" if s2 and s2.get("three_expert_oracle_accuracy", 0) >= 0.85 else "NO"
        ),
        all_wrong=s2.get("all_three_wrong", "n/a"),
        rec_fam=rec_fam_txt,
        rec_classes=rec_class_txt,
        stage_a_label=decision.get("label", "not evaluated"),
        pattern=(
            "supported"
            if s0 and s1 and s2 and s0.get("new_unique_recoveries", 0) < s1.get("new_unique_recoveries", 0) < s2.get("new_unique_recoveries", 0)
            else "not supported by the observed unique-recovery ordering"
        ),
        leak_txt=json.dumps(leak, indent=2, sort_keys=True, default=json_default),
        final=final["label"],
        reason=final["reason"],
        next_action=final["next_action"],
        py=env["python"],
        torch=env["torch"],
        dev=env["device"],
        seed=env["seed"],
    )


def evaluate_student_on_competition(
    *,
    torch,
    model,
    x: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    folds: np.ndarray,
    meta,
    ei_of_label,
    device,
    lzh_pred: np.ndarray,
    wyh_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if x.shape[1] != 200:
        raise RuntimeError("student competition input dim {} != 200".format(x.shape[1]))
    _, proba, _ = evaluate_model(torch, model, x, None, device)
    assert_probability_rows(proba, atol=1e-4)
    pred = argmax_labels(proba, class_names)
    metrics = classification_metrics(y_true, pred, class_names, proba)
    folds_acc = fold_slice_accuracy(y_true, pred, folds)
    slices = slice_metrics(y_true, pred, meta, class_names, ei_of_label)
    fam = family_block(metrics["per_class_recall"])
    comp = complementarity_block(y_true, lzh_pred, wyh_pred, pred, folds)
    row = {
        **metrics,
        **folds_acc,
        "hard_bucket_accuracy": slices["hard_bucket_accuracy"],
        "neuron_accuracy": slices["neuron_accuracy"],
        "glial_accuracy": slices["glial_accuracy"],
        "family_recall": fam,
        **comp,
        "evaluation_term": "honest_external_validation_not_competition_oof",
    }
    return pred, proba, row


def main() -> int:
    t0 = time.time()
    assert_v3_interpreter()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert_work_dir_ignored(WORK_DIR)

    import torch
    from torch import nn

    device = require_mps()
    set_seed(SEED)
    TeacherMLP, StudentMLP = build_model_classes(torch, nn)
    env = environment_payload(torch, device)
    print("V3-E02D environment:", json.dumps({k: env[k] for k in ("python", "torch", "device", "mps_available")}), flush=True)

    data = load_dataset(ROOT)
    validate_contract(data)
    class_names = allowed_labels(ROOT)
    if len(class_names) != N_CLASSES:
        raise RuntimeError("expected 60 classes")
    folds_df, _fold_messages = load_and_validate_team_folds(
        data.counts_train.index, data.counts_test.index, data.y_train
    )
    fold_map = folds_df.set_index("Cell_ID")["fold"]
    train_ids = [str(v) for v in data.meta_train.index.tolist()]
    y_true = data.y_train.astype(str).to_numpy()
    canonical_fold = fold_map.reindex(train_ids).to_numpy()
    ei_of_label = ei_of_label_from_train(data.meta_train, class_names)

    print("Loading and verifying approved reference...", flush=True)
    ref = load_usable_reference(data, class_names)
    ref_info = {
        "md5": ref["audit"]["md5"],
        "raw_n_obs": ref["audit"]["raw_n_obs"],
        "raw_n_vars": ref["audit"]["raw_n_vars"],
        "n_usable": int(len(ref["ids"])),
        "n_train_id_overlaps_removed": ref["audit"]["n_train_id_overlaps_removed"],
        "n_test_id_overlaps_removed": ref["audit"]["n_test_id_overlaps_removed"],
        "n_exact_vector_duplicates_removed": ref["audit"]["n_exact_vector_duplicates_removed"],
        "gene200_sha256": ref["gene200_sha256"],
        "gene500_sha256": ref["gene500_sha256"],
        "subset_ok": ref["subset_ok"],
        "genes200": ref["genes200"],
        "genes500": ref["genes500"],
        "class_names": list(class_names),
    }
    print(
        "Reference OK md5={} usable={} gene200={} gene500={}".format(
            ref_info["md5"], ref_info["n_usable"], ref_info["gene200_sha256"][:12], ref_info["gene500_sha256"][:12]
        ),
        flush=True,
    )

    y_ref = ref["y_codes"]
    y_ref_names = np.asarray(class_names)[y_ref]
    idx = np.arange(len(y_ref))
    train_idx, val_idx = train_test_split(
        idx, test_size=STAGE_A_VAL_FRAC, random_state=SEED, stratify=y_ref
    )
    np.save(WORK_DIR / "stage_a_train_idx.npy", train_idx)
    np.save(WORK_DIR / "stage_a_val_idx.npy", val_idx)

    print("Stage A: A200 vs A500...", flush=True)
    stage_a_models = {}
    stage_a_rows = {}
    for tag, x_all, in_dim in (("A200", ref["X200"], 200), ("A500", ref["X500"], 500)):
        if x_all.shape[1] != in_dim:
            raise RuntimeError("{} width {} != {}".format(tag, x_all.shape[1], in_dim))
        model = TeacherMLP(in_dim)
        fit = train_model(
            torch=torch,
            model=model,
            x=x_all[train_idx],
            y=y_ref[train_idx],
            val_x=x_all[val_idx],
            val_y=y_ref[val_idx],
            device=device,
            epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=SEED,
            log_name=tag,
        )
        acc, proba, extras = evaluate_model(torch, model, x_all[val_idx], y_ref[val_idx], device)
        pred_names = np.asarray(class_names)[np.argmax(proba, axis=1)]
        metrics = classification_metrics(y_ref_names[val_idx], pred_names, class_names, proba)
        metrics["family_recall"] = family_block(metrics["per_class_recall"])
        metrics["glial"] = glial_family_accuracy(y_ref_names[val_idx], pred_names, class_names, ei_of_label)
        metrics["fit"] = {"best_epoch": fit["best_epoch"], "best_val_acc": fit["best_val_acc"]}
        stage_a_models[tag] = model
        stage_a_rows[tag] = metrics
        torch.save({"state_dict": model.state_dict(), "in_dim": in_dim}, WORK_DIR / "{}.pt".format(tag.lower()))
        print("  {} val acc={:.4f} macro-F1={:.4f}".format(tag, metrics["accuracy"], metrics["macro_f1"]), flush=True)

    decision = privileged_signal_decision(stage_a_rows["A200"], stage_a_rows["A500"])
    stage_a_payload = {
        "experiment_id": EXPERIMENT_ID,
        "split": {"train_frac": 0.80, "val_frac": 0.20, "random_state": SEED, "stratified": True},
        "A200": stage_a_rows["A200"],
        "A500": stage_a_rows["A500"],
        "decision": decision,
        "environment": env,
        "reference": {k: v for k, v in ref_info.items() if k not in {"genes200", "genes500"}},
    }
    write_json(OUT_DIR / "v3_e02d_reference_signal_metrics.json", stage_a_payload)
    print("Stage A decision:", decision["label"], flush=True)

    leakage = {
        "competition_test_labels_used": False,
        "competition_train_labels_used_for_student_training": False,
        "reference_excludes_train_and_test_ids": True,
        "student_inference_input_dim": 200,
        "privileged_500_genes_at_deployment": False,
        "prediction_csv_modified": False,
        "model_v1_modified": False,
        "model_v2_modified": False,
        "v3_e00t_metrics_modified": False,
        "canonical_folds_3_4_used_for_tuning": False,
        "seed_search": False,
        "hyperparameter_search": False,
        "cell_id_dtype": "lossless_string",
    }

    payload = {
        "environment": env,
        "reference": ref_info,
        "stage_a": stage_a_payload,
        "teacher": None,
        "students": {},
        "complementarity": {},
        "recoveries": {},
        "ran_stage_bc": False,
        "leakage_audit": leakage,
        "final_classification": {
            "label": "FAILED TO ADD NEW INFORMATION",
            "reason": "Stage A did not support a privileged 500-gene signal.",
            "next_action": next_action_for("FAILED TO ADD NEW INFORMATION"),
        },
    }

    if not decision["supported"]:
        REPORT_PATH.write_text(render_report(payload))
        print("STOP: PRIVILEGED SIGNAL NOT SUPPORTED. Report:", REPORT_PATH, flush=True)
        print("elapsed_sec", round(time.time() - t0, 1), flush=True)
        return 0

    print("Stage B: cross-fitted 500-gene teacher...", flush=True)
    splitter = StratifiedKFold(n_splits=TEACHER_FOLDS, shuffle=True, random_state=SEED)
    teacher_logits = np.zeros((len(y_ref), N_CLASSES), dtype=np.float32)
    teacher_relation = np.zeros((len(y_ref), N_CLASSES), dtype=np.float32)
    teacher_fold = np.full(len(y_ref), -1, dtype=np.int64)
    fold_metrics = []
    dummy = np.zeros((len(y_ref), 1))
    for fold_id, (tr, te) in enumerate(splitter.split(dummy, y_ref)):
        teacher_fold[te] = fold_id
        tr_fit, tr_val = train_test_split(tr, test_size=0.20, random_state=SEED, stratify=y_ref[tr])
        model, fit_info = fit_with_early_stopping_then_full(
            torch=torch,
            factory=lambda: TeacherMLP(500),
            x=ref["X500"],
            y=y_ref,
            train_idx=tr_fit,
            val_idx=tr_val,
            full_idx=tr,
            device=device,
            seed=SEED,
            log_name="teacher_fold{}".format(fold_id),
            checkpoint=WORK_DIR / "teacher_fold{}.pt".format(fold_id),
        )
        # Honest held-out outputs from the fold teacher trained without those cells.
        # Prototypes come from the teacher-training cells of this fold (all `tr`).
        _, _, tr_extra = evaluate_model(torch, model, ref["X500"][tr], y_ref[tr], device)
        protos = class_prototypes(tr_extra["latents"], y_ref[tr], N_CLASSES)
        _, te_proba, te_extra = evaluate_model(torch, model, ref["X500"][te], y_ref[te], device)
        rel = cosine_to_prototypes(te_extra["latents"], protos)
        if rel.shape != (len(te), N_CLASSES):
            raise RuntimeError("relation shape {} for fold {}".format(rel.shape, fold_id))
        teacher_logits[te] = te_extra["logits"].astype(np.float32)
        teacher_relation[te] = rel.astype(np.float32)
        pred_names = np.asarray(class_names)[np.argmax(te_proba, axis=1)]
        m = classification_metrics(y_ref_names[te], pred_names, class_names, te_proba)
        m["fold"] = fold_id
        m["selected_epochs"] = fit_info["selected_epochs"]
        fold_metrics.append(m)
        print("  teacher fold {} held-out acc={:.4f}".format(fold_id, m["accuracy"]), flush=True)

    if (teacher_fold < 0).any():
        raise RuntimeError("cross-fitted teacher targets incomplete")
    teacher_proba = softmax_from_logits(teacher_logits, 1.0)
    if teacher_relation.shape != (len(y_ref), N_CLASSES):
        raise RuntimeError("teacher relation shape {}".format(teacher_relation.shape))
    np.save(WORK_DIR / "teacher_logits.npy", teacher_logits)
    np.save(WORK_DIR / "teacher_proba.npy", teacher_proba)
    np.save(WORK_DIR / "teacher_relation.npy", teacher_relation)
    np.save(WORK_DIR / "teacher_fold.npy", teacher_fold)
    t_pred = np.asarray(class_names)[np.argmax(teacher_proba, axis=1)]
    t_metrics = classification_metrics(y_ref_names, t_pred, class_names, teacher_proba)
    teacher_payload = {
        "protocol": "StratifiedKFold(n_splits=3, shuffle=True, random_state=20260820)",
        "architecture": "Linear(500,256)-LN-GELU-Dropout-Linear(256,128)-LN-GELU-Dropout-Linear(128,60)",
        "crossfit_accuracy": t_metrics["accuracy"],
        "crossfit_correct": t_metrics["correct"],
        "crossfit_macro_f1": t_metrics["macro_f1"],
        "crossfit_log_loss": t_metrics["log_loss"],
        "relation_target_shape": list(teacher_relation.shape),
        "fold_metrics": fold_metrics,
        "environment": env,
    }
    write_json(OUT_DIR / "v3_e02d_teacher_metrics.json", teacher_payload)
    payload["teacher"] = teacher_payload

    print("Loading V3-E00T registry...", flush=True)
    registry = pd.read_parquet(REGISTRY_PATH)
    registry["Cell_ID"] = registry["Cell_ID"].astype(str)
    registry = registry.set_index("Cell_ID").reindex(train_ids)
    if registry["true_label"].isna().any():
        raise RuntimeError("registry missing competition train Cell_IDs")
    if list(registry["true_label"].astype(str)) != list(y_true):
        raise RuntimeError("registry labels do not match official meta_train")
    lzh_pred = registry["lzh_prior_h_pred"].astype(str).to_numpy()
    wyh_pred = registry["wyh_model_v2_pred"].astype(str).to_numpy()
    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    pair_ok = lzh_ok | wyh_ok
    if int(pair_ok.sum()) != LZH_WYH_ORACLE_CORRECT:
        raise RuntimeError("LZH+WYH oracle {} != 4215".format(int(pair_ok.sum())))
    if int((~pair_ok).sum()) != SHARED_ERROR_N:
        raise RuntimeError("both-wrong {} != 785".format(int((~pair_ok).sum())))

    x_comp = np.log1p(data.counts_train.loc[train_ids, ref["genes200"]].to_numpy(dtype=np.float32))
    if x_comp.shape != (N_TRAIN, 200):
        raise RuntimeError("competition train feature shape {}".format(x_comp.shape))

    specs = [
        ("S0", 0.0, 0.0),
        ("S1", LAMBDA_KD, 0.0),
        ("S2", LAMBDA_KD, LAMBDA_REL),
    ]
    student_rows = {}
    student_preds = {}
    student_probas = {}
    for name, lam_kd, lam_rel in specs:
        print("Training student {}...".format(name), flush=True)
        model, fit_info = fit_with_early_stopping_then_full(
            torch=torch,
            factory=lambda: StudentMLP(200),
            x=ref["X200"],
            y=y_ref,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            seed=SEED,
            log_name=name,
            teacher_logits=teacher_logits if lam_kd > 0 else None,
            teacher_relation=teacher_relation if lam_rel > 0 else None,
            lambda_kd=lam_kd,
            lambda_rel=lam_rel,
            checkpoint=WORK_DIR / "{}.pt".format(name.lower()),
        )
        if model.in_dim != 200:
            raise RuntimeError("{} in_dim {}".format(name, model.in_dim))
        pred, proba, row = evaluate_student_on_competition(
            torch=torch,
            model=model,
            x=x_comp,
            y_true=y_true,
            class_names=class_names,
            folds=canonical_fold,
            meta=data.meta_train.loc[train_ids],
            ei_of_label=ei_of_label,
            device=device,
            lzh_pred=lzh_pred,
            wyh_pred=wyh_pred,
        )
        row["variant"] = name
        row["lambda_kd"] = lam_kd
        row["lambda_rel"] = lam_rel
        row["selected_epochs"] = fit_info["selected_epochs"]
        student_rows[name] = row
        student_preds[name] = pred
        student_probas[name] = proba
        write_validation_csv(OUT_DIR / "v3_e02d_{}_validation.csv".format(name.lower()), train_ids, y_true, pred, canonical_fold)
        print(
            "  {} acc={:.4f} unique_recoveries={} three-oracle={:.4f}".format(
                name, row["accuracy"], row["new_unique_recoveries"], row["three_expert_oracle_accuracy"]
            ),
            flush=True,
        )

    write_proba(OUT_DIR / "v3_e02d_s2_validation_probabilities.csv.gz", train_ids, student_probas["S2"], class_names)

    print("S2 test inference on official 200 genes...", flush=True)
    x_test = np.log1p(data.counts_test.loc[:, ref["genes200"]].to_numpy(dtype=np.float32))
    if x_test.shape[1] != 200:
        raise RuntimeError("test input dim {} != 200".format(x_test.shape[1]))
    s2_model = StudentMLP(200)
    ckpt = torch.load(WORK_DIR / "s2.pt", map_location="cpu", weights_only=False)
    s2_model.load_state_dict(ckpt["state_dict"])
    s2_model.to(device)
    if x_test.shape[1] != s2_model.in_dim:
        raise RuntimeError("deployment input dim mismatch")
    _, test_proba, _ = evaluate_model(torch, s2_model, x_test, None, device)
    assert_probability_rows(test_proba, atol=1e-4)
    test_ids = [str(v) for v in data.counts_test.index.tolist()]
    write_proba(OUT_DIR / "v3_e02d_s2_test_probabilities.csv.gz", test_ids, test_proba, class_names)

    comparison_rows = []
    for name in ("S0", "S1", "S2"):
        row = student_rows[name]
        comparison_rows.append(
            {
                "variant": name,
                "accuracy": row["accuracy"],
                "correct": row["correct"],
                "macro_f1": row["macro_f1"],
                "log_loss": row["log_loss"],
                "folds_0_2": row["folds_0_2"],
                "folds_3_4": row["folds_3_4"],
                "hard_bucket_accuracy": row["hard_bucket_accuracy"],
                "neuron_accuracy": row["neuron_accuracy"],
                "glial_accuracy": row["glial_accuracy"],
                "new_unique_recoveries": row["new_unique_recoveries"],
                "three_expert_oracle_correct": row["three_expert_oracle_correct"],
                "three_expert_oracle_accuracy": row["three_expert_oracle_accuracy"],
                "three_expert_oracle_accuracy_folds_3_4": row["three_expert_oracle_accuracy_folds_3_4"],
                "all_three_wrong": row["all_three_wrong"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "v3_e02d_student_comparison.csv", index=False)

    s2_pred = student_preds["S2"]
    s2_proba = student_probas["S2"]
    s2_conf = confidence_from_proba(s2_proba)
    rec_mask = unique_recovery_mask(lzh_ok, wyh_ok, s2_pred == y_true)
    both_wrong = (~lzh_ok) & (~wyh_ok)
    rec_rows = []
    for i in np.where(both_wrong)[0]:
        rec_rows.append(
            {
                "Cell_ID": train_ids[i],
                "true_label": y_true[i],
                "LZH_pred": lzh_pred[i],
                "WYH_pred": wyh_pred[i],
                "S2_pred": s2_pred[i],
                "S2_top1": float(s2_conf["top1"][i]),
                "S2_margin": float(s2_conf["margin"][i]),
                "S2_entropy": float(s2_conf["entropy"][i]),
                "Region": registry.iloc[i]["Region"] if "Region" in registry.columns else "",
                "E/I": registry.iloc[i]["E/I"] if "E/I" in registry.columns else "",
                "Segment": registry.iloc[i]["Segment"] if "Segment" in registry.columns else "",
                "Section_ID": registry.iloc[i]["Section_ID"] if "Section_ID" in registry.columns else "",
                "hard_bucket": bool(registry.iloc[i]["hard_bucket"]) if "hard_bucket" in registry.columns else False,
                "n_detected": registry.iloc[i]["n_detected"] if "n_detected" in registry.columns else np.nan,
                "library_size": registry.iloc[i]["library_size"] if "library_size" in registry.columns else np.nan,
                "canonical_fold": int(canonical_fold[i]),
                "true_family": family_of(str(y_true[i])),
                "s2_family": family_of(str(s2_pred[i])),
                "s2_recovered": bool(rec_mask[i]),
            }
        )
    rec_df = pd.DataFrame(rec_rows)
    rec_df[rec_df["s2_recovered"]].to_csv(OUT_DIR / "v3_e02d_unique_recoveries.csv", index=False)

    recovered = rec_df[rec_df["s2_recovered"]]
    by_class = Counter(recovered["true_label"].tolist())
    by_family = Counter(recovered["true_family"].tolist())
    recoveries_summary = {
        "n_shared_errors": int(both_wrong.sum()),
        "s0_unique_recoveries": student_rows["S0"]["new_unique_recoveries"],
        "s1_unique_recoveries": student_rows["S1"]["new_unique_recoveries"],
        "s2_unique_recoveries": student_rows["S2"]["new_unique_recoveries"],
        "top_recovered_classes": [{"true_label": k, "n": int(n)} for k, n in by_class.most_common(15)],
        "by_family": [{"family": k, "n": int(n)} for k, n in by_family.most_common()],
        "family_comparison": {
            fam: {
                "S0": int(np.sum(unique_recovery_mask(lzh_ok, wyh_ok, student_preds["S0"] == y_true) & (np.array([family_of(v) for v in y_true]) == fam))),
                "S1": int(np.sum(unique_recovery_mask(lzh_ok, wyh_ok, student_preds["S1"] == y_true) & (np.array([family_of(v) for v in y_true]) == fam))),
                "S2": int(np.sum(unique_recovery_mask(lzh_ok, wyh_ok, student_preds["S2"] == y_true) & (np.array([family_of(v) for v in y_true]) == fam))),
            }
            for fam in list(CLASS_FAMILIES) + ["neuronal_or_other"]
        },
    }

    label, reason = experiment_classification(True, student_rows)
    payload["students"] = student_rows
    payload["complementarity"] = {
        "lzh_wyh_oracle_correct": int(pair_ok.sum()),
        "lzh_wyh_oracle_accuracy": float(np.mean(pair_ok)),
        "shared_error_n": int((~pair_ok).sum()),
        "S0": {k: student_rows["S0"][k] for k in student_rows["S0"] if "oracle" in k or "unique" in k or "wrong" in k or "disagree" in k},
        "S1": {k: student_rows["S1"][k] for k in student_rows["S1"] if "oracle" in k or "unique" in k or "wrong" in k or "disagree" in k},
        "S2": {k: student_rows["S2"][k] for k in student_rows["S2"] if "oracle" in k or "unique" in k or "wrong" in k or "disagree" in k},
    }
    payload["recoveries"] = recoveries_summary
    payload["ran_stage_bc"] = True
    payload["final_classification"] = {
        "label": label,
        "reason": reason,
        "next_action": next_action_for(label),
    }
    payload["elapsed_sec"] = time.time() - t0
    write_json(OUT_DIR / "v3_e02d_complementarity.json", {
        "experiment_id": EXPERIMENT_ID,
        "lzh_wyh_oracle_correct": int(pair_ok.sum()),
        "shared_error_n": int((~pair_ok).sum()),
        "students": student_rows,
        "recoveries": recoveries_summary,
        "final_classification": payload["final_classification"],
        "environment": env,
        "leakage_audit": leakage,
        "oracle_is_not_deployable_accuracy": True,
        "three_expert_reaches_0_85": bool(student_rows["S2"]["three_expert_oracle_accuracy"] >= 0.85),
        "three_expert_reaches_0_85_folds_3_4": bool(student_rows["S2"]["three_expert_oracle_accuracy_folds_3_4"] >= 0.85),
        "s2_new_unique_recoveries": student_rows["S2"]["new_unique_recoveries"],
        "oracle_4215_plus_recoveries": LZH_WYH_ORACLE_CORRECT + student_rows["S2"]["new_unique_recoveries"],
    })
    REPORT_PATH.write_text(render_report(payload))
    print("Wrote", REPORT_PATH, flush=True)
    print("Final classification:", label, flush=True)
    print("elapsed_sec", round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
