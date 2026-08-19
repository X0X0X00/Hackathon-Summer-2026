"""Leakage-aware V7 optimization on top of the reproduced V6 ensemble.

Stages:
1. conditional glial-family specialists (external reference + competition OOF stacker)
2. low-confidence multiclass router/stacker
3. anatomy-aware router features (the static cache contains Region/EI/Segment, including NaN)
4. rare neuronal pair specialists

All configuration choices are made on folds 0-2 and reported once on folds 3-4.
The final accepted pipeline is then fitted on all 5,000 OOF rows and applied to test.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import load
from common_ext import load_ext
from ext_post import postprocess


WORK = Path(__file__).resolve().parent
BASE = WORK.parent
N_CLASSES = 60
MEMBERS = ["refonly_full", "ext_all25", "mlp3", "yhh_v1", "bag8mix"]


def acc(p: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None) -> float:
    if rows is None:
        rows = np.arange(len(y))
    return float(np.mean(p[rows].argmax(1) == y[rows]))


def correct(p: np.ndarray, y: np.ndarray, rows: np.ndarray) -> int:
    return int(np.sum(p[rows].argmax(1) == y[rows]))


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-12)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def load_members(D, E):
    tr = np.where(D["is_train"])[0]
    te = np.where(~D["is_train"])[0]
    ids_tr = D["ids"][tr].astype(str)
    ids_te = D["ids"][te].astype(str)
    e_ids_tr = E["ids"][E["is_train"]].astype(str)
    e_ids_te = E["ids"][E["is_test"]].astype(str)
    pos_tr = {v: i for i, v in enumerate(e_ids_tr)}
    pos_te = {v: i for i, v in enumerate(e_ids_te)}
    map_tr = np.asarray([pos_tr[v] for v in ids_tr])
    map_te = np.asarray([pos_te[v] for v in ids_te])
    oofs, tests = [], []
    for name in MEMBERS:
        local = WORK / "oof" / f"{name}.npz"
        if local.exists():
            z = np.load(local)
            oo, tt = z["oof"], z["test"]
        else:
            z = np.load(WORK / "oof_ext" / f"{name}.npz")
            oo, tt = z["oof"], z["test"]
            if not name.startswith("yhh"):
                oo, tt = oo[map_tr], tt[map_te]
        oofs.append(oo.astype(np.float32))
        tests.append(tt.astype(np.float32))
    return np.stack(oofs), np.stack(tests)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    return np.sqrt(counts.max() / counts[y]).astype(np.float32)


def aligned_predict(model, X: np.ndarray, n: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), n), np.float32)
    out[:, np.asarray(model.classes_, int)] = raw
    return out


def classifier(seed: int, n_classes: int, rare: bool = False):
    return lgb.LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=320 if not rare else 220,
        learning_rate=0.035,
        num_leaves=15 if not rare else 9,
        min_child_samples=14 if not rare else 8,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.15,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )


def fit_family(
    name: str,
    labels: list[str],
    D,
    E,
    member_oof: np.ndarray,
    member_test: np.ndarray,
):
    """Return external and competition-CV conditional probabilities."""
    names = [str(x) for x in D["labels"]]
    indices = np.asarray([names.index(x) for x in labels], int)
    g2l = {g: i for i, g in enumerate(indices)}
    tr = np.where(D["is_train"])[0]
    te = np.where(~D["is_train"])[0]
    y = D["y"][tr]
    folds = D["folds"][tr]
    local_y_all = np.asarray([g2l.get(int(v), -1) for v in y], int)
    family = local_y_all >= 0
    n = len(indices)

    # External-only expert. Static cache layouts are asserted identical.
    assert list(map(str, D["names"])) == list(map(str, E["names"]))
    ref = E["is_ref"] & np.isin(E["y"], indices)
    ref_global = E["y"][ref]
    ref_y = np.asarray([g2l[int(v)] for v in ref_global], int)
    ext_model = classifier(1103 + len(labels), n, rare=len(labels) == 2)
    ext_model.fit(E["X"][ref], ref_y, sample_weight=balanced_weights(ref_y))
    ext_oof = aligned_predict(ext_model, D["X"][tr], n)
    ext_test = aligned_predict(ext_model, D["X"][te], n)

    # Competition stacker: raw/static features plus all member probabilities.
    Xo = np.hstack([D["X"][tr], member_oof.transpose(1, 0, 2).reshape(len(tr), -1)])
    Xt = np.hstack([D["X"][te], member_test.transpose(1, 0, 2).reshape(len(te), -1)])
    cv_oof = np.zeros((len(tr), n), np.float32)
    for f in range(5):
        fit = family & (folds != f)
        va = folds == f
        model = classifier(2101 + f * 31 + len(labels), n, rare=len(labels) == 2)
        model.fit(Xo[fit], local_y_all[fit], sample_weight=balanced_weights(local_y_all[fit]))
        cv_oof[va] = aligned_predict(model, Xo[va], n)
    final_model = classifier(3907 + len(labels), n, rare=len(labels) == 2)
    final_model.fit(Xo[family], local_y_all[family], sample_weight=balanced_weights(local_y_all[family]))
    cv_test = aligned_predict(final_model, Xt, n)

    family_rows = np.where(family)[0]
    diagnostics = {
        "name": name,
        "labels": labels,
        "competition_cells": int(family.sum()),
        "reference_cells": int(ref.sum()),
        "external_conditional_accuracy": acc(ext_oof, local_y_all, family_rows),
        "competition_conditional_accuracy": acc(cv_oof, local_y_all, family_rows),
    }
    return indices, ext_oof, ext_test, cv_oof, cv_test, diagnostics


def apply_family(
    base: np.ndarray,
    specialist: np.ndarray,
    indices: np.ndarray,
    mass_gate: float,
    cap: float,
    spec_conf: float,
    weight: float,
):
    result = base.copy()
    mass = base[:, indices].sum(1)
    top = base.argmax(1)
    sorted_p = np.partition(base, -2, axis=1)[:, -2:]
    margin = sorted_p[:, 1] - sorted_p[:, 0]
    selected = (
        np.isin(top, indices)
        & (mass >= mass_gate)
        & (margin <= cap)
        & (specialist.max(1) >= spec_conf)
    )
    if selected.any():
        within = base[np.ix_(selected, indices)]
        within /= np.maximum(within.sum(1, keepdims=True), 1e-12)
        corrected = normalize((1.0 - weight) * within + weight * specialist[selected])
        result[np.ix_(selected, indices)] = corrected * mass[selected, None]
    return normalize(result), selected


def select_family_config(
    base: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    indices: np.ndarray,
    ext_oof: np.ndarray,
    ext_test: np.ndarray,
    cv_oof: np.ndarray,
    cv_test: np.ndarray,
):
    tune = np.where(folds <= 2)[0]
    hold = np.where(folds >= 3)[0]
    base_tune = correct(base, y, tune)
    candidates = []
    for ew in [0.0, 0.25, 0.5, 0.75, 1.0]:
        so = normalize(ew * ext_oof + (1 - ew) * cv_oof)
        for gate in [0.45, 0.60, 0.75, 0.88]:
            for cap in [0.10, 0.20, 0.40, 1.01]:
                for sc in [0.0, 0.45, 0.60, 0.75]:
                    for w in [0.25, 0.50, 0.75, 1.0]:
                        p, selected = apply_family(base, so, indices, gate, cap, sc, w)
                        fold_delta = [
                            correct(p, y, np.where(folds == f)[0])
                            - correct(base, y, np.where(folds == f)[0])
                            for f in range(3)
                        ]
                        candidates.append((
                            correct(p, y, tune) - base_tune,
                            min(fold_delta),
                            -int(selected[tune].sum()),
                            ew, gate, cap, sc, w,
                        ))
    # The minimum-fold tie break penalizes brittle configurations.
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, ew, gate, cap, sc, w = best
    so = normalize(ew * ext_oof + (1 - ew) * cv_oof)
    st = normalize(ew * ext_test + (1 - ew) * cv_test)
    po, selected_oof = apply_family(base, so, indices, gate, cap, sc, w)
    config = {
        "external_weight": ew,
        "mass_gate": gate,
        "margin_cap": cap,
        "specialist_confidence": sc,
        "correction_weight": w,
        "tune_delta_correct": correct(po, y, tune) - correct(base, y, tune),
        "holdout_delta_correct": correct(po, y, hold) - correct(base, y, hold),
        "per_fold_delta": [
            correct(po, y, np.where(folds == f)[0])
            - correct(base, y, np.where(folds == f)[0]) for f in range(5)
        ],
        "selected_oof": int(selected_oof.sum()),
    }
    return config, po, st


def router_features(D, member_p: np.ndarray, base: np.ndarray, which: str, anatomy: bool):
    rows = np.where(D["is_train"] if which == "train" else ~D["is_train"])[0]
    # Probabilities, dispersion, uncertainty and member disagreement.
    m = member_p.transpose(1, 0, 2)
    flat = m.reshape(len(rows), -1)
    mean = m.mean(1)
    std = m.std(1)
    conf = m.max(2)
    entropy = -(m * np.log(np.maximum(m, 1e-8))).sum(2)
    top = m.argmax(2)
    agreement = np.asarray([np.bincount(r, minlength=60).max() for r in top], np.float32)[:, None]
    bsorted = np.partition(base, -2, axis=1)[:, -2:]
    bmargin = (bsorted[:, 1] - bsorted[:, 0])[:, None]
    pieces = [flat, mean, std, conf, entropy, agreement, bmargin]
    if anatomy:
        # Includes Region/EI/Segment, expression PCs and spatial features. LightGBM handles NaN.
        pieces.append(D["X"][rows])
        pieces.append(np.column_stack([
            np.isnan(D["X"][rows, -7]),
            np.isnan(D["X"][rows, -6]),
            np.isnan(D["X"][rows, -5]),
        ]).astype(np.float32))
    return np.hstack(pieces).astype(np.float32)


def router_model(seed: int):
    return lgb.LGBMClassifier(
        objective="multiclass", n_estimators=180, learning_rate=0.035,
        num_leaves=13, min_child_samples=28, colsample_bytree=0.55,
        reg_alpha=0.25, reg_lambda=8.0, random_state=seed,
        n_jobs=8, verbosity=-1,
    )


def crossfit_router(Xo, Xt, y, folds):
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    for f in range(5):
        fit, va = folds != f, folds == f
        model = router_model(5003 + f * 43)
        model.fit(Xo[fit], y[fit])
        oof[va] = aligned_predict(model, Xo[va], N_CLASSES)
    model = router_model(7907)
    model.fit(Xo, y)
    test = aligned_predict(model, Xt, N_CLASSES)
    return oof, test


def select_router(base, router_oof, router_test, y, folds):
    tune = np.where(folds <= 2)[0]
    hold = np.where(folds >= 3)[0]
    sorted_p = np.partition(base, -2, axis=1)[:, -2:]
    margin = sorted_p[:, 1] - sorted_p[:, 0]
    candidates = []
    for cap in [0.05, 0.10, 0.20, 0.40, 1.01]:
        for rc in [0.25, 0.40, 0.55, 0.70]:
            for w in [0.10, 0.20, 0.35, 0.50, 0.75]:
                selected = (margin <= cap) & (router_oof.max(1) >= rc)
                p = base.copy()
                p[selected] = normalize((1-w) * base[selected] + w * router_oof[selected])
                fd = [correct(p, y, np.where(folds == f)[0]) - correct(base, y, np.where(folds == f)[0]) for f in range(3)]
                candidates.append((correct(p, y, tune)-correct(base, y, tune), min(fd), -int(selected[tune].sum()), cap, rc, w))
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, cap, rc, w = best
    selected = (margin <= cap) & (router_oof.max(1) >= rc)
    po = base.copy()
    po[selected] = normalize((1-w) * base[selected] + w * router_oof[selected])
    config = {
        "margin_cap": cap, "router_confidence": rc, "weight": w,
        "tune_delta_correct": correct(po,y,tune)-correct(base,y,tune),
        "holdout_delta_correct": correct(po,y,hold)-correct(base,y,hold),
        "per_fold_delta": [correct(po,y,np.where(folds==f)[0])-correct(base,y,np.where(folds==f)[0]) for f in range(5)],
        "selected_oof": int(selected.sum()),
    }
    # Apply the identical gate to test using test base margins.
    return config, po, (cap, rc, w, router_test)


def apply_router_test(base_test, spec):
    cap, rc, w, router_test = spec
    s = np.partition(base_test, -2, axis=1)[:, -2:]
    selected = ((s[:,1]-s[:,0]) <= cap) & (router_test.max(1) >= rc)
    out = base_test.copy()
    out[selected] = normalize((1-w)*base_test[selected] + w*router_test[selected])
    return normalize(out), selected


def stage_report(name, before, after, y, folds, extra=None):
    tune = np.where(folds <= 2)[0]
    hold = np.where(folds >= 3)[0]
    row = {
        "stage": name,
        "accuracy": acc(after, y),
        "correct": correct(after, y, np.arange(len(y))),
        "delta_correct": correct(after,y,np.arange(len(y)))-correct(before,y,np.arange(len(y))),
        "tune_accuracy": acc(after,y,tune),
        "holdout_accuracy": acc(after,y,hold),
        "holdout_delta_correct": correct(after,y,hold)-correct(before,y,hold),
    }
    if extra: row.update(extra)
    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    D, E = load(), load_ext()
    tr = np.where(D["is_train"])[0]
    y = D["y"][tr]
    folds = D["folds"][tr]
    member_oof, member_test = load_members(D, E)
    raw_oof = member_oof.mean(0)
    raw_test = member_test.mean(0)
    current_oof = postprocess(raw_oof, "train")
    current_test = postprocess(raw_test, "test")
    report = {"baseline": stage_report("reproduced_v6", current_oof, current_oof, y, folds)}

    # 1) Main glial family specialists, sequentially accepted only with positive holdout gain.
    specs = [
        ("oligodendrocyte_lineage", ["oligodendrocyte_1","oligodendrocyte_2","oligodendrocyte_precursor_cell","oligodendrocyte_progenitor_1","oligodendrocyte_progenitor_2"]),
        ("astro_vascular_meningeal", ["astrocyte_1","astrocyte_2","endothelial","pericyte","meninges_1","meninges_2","meninges_3"]),
    ]
    report["family_specialists"] = {}
    for name, labels in specs:
        fitted = fit_family(name, labels, D, E, member_oof, member_test)
        indices, eo, et, co, ct, diagnostics = fitted
        config, candidate_oof, candidate_spec_test = select_family_config(current_oof,y,folds,indices,eo,et,co,ct)
        candidate_test, selected_test = apply_family(current_test,candidate_spec_test,indices,config["mass_gate"],config["margin_cap"],config["specialist_confidence"],config["correction_weight"])
        accepted = config["tune_delta_correct"] > 0 and config["holdout_delta_correct"] > 0
        before = current_oof
        if accepted:
            current_oof, current_test = candidate_oof, candidate_test
        report["family_specialists"][name] = {**diagnostics, **config, "accepted": accepted, "selected_test": int(selected_test.sum())}
        stage_report(name,before,current_oof,y,folds,report["family_specialists"][name])

    # 2/3) Compare probability-only router against anatomy/expression-aware router.
    report["routers"] = {}
    best_router = None
    for anatomy in [False, True]:
        tag = "anatomy_aware" if anatomy else "probability_only"
        Xo = router_features(D,member_oof,current_oof,"train",anatomy)
        Xt = router_features(D,member_test,current_test,"test",anatomy)
        ro, rt = crossfit_router(Xo,Xt,y,folds)
        config, po, test_spec = select_router(current_oof,ro,rt,y,folds)
        score = (config["holdout_delta_correct"], config["tune_delta_correct"])
        report["routers"][tag] = {**config, "router_raw_accuracy":acc(ro,y), "feature_count":int(Xo.shape[1])}
        if config["tune_delta_correct"] > 0 and config["holdout_delta_correct"] > 0 and (best_router is None or score > best_router[0]):
            best_router = (score,tag,po,test_spec)
        print(json.dumps({"router":tag,**report["routers"][tag]},indent=2),flush=True)
    if best_router is not None:
        _, tag, po, test_spec = best_router
        before=current_oof
        current_oof=po
        current_test,selected_test=apply_router_test(current_test,test_spec)
        report["routers"][tag]["accepted"]=True
        report["routers"][tag]["selected_test"]=int(selected_test.sum())
        stage_report("router_"+tag,before,current_oof,y,folds,report["routers"][tag])

    # 4) Rare neuron pair specialists. Same strict tune/holdout acceptance rule.
    rare_specs = [
        ("oligodendrocyte_1_vs_progenitor_2",["oligodendrocyte_1","oligodendrocyte_progenitor_2"]),
        ("oligodendrocyte_2_vs_progenitor_2",["oligodendrocyte_2","oligodendrocyte_progenitor_2"]),
        ("oligodendrocyte_progenitor_1_vs_precursor",["oligodendrocyte_progenitor_1","oligodendrocyte_precursor_cell"]),
        ("astrocyte_1_vs_endothelial",["astrocyte_1","endothelial"]),
        ("meninges_1_vs_meninges_2",["meninges_1","meninges_2"]),
        ("DH_in_Klhl14_vs_DH_in_Cdh3",["DH_in_Klhl14","DH_in_Cdh3"]),
        ("DH_ex_Maf_Slc17a8_vs_DH_ex_Maf_Cck",["DH_ex_Maf/Slc17a8","DH_ex_Maf/Cck"]),
        ("DH_ex_Prkcg_Nts_vs_DH_ex_Prkcg_Cck",["DH_ex_Prkcg/Nts","DH_ex_Prkcg/Cck"]),
    ]
    report["rare_specialists"]={}
    for name, labels in rare_specs:
        indices,eo,et,co,ct,diagnostics=fit_family(name,labels,D,E,member_oof,member_test)
        config,candidate_oof,candidate_spec_test=select_family_config(current_oof,y,folds,indices,eo,et,co,ct)
        candidate_test,selected_test=apply_family(current_test,candidate_spec_test,indices,config["mass_gate"],config["margin_cap"],config["specialist_confidence"],config["correction_weight"])
        accepted=config["tune_delta_correct"]>0 and config["holdout_delta_correct"]>0
        before=current_oof
        if accepted: current_oof,current_test=candidate_oof,candidate_test
        report["rare_specialists"][name]={**diagnostics,**config,"accepted":accepted,"selected_test":int(selected_test.sum())}
        stage_report(name,before,current_oof,y,folds,report["rare_specialists"][name])

    current_oof=postprocess(current_oof,"train")
    current_test=postprocess(current_test,"test")
    report["final"]=stage_report("v7_final",postprocess(raw_oof,"train"),current_oof,y,folds)
    np.savez_compressed(WORK/"v7_final_probs.npz",oof=current_oof,test=current_test,member_names=np.asarray(MEMBERS))

    template=pd.read_csv(BASE/"prediction"/"prediction.csv")
    ids=D["ids"][~D["is_train"]]
    assert template.iloc[:,0].astype(str).to_numpy().tolist()==ids.astype(str).tolist()
    out=pd.DataFrame({template.columns[0]:ids,template.columns[1]:D["labels"][current_test.argmax(1)]})
    out.to_csv(BASE/"prediction"/"prediction.csv",index=False)
    (WORK/"v7_optimization_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(BASE/"prediction"/"prediction.csv"),"rows":len(out),"final":report["final"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
