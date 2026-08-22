"""MODEL-V1: full-training YW-004 per-signature logistic specialists.

Model selection is frozen at YW-004 OOF accuracy 0.7598. This module fits
the same architecture on all 5000 labeled training cells and predicts the
official test set. It does not read meta_test target values.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from merfish60.io import N_GENES, N_TEST_CELLS, N_TRAIN_CELLS, MerfishData
from merfish60.models import (
    LR_KWARGS,
    align_predict_proba,
    argmax_labels,
    assert_probability_rows,
    log1p_counts,
    make_logistic_regression,
    one_hot_class,
)
from merfish60.official_contract import (
    SUBMISSION_CELL_ID_COL,
    SUBMISSION_COLUMNS,
    SUBMISSION_PRED_COL,
    YW002_METADATA_FIELDS,
    allowed_labels,
    expected_test_cell_ids,
)
from merfish60.signatures import (
    SIGNATURE_FIELDS,
    build_candidate_map,
    is_deterministic_map,
    mask_and_renormalize,
    signature_key,
    signature_tuple,
    signatures_from_meta,
)


SELECTED_FROM_RUN = "YW-004"
SELECTED_OOF_ACCURACY = 0.7598
SELECTED_OOF_CORRECT = 3799
MODEL_NAME = "MODEL-V1"
ARCHITECTURE = "YW-004 full-train per-signature specialists"


@dataclass
class FittedModelV1:
    """Full-training YW-004 artifacts. Contains no test labels."""

    class_names: List[str]
    gene_names: List[str]
    train_cell_ids: List[str]
    candidate_map: Dict[str, Set[str]]
    signature_components: Dict[str, Tuple[str, str, str]]
    signature_train_ids: Dict[str, List[str]]
    routing_type: Dict[str, str]
    specialists: Dict[str, LogisticRegression]
    specialist_fit_failures: Dict[str, str]
    global_fallback: LogisticRegression
    n_features: int
    lr_kwargs: dict
    n_train: int
    warnings: List[str] = field(default_factory=list)

    @property
    def deterministic_signatures(self) -> List[str]:
        return sorted(k for k, t in self.routing_type.items() if t == "deterministic")

    @property
    def specialist_signatures(self) -> List[str]:
        return sorted(k for k, t in self.routing_type.items() if t == "specialist")


def _components_from_meta(meta: pd.DataFrame) -> pd.DataFrame:
    rows = [
        signature_tuple(row["Region"], row["Excitatory_vs_Inhibitory"], row["Segment"])
        for row in meta.loc[:, SIGNATURE_FIELDS].to_dict("records")
    ]
    return pd.DataFrame(
        rows,
        index=meta.index.astype(str),
        columns=list(YW002_METADATA_FIELDS),
    )


def train_model_v1(data: MerfishData, class_names: Sequence[str]) -> FittedModelV1:
    """Fit specialists and the global fallback on all labeled training cells."""
    y = data.y_train.astype(str)
    train_ids = [str(v) for v in y.index.tolist()]
    if len(train_ids) != N_TRAIN_CELLS:
        raise RuntimeError("Model V1 expected {} training cells, got {}".format(N_TRAIN_CELLS, len(train_ids)))
    if len(set(train_ids)) != N_TRAIN_CELLS:
        raise RuntimeError("Model V1 training Cell_IDs are not unique")
    gene_names = list(data.counts_train.columns)
    if len(gene_names) != N_GENES:
        raise RuntimeError("Model V1 expected {} genes, got {}".format(N_GENES, len(gene_names)))

    X_log = log1p_counts(data.counts_train.loc[train_ids])
    if X_log.shape != (N_TRAIN_CELLS, N_GENES):
        raise RuntimeError("log1p training matrix shape {} is invalid".format(X_log.shape))

    sigs = signatures_from_meta(data.meta_train.loc[train_ids])
    comps = _components_from_meta(data.meta_train.loc[train_ids])
    y_arr = y.loc[train_ids].to_numpy()
    cmap = build_candidate_map(sigs.to_numpy(), y_arr)

    signature_train_ids: Dict[str, List[str]] = {}
    signature_components: Dict[str, Tuple[str, str, str]] = {}
    routing_type: Dict[str, str] = {}
    for sid, sig in zip(train_ids, sigs.to_numpy()):
        signature_train_ids.setdefault(str(sig), []).append(sid)
    for sig, ids in signature_train_ids.items():
        row = comps.loc[ids[0]]
        signature_components[sig] = (
            str(row["Region"]),
            str(row["Excitatory_vs_Inhibitory"]),
            str(row["Segment"]),
        )
        routing_type[sig] = (
            "deterministic" if is_deterministic_map(cmap[sig]) else "specialist"
        )

    warn: List[str] = []
    global_model = make_logistic_regression()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        global_model.fit(X_log, y_arr)
        for item in caught:
            warn.append("global_fallback: {}: {}".format(item.category.__name__, item.message))

    specialists: Dict[str, LogisticRegression] = {}
    failures: Dict[str, str] = {}
    id_to_pos = {cid: i for i, cid in enumerate(train_ids)}
    for sig in sorted(k for k, t in routing_type.items() if t == "specialist"):
        local_ids = signature_train_ids[sig]
        local_pos = np.array([id_to_pos[cid] for cid in local_ids], dtype=int)
        y_local = y_arr[local_pos]
        n_local = int(len(local_ids))
        n_cls = int(len(set(y_local.tolist())))
        cands = cmap[sig]
        reason = None
        if n_cls < 2:
            reason = "fewer_than_two_target_classes"
        elif n_local < 2:
            reason = "insufficient_training_rows"
        elif n_cls < len(cands):
            reason = "required_class_absent"
        if reason is None:
            model_local = make_logistic_regression()
            try:
                with warnings.catch_warnings(record=True) as caught_local:
                    warnings.simplefilter("always")
                    model_local.fit(X_log[local_pos], y_local)
                    for item in caught_local:
                        warn.append(
                            "specialist {}: {}: {}".format(
                                sig, item.category.__name__, item.message
                            )
                        )
                if int(getattr(model_local, "n_features_in_", X_log.shape[1])) != N_GENES:
                    reason = "unexpected_feature_count"
                else:
                    specialists[sig] = model_local
            except Exception as exc:
                reason = "specialist_fit_failed: {}".format(exc)
        if reason is not None:
            failures[sig] = reason

    return FittedModelV1(
        class_names=list(class_names),
        gene_names=gene_names,
        train_cell_ids=train_ids,
        candidate_map=cmap,
        signature_components=signature_components,
        signature_train_ids=signature_train_ids,
        routing_type=routing_type,
        specialists=specialists,
        specialist_fit_failures=failures,
        global_fallback=global_model,
        n_features=N_GENES,
        lr_kwargs=dict(LR_KWARGS),
        n_train=N_TRAIN_CELLS,
        warnings=warn,
    )


def predict_model_v1(
    fitted: FittedModelV1,
    data: MerfishData,
    test_cell_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Predict official test cells. Uses test metadata signatures, never test labels."""
    test_ids = [str(v) for v in test_cell_ids]
    if len(test_ids) != N_TEST_CELLS:
        raise RuntimeError("expected {} test Cell_IDs, got {}".format(N_TEST_CELLS, len(test_ids)))
    train_set = set(fitted.train_cell_ids)
    overlap = set(test_ids) & train_set
    if overlap:
        raise RuntimeError("test Cell_IDs overlap training IDs: n={}".format(len(overlap)))

    counts_test = data.counts_test.loc[test_ids]
    if list(counts_test.columns) != fitted.gene_names:
        raise RuntimeError("test gene columns do not match training gene order")
    X_log = log1p_counts(counts_test)
    meta_sig_only = data.meta_test.loc[test_ids, SIGNATURE_FIELDS]
    test_sigs = signatures_from_meta(meta_sig_only).to_numpy()

    n = len(test_ids)
    k = len(fitted.class_names)
    proba = np.zeros((n, k), dtype=np.float64)
    routes = np.empty(n, dtype=object)
    n_det = n_spec = n_fallback = 0

    groups: Dict[str, List[int]] = {}
    for i, sig in enumerate(test_sigs):
        groups.setdefault(str(sig), []).append(i)

    for sig, idx_list in groups.items():
        idx = np.array(idx_list, dtype=int)
        if sig not in fitted.candidate_map:
            P = align_predict_proba(fitted.global_fallback, X_log[idx], fitted.class_names)
            proba[idx] = P
            routes[idx] = "fallback"
            n_fallback += int(len(idx))
            continue
        cands = fitted.candidate_map[sig]
        if is_deterministic_map(cands):
            lab = next(iter(cands))
            proba[idx] = one_hot_class(len(idx), lab, fitted.class_names)
            routes[idx] = "deterministic"
            n_det += int(len(idx))
            continue
        model = fitted.specialists.get(sig)
        if model is None:
            P = align_predict_proba(fitted.global_fallback, X_log[idx], fitted.class_names)
            masked = np.vstack(
                [mask_and_renormalize(row, fitted.class_names, cands)[0] for row in P]
            )
            proba[idx] = masked
            routes[idx] = "fallback"
            n_fallback += int(len(idx))
            continue
        P = align_predict_proba(model, X_log[idx], fitted.class_names)
        masked = np.vstack(
            [mask_and_renormalize(row, fitted.class_names, cands)[0] for row in P]
        )
        proba[idx] = masked
        routes[idx] = "specialist"
        n_spec += int(len(idx))

    assert_probability_rows(proba)
    pred = argmax_labels(proba, fitted.class_names)
    counts = {
        "test_cells_routed_deterministically": n_det,
        "test_cells_routed_to_specialists": n_spec,
        "fallback_count": n_fallback,
    }
    if n_det + n_spec + n_fallback != n:
        raise RuntimeError("routing counts do not cover all test cells")
    return pred, proba, counts


def signature_summary_frame(fitted: FittedModelV1) -> pd.DataFrame:
    rows = []
    for sig in sorted(fitted.routing_type):
        region, ei, segment = fitted.signature_components[sig]
        cands = sorted(fitted.candidate_map[sig])
        rows.append(
            {
                "Region": region,
                "Excitatory_vs_Inhibitory": ei,
                "Segment": segment,
                "training_cell_count": int(len(fitted.signature_train_ids[sig])),
                "candidate_class_count": int(len(cands)),
                "candidate_classes": ";".join(cands),
                "routing_type": fitted.routing_type[sig],
            }
        )
    return pd.DataFrame(rows)


def write_submission_csv(path, cell_ids: Sequence[str], pred: Sequence[str]) -> None:
    frame = pd.DataFrame(
        {
            SUBMISSION_CELL_ID_COL: [str(v) for v in cell_ids],
            SUBMISSION_PRED_COL: [str(v) for v in pred],
        }
    )
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise RuntimeError("submission columns {}".format(list(frame.columns)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
