from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import train_top5_candidate_reranker_decoder as metric_helpers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_WORK = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work"
)
CACHE_PATH = EXTERNAL_WORK / "cache_ext" / "static.npz"
ANCHOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "external_reference_fusion"
    / "oof_probabilities_external_primary_crossfit.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "biology_aware_glia_head_expanded_v2"

CENTRAL_GLIA = [
    "astrocyte_1",
    "astrocyte_2",
    "ependymal",
    "oligodendrocyte_1",
    "oligodendrocyte_2",
    "oligodendrocyte_precursor_cell",
    "oligodendrocyte_progenitor_1",
    "oligodendrocyte_progenitor_2",
]

NON_NEURONAL = [
    *CENTRAL_GLIA,
    "microglia",
    "endothelial",
    "pericyte",
    "Schwann_cell",
    "peripheral_glia",
    "meninges_1",
    "meninges_2",
    "meninges_3",
]

# Coefficients encode only directions supported by the spinal-cord literature and
# the current-panel diagnostic audit. They create features, not hard labels.
MODULES = {
    "cilia": {"Dnah12": 1.0, "Cfap43": 0.75, "P2rx6": 0.35},
    "early_ol": {
        "Tnr": 1.0,
        "Dll1": 0.50,
        "Grid1": 0.25,
        "Grik3": 0.20,
        "Ptgds": -0.30,
    },
    "transition_ol": {
        "Ptgds": 1.0,
        "Tnr": -0.45,
        "Rnf220": -0.20,
        "Sema4d": -0.20,
        "Cyp27a1": -0.15,
        "Pmp22": -0.15,
    },
    "mature_ol": {
        "Rnf220": 0.80,
        "Sema4d": 0.80,
        "Cyp27a1": 0.60,
        "Pmp22": 0.60,
        "Dpy19l1": 0.45,
        "Ptgds": -0.35,
        "Tnr": -0.20,
    },
    "astro_regular": {
        "Ntrk3": 1.0,
        "Notch2": 0.60,
        "Adcyap1r1": 0.30,
        "Ptgds": -0.65,
        "Sema6d": -0.20,
        "Col23a1": -0.15,
    },
    "astro_white_reactive": {
        "Sema6d": 1.0,
        "Col23a1": 0.65,
        "Ramp2": 0.25,
        "Ntrk3": -0.30,
    },
    "microglia": {
        "Aif1": 1.0,
        "Tyrobp": 1.0,
        "Trem2": 0.85,
        "Ctss": 0.80,
        "C3ar1": 0.55,
        "C5ar1": 0.45,
        "P2rx7": 0.30,
    },
    "endothelial": {
        "Ramp2": 1.0,
        "Calcrl": 0.90,
        "Plxnd1": 0.70,
        "Kng1": 0.45,
        "Cyp27a1": 0.25,
    },
    "perivascular": {
        "Ccn2": 1.0,
        "Adamts2": 0.70,
        "Col23a1": 0.55,
        "Fbn2": 0.45,
        "Kng1": 0.30,
    },
    "peripheral_myelin": {
        "Cdh19": 1.0,
        "Erbb2": 0.80,
        "Pmp22": 0.65,
        "Plxnb3": 0.25,
    },
    "meninges_border": {
        "Ccn2": 0.90,
        "Fbn2": 0.75,
        "Adamts2": 0.65,
        "Kng1": 0.50,
        "Ptgds": 0.35,
        "Col23a1": 0.30,
    },
    # This is intentionally weak and receives only a medium-tier correction.
    # In the current competition data, lower Prox1/Pcdhga11/Adamts2 is the
    # remaining signal for oligo-1 relative to progenitor-2.
    "oligo1_weak": {
        "Prox1": -1.0,
        "Pcdhga11": -0.70,
        "Adamts2": -0.65,
    },
}

AUXILIARY_FEATURES = [
    "log_total",
    "log_volume",
    "density",
    "rel_x",
    "rel_y",
    "sp_d1",
    "sp_d5",
    "sp_d15",
    "ex_d1",
    "ex_d10",
    "Region",
    "Segment",
    "AP",
]

PAIR_SPECS = [
    {
        "name": "ependymal_positive",
        "positive": "ependymal",
        "negative": None,
        "positive_module": "cilia",
        "negative_module": None,
        "tier": "high",
        "family": CENTRAL_GLIA,
    },
    {
        "name": "progenitor_1_vs_2",
        "positive": "oligodendrocyte_progenitor_1",
        "negative": "oligodendrocyte_progenitor_2",
        "positive_module": "early_ol",
        "negative_module": "transition_ol",
        "tier": "moderate",
    },
    {
        "name": "precursor_vs_progenitor_2",
        "positive": "oligodendrocyte_precursor_cell",
        "negative": "oligodendrocyte_progenitor_2",
        "positive_module": "early_ol",
        "negative_module": "transition_ol",
        "tier": "moderate",
    },
    {
        "name": "progenitor_2_vs_oligodendrocyte_2",
        "positive": "oligodendrocyte_progenitor_2",
        "negative": "oligodendrocyte_2",
        "positive_module": "transition_ol",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "oligodendrocyte_1_vs_2",
        "positive": "oligodendrocyte_1",
        "negative": "oligodendrocyte_2",
        "positive_module": "transition_ol",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "astrocyte_1_vs_progenitor_2",
        "positive": "astrocyte_1",
        "negative": "oligodendrocyte_progenitor_2",
        "positive_module": "astro_regular",
        "negative_module": "transition_ol",
        "tier": "moderate",
    },
    {
        "name": "astrocyte_1_vs_oligodendrocyte_1",
        "positive": "astrocyte_1",
        "negative": "oligodendrocyte_1",
        "positive_module": "astro_regular",
        "negative_module": "transition_ol",
        "tier": "high",
    },
    {
        "name": "astrocyte_1_vs_oligodendrocyte_2",
        "positive": "astrocyte_1",
        "negative": "oligodendrocyte_2",
        "positive_module": "astro_regular",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "precursor_vs_oligodendrocyte_2",
        "positive": "oligodendrocyte_precursor_cell",
        "negative": "oligodendrocyte_2",
        "positive_module": "early_ol",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "progenitor_1_vs_oligodendrocyte_2",
        "positive": "oligodendrocyte_progenitor_1",
        "negative": "oligodendrocyte_2",
        "positive_module": "early_ol",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "astrocyte_1_vs_2",
        "positive": "astrocyte_1",
        "negative": "astrocyte_2",
        "positive_module": "astro_regular",
        "negative_module": "astro_white_reactive",
        "tier": "moderate",
    },
    {
        "name": "oligodendrocyte_1_vs_progenitor_2_weak",
        "positive": "oligodendrocyte_1",
        "negative": "oligodendrocyte_progenitor_2",
        "positive_module": "oligo1_weak",
        "negative_module": None,
        "tier": "moderate",
    },
    {
        "name": "microglia_positive",
        "positive": "microglia",
        "negative": None,
        "positive_module": "microglia",
        "negative_module": None,
        "tier": "moderate",
        "family": NON_NEURONAL,
    },
    {
        "name": "endothelial_vs_astrocyte_1",
        "positive": "endothelial",
        "negative": "astrocyte_1",
        "positive_module": "endothelial",
        "negative_module": "astro_regular",
        "tier": "moderate",
    },
    {
        "name": "endothelial_vs_astrocyte_2",
        "positive": "endothelial",
        "negative": "astrocyte_2",
        "positive_module": "endothelial",
        "negative_module": "astro_white_reactive",
        "tier": "moderate",
    },
    {
        "name": "endothelial_vs_pericyte",
        "positive": "endothelial",
        "negative": "pericyte",
        "positive_module": "endothelial",
        "negative_module": "perivascular",
        "tier": "moderate",
    },
    {
        "name": "pericyte_positive",
        "positive": "pericyte",
        "negative": None,
        "positive_module": "perivascular",
        "negative_module": None,
        "tier": "moderate",
        "family": NON_NEURONAL,
    },
    {
        "name": "schwann_vs_oligodendrocyte_2",
        "positive": "Schwann_cell",
        "negative": "oligodendrocyte_2",
        "positive_module": "peripheral_myelin",
        "negative_module": "mature_ol",
        "tier": "high",
    },
    {
        "name": "peripheral_vs_oligodendrocyte_2",
        "positive": "peripheral_glia",
        "negative": "oligodendrocyte_2",
        "positive_module": "peripheral_myelin",
        "negative_module": "mature_ol",
        "tier": "moderate",
    },
    {
        "name": "meninges_1_positive",
        "positive": "meninges_1",
        "negative": None,
        "positive_module": "meninges_border",
        "negative_module": None,
        "tier": "moderate",
        "family": NON_NEURONAL,
    },
    {
        "name": "meninges_2_positive",
        "positive": "meninges_2",
        "negative": None,
        "positive_module": "meninges_border",
        "negative_module": None,
        "tier": "moderate",
        "family": NON_NEURONAL,
    },
    {
        "name": "meninges_3_positive",
        "positive": "meninges_3",
        "negative": None,
        "positive_module": "meninges_border",
        "negative_module": None,
        "tier": "moderate",
        "family": NON_NEURONAL,
    },
]

MODES = ["competition_only", "rare_reference"]
HIGH_CONFIDENCE_THRESHOLDS = [0.85, 0.90, 0.95]
HIGH_MAIN_MARGINS = [0.75, 1.00, 1.50]
MEDIUM_CONFIDENCE_THRESHOLDS = [0.65, 0.75]
MEDIUM_MAIN_MARGINS = [0.20, 0.50]
ALPHAS = [1.0, 2.0, 4.0]
MODERATE_WEIGHTS = [0.05, 0.10, 0.15, 0.25]
REFERENCE_LIMIT_PER_CLASS = 500
REFERENCE_WEIGHT = 0.15
EVIDENCE_CLIP = 4.0
SEED = 20260820


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_index(names: np.ndarray, name: str) -> int:
    candidates = [name, f"g_{name}"]
    for candidate in candidates:
        found = np.flatnonzero(names == candidate)
        if len(found):
            return int(found[0])
    raise KeyError(f"Missing required feature: {name}")


def standardized_columns(
    values: np.ndarray, fit_rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    fit = values[np.ix_(fit_rows, columns)]
    mean = np.nanmean(fit, axis=0)
    std = np.nanstd(fit, axis=0)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0
    transformed = (values[:, columns] - mean) / std
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(transformed, -10.0, 10.0).astype(np.float32)


def build_fold_features(
    values: np.ndarray, names: np.ndarray, fit_rows: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    genes = sorted({gene for definition in MODULES.values() for gene in definition})
    gene_columns = np.asarray([feature_index(names, gene) for gene in genes])
    gene_z = standardized_columns(values, fit_rows, gene_columns)
    gene_position = {gene: position for position, gene in enumerate(genes)}

    module_values: dict[str, np.ndarray] = {}
    for module_name, definition in MODULES.items():
        score = np.zeros(len(values), dtype=np.float32)
        normalizer = 0.0
        for gene, coefficient in definition.items():
            score += coefficient * gene_z[:, gene_position[gene]]
            normalizer += abs(coefficient)
        module_values[module_name] = score / max(normalizer, 1e-6)

    auxiliary_columns = np.asarray(
        [feature_index(names, name) for name in AUXILIARY_FEATURES]
    )
    auxiliary = standardized_columns(values, fit_rows, auxiliary_columns)
    return module_values, auxiliary, AUXILIARY_FEATURES.copy()


def pair_inputs(
    spec: dict[str, object],
    modules: dict[str, np.ndarray],
    auxiliary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    positive_module = str(spec["positive_module"])
    negative_module = spec["negative_module"]
    main = modules[positive_module].copy()
    excluded = {positive_module}
    if negative_module is not None:
        negative_module = str(negative_module)
        main -= modules[negative_module]
        excluded.add(negative_module)
    other_names = [name for name in MODULES if name not in excluded]
    other_modules = np.column_stack([modules[name] for name in other_names])
    inputs = np.column_stack([main, other_modules, auxiliary]).astype(np.float32)
    input_names = ["main_biology_axis", *other_names, *AUXILIARY_FEATURES]
    return inputs, main.astype(np.float32), input_names


class MonotonicBiologyHead(nn.Module):
    def __init__(self, auxiliary_dim: int) -> None:
        super().__init__()
        self.raw_main_weight = nn.Parameter(torch.tensor(0.0))
        self.auxiliary = nn.Linear(auxiliary_dim, 1)
        nn.init.zeros_(self.auxiliary.weight)
        nn.init.zeros_(self.auxiliary.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        main = inputs[:, :1]
        auxiliary = inputs[:, 1:]
        main_weight = F.softplus(self.raw_main_weight) + 0.25
        return (main_weight * main + self.auxiliary(auxiliary)).squeeze(1)


def sample_rows(
    rows: np.ndarray, limit: int, rng: np.random.Generator
) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    return rng.choice(rows, size=limit, replace=False)


def training_rows_for_spec(
    spec: dict[str, object],
    labels: np.ndarray,
    competition_rows: np.ndarray,
    reference_rows: np.ndarray,
    class_index: dict[str, int],
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    positive = class_index[str(spec["positive"])]
    negative_name = spec["negative"]
    if negative_name is None:
        family_names = spec.get("family", CENTRAL_GLIA)
        family_indices = np.asarray([class_index[name] for name in family_names])
        competition_mask = np.isin(labels[competition_rows], family_indices)
        competition_selected = competition_rows[competition_mask]
        competition_target = (labels[competition_selected] == positive).astype(np.float32)
        positive_count = int(competition_target.sum())
        negative_count = int(len(competition_target) - positive_count)
    else:
        negative = class_index[str(negative_name)]
        competition_mask = np.isin(labels[competition_rows], [positive, negative])
        competition_selected = competition_rows[competition_mask]
        competition_target = (labels[competition_selected] == positive).astype(np.float32)
        positive_count = int(competition_target.sum())
        negative_count = int(len(competition_target) - positive_count)

    fit_rows = competition_selected.copy()
    targets = competition_target.copy()
    source_weights = np.ones(len(fit_rows), dtype=np.float32)
    reference_added = 0
    if mode == "rare_reference" and min(positive_count, negative_count) < 100:
        positive_reference = reference_rows[labels[reference_rows] == positive]
        positive_reference = sample_rows(
            positive_reference, REFERENCE_LIMIT_PER_CLASS, rng
        )
        if negative_name is None:
            family_names = spec.get("family", CENTRAL_GLIA)
            family_indices = np.asarray([class_index[name] for name in family_names])
            negative_reference = reference_rows[
                np.isin(labels[reference_rows], family_indices)
                & (labels[reference_rows] != positive)
            ]
            negative_reference = sample_rows(
                negative_reference, 2 * REFERENCE_LIMIT_PER_CLASS, rng
            )
        else:
            negative = class_index[str(negative_name)]
            negative_reference = reference_rows[labels[reference_rows] == negative]
            negative_reference = sample_rows(
                negative_reference, REFERENCE_LIMIT_PER_CLASS, rng
            )
        added = np.concatenate([positive_reference, negative_reference])
        added_targets = (labels[added] == positive).astype(np.float32)
        fit_rows = np.concatenate([fit_rows, added])
        targets = np.concatenate([targets, added_targets])
        source_weights = np.concatenate(
            [
                source_weights,
                np.full(len(added), REFERENCE_WEIGHT, dtype=np.float32),
            ]
        )
        reference_added = int(len(added))

    return fit_rows, targets, source_weights, {
        "competition_positive": positive_count,
        "competition_negative": negative_count,
        "reference_added": reference_added,
    }


def fit_predict_head(
    inputs: np.ndarray,
    fit_rows: np.ndarray,
    targets: np.ndarray,
    source_weights: np.ndarray,
    query_rows: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    set_seed(seed)
    positive_weight = len(targets) / max(2.0 * float(targets.sum()), 1.0)
    negative_weight = len(targets) / max(
        2.0 * float(len(targets) - targets.sum()), 1.0
    )
    class_weights = np.where(targets > 0.5, positive_weight, negative_weight)
    weights = source_weights * class_weights.astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)

    x = torch.as_tensor(inputs[fit_rows], dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    query = torch.as_tensor(inputs[query_rows], dtype=torch.float32, device=device)
    model = MonotonicBiologyHead(inputs.shape[1] - 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.02)
    final_loss = 0.0
    for _ in range(300):
        logits = model(x)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
        loss = loss + 1e-3 * model.auxiliary.weight.abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(query)).cpu().numpy().astype(np.float32)
        main_weight = float(F.softplus(model.raw_main_weight).cpu() + 0.25)
    return probabilities, final_loss, main_weight


def apply_heads(
    anchor: np.ndarray,
    anchor_prediction: np.ndarray,
    top5: np.ndarray,
    pair_probabilities: np.ndarray,
    pair_main_scores: np.ndarray,
    pair_classes: list[tuple[int, int | None]],
    pair_families: list[np.ndarray],
    high_confidence_threshold: float,
    high_main_margin: float,
    medium_confidence_threshold: float,
    medium_main_margin: float,
    alpha: float,
    moderate_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    logits = np.log(np.clip(anchor, 1e-12, None))
    application_masks: dict[str, np.ndarray] = {}
    total_applications = 0
    for pair_index, spec in enumerate(PAIR_SPECS):
        is_moderate = spec["tier"] == "moderate"
        pair_alpha = alpha * (moderate_weight if is_moderate else 1.0)
        required_confidence = (
            medium_confidence_threshold if is_moderate else high_confidence_threshold
        )
        required_margin = medium_main_margin if is_moderate else high_main_margin
        positive, negative = pair_classes[pair_index]
        probability = np.clip(pair_probabilities[:, pair_index], 1e-6, 1.0 - 1e-6)
        main = pair_main_scores[:, pair_index]
        if negative is None:
            candidate = np.isin(top5, positive).any(axis=1)
            candidate &= anchor_prediction != positive
            candidate &= np.isin(anchor_prediction, pair_families[pair_index])
            active = candidate & (probability >= required_confidence) & (main >= required_margin)
            rows = np.flatnonzero(active)
            evidence = np.clip(
                np.log(probability[rows]) - np.log1p(-probability[rows]),
                0.0,
                EVIDENCE_CLIP,
            )
            opponents = anchor_prediction[rows]
            logits[rows, positive] += 0.5 * pair_alpha * evidence
            logits[rows, opponents] -= 0.5 * pair_alpha * evidence
        else:
            candidate = np.isin(anchor_prediction, [positive, negative])
            candidate &= np.isin(top5, positive).any(axis=1)
            candidate &= np.isin(top5, negative).any(axis=1)
            head_positive = probability >= 0.5
            head_prediction = np.where(head_positive, positive, negative)
            confidence = np.maximum(probability, 1.0 - probability)
            directional_support = np.where(
                head_positive, main >= required_margin, main <= -required_margin
            )
            active = (
                candidate
                & (head_prediction != anchor_prediction)
                & (confidence >= required_confidence)
                & directional_support
            )
            rows = np.flatnonzero(active)
            evidence = np.clip(
                np.log(probability[rows]) - np.log1p(-probability[rows]),
                -EVIDENCE_CLIP,
                EVIDENCE_CLIP,
            )
            logits[rows, positive] += 0.5 * pair_alpha * evidence
            logits[rows, negative] -= 0.5 * pair_alpha * evidence
        application_masks[str(spec["name"])] = active
        total_applications += int(active.sum())

    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32), application_masks, total_applications


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache = np.load(CACHE_PATH, allow_pickle=True)
    values = cache["X"].astype(np.float32)
    names = cache["names"].astype(str)
    ids = cache["ids"].astype(str)
    labels = cache["y"].astype(np.int64)
    class_names = cache["labels"].astype(str).tolist()
    class_index = {name: index for index, name in enumerate(class_names)}
    is_train = cache["is_train"].astype(bool)
    is_reference = cache["is_ref"].astype(bool)
    folds_universe = cache["folds"].astype(np.int64)
    train_rows = np.flatnonzero(is_train)
    reference_rows = np.flatnonzero(is_reference)
    train_labels = labels[train_rows]
    train_folds = folds_universe[train_rows]

    anchor_frame = pd.read_csv(ANCHOR_PATH, dtype={"Cell_ID": str}).set_index("Cell_ID")
    probability_columns = [f"p__{label}" for label in class_names]
    anchor = anchor_frame.reindex(ids[train_rows])[probability_columns].to_numpy(
        dtype=np.float32
    )
    anchor = np.clip(anchor, 1e-12, None)
    anchor /= anchor.sum(axis=1, keepdims=True)
    anchor_prediction = anchor.argmax(axis=1)
    top5 = np.argsort(-anchor, axis=1)[:, :5]

    pair_classes: list[tuple[int, int | None]] = []
    pair_families: list[np.ndarray] = []
    for spec in PAIR_SPECS:
        positive = class_index[str(spec["positive"])]
        negative = (
            None if spec["negative"] is None else class_index[str(spec["negative"])]
        )
        pair_classes.append((positive, negative))
        family_names = spec.get("family", CENTRAL_GLIA)
        pair_families.append(
            np.asarray([class_index[name] for name in family_names], dtype=np.int64)
        )

    pair_oof = {
        mode: np.zeros((len(train_rows), len(PAIR_SPECS)), dtype=np.float32)
        for mode in MODES
    }
    pair_main_oof = np.zeros((len(train_rows), len(PAIR_SPECS)), dtype=np.float32)
    training_report: list[dict[str, object]] = []

    for fold in range(5):
        competition_fit = train_rows[train_folds != fold]
        validation_rows = train_rows[train_folds == fold]
        validation_local = np.flatnonzero(train_folds == fold)
        modules, auxiliary, _ = build_fold_features(values, names, competition_fit)
        for pair_index, spec in enumerate(PAIR_SPECS):
            inputs, main_scores, input_names = pair_inputs(spec, modules, auxiliary)
            pair_main_oof[validation_local, pair_index] = main_scores[validation_rows]
            for mode_index, mode in enumerate(MODES):
                rng = np.random.default_rng(
                    SEED + fold * 1000 + pair_index * 20 + mode_index
                )
                fit_rows, targets, source_weights, counts = training_rows_for_spec(
                    spec,
                    labels,
                    competition_fit,
                    reference_rows,
                    class_index,
                    mode,
                    rng,
                )
                predicted, final_loss, main_weight = fit_predict_head(
                    inputs,
                    fit_rows,
                    targets,
                    source_weights,
                    validation_rows,
                    device,
                    SEED + fold * 1000 + pair_index * 20 + mode_index,
                )
                pair_oof[mode][validation_local, pair_index] = predicted
                training_report.append(
                    {
                        "fold": fold,
                        "mode": mode,
                        "pair": spec["name"],
                        "tier": spec["tier"],
                        **counts,
                        "final_loss": final_loss,
                        "learned_positive_main_weight": main_weight,
                        "input_features": "|".join(input_names),
                    }
                )

    anchor_metrics = metric_helpers.evaluate(
        train_labels, anchor, anchor_prediction, len(class_names)
    )
    configuration_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    probability_by_configuration: dict[str, np.ndarray] = {}
    masks_by_configuration: dict[str, dict[str, np.ndarray]] = {}

    sweep = product(
        MODES,
        HIGH_CONFIDENCE_THRESHOLDS,
        HIGH_MAIN_MARGINS,
        MEDIUM_CONFIDENCE_THRESHOLDS,
        MEDIUM_MAIN_MARGINS,
        ALPHAS,
        MODERATE_WEIGHTS,
    )
    for (
        mode,
        high_confidence_threshold,
        high_main_margin,
        medium_confidence_threshold,
        medium_main_margin,
        alpha,
        moderate_weight,
    ) in sweep:
        name = (
            f"{mode}_hc{high_confidence_threshold:g}_hm{high_main_margin:g}"
            f"_mc{medium_confidence_threshold:g}_mm{medium_main_margin:g}"
            f"_a{alpha:g}_medium{moderate_weight:g}"
        )
        probabilities, masks, applications = apply_heads(
            anchor,
            anchor_prediction,
            top5,
            pair_oof[mode],
            pair_main_oof,
            pair_classes,
            pair_families,
            high_confidence_threshold,
            high_main_margin,
            medium_confidence_threshold,
            medium_main_margin,
            alpha,
            moderate_weight,
        )
        metrics = metric_helpers.evaluate(
            train_labels,
            probabilities,
            anchor_prediction,
            len(class_names),
        )
        invoked = np.zeros(len(train_rows), dtype=bool)
        for mask in masks.values():
            invoked |= mask
        positive_folds = 0
        for fold in range(5):
            fold_mask = train_folds == fold
            fold_metrics = metric_helpers.evaluate(
                train_labels[fold_mask],
                probabilities[fold_mask],
                anchor_prediction[fold_mask],
                len(class_names),
            )
            positive_folds += int(fold_metrics["net_corrections"] > 0)
            fold_rows.append(
                {
                    "configuration": name,
                    "fold": fold,
                    **fold_metrics,
                }
            )
        configuration_rows.append(
            {
                "configuration": name,
                "mode": mode,
                "high_confidence_threshold": high_confidence_threshold,
                "high_main_margin": high_main_margin,
                "medium_confidence_threshold": medium_confidence_threshold,
                "medium_main_margin": medium_main_margin,
                "alpha": alpha,
                "moderate_weight": moderate_weight,
                "invoked_cells": int(invoked.sum()),
                "head_applications": applications,
                "positive_folds": positive_folds,
                **metrics,
            }
        )
        probability_by_configuration[name] = probabilities
        masks_by_configuration[name] = masks

    configuration_table = pd.DataFrame(configuration_rows).sort_values(
        ["accuracy", "macro_f1", "positive_folds", "net_corrections"],
        ascending=False,
    )
    best_name = str(configuration_table.iloc[0]["configuration"])
    best_row = configuration_table.iloc[0]
    best_probabilities = probability_by_configuration[best_name]
    best_prediction = best_probabilities.argmax(axis=1)

    pair_audit_rows: list[dict[str, object]] = []
    for spec in PAIR_SPECS:
        mask = masks_by_configuration[best_name][str(spec["name"])]
        anchor_correct = anchor_prediction == train_labels
        new_correct = best_prediction == train_labels
        pair_audit_rows.append(
            {
                "pair": spec["name"],
                "tier": spec["tier"],
                "applications": int(mask.sum()),
                "fixed": int((mask & ~anchor_correct & new_correct).sum()),
                "harmed": int((mask & anchor_correct & ~new_correct).sum()),
                "net": int((mask & ~anchor_correct & new_correct).sum())
                - int((mask & anchor_correct & ~new_correct).sum()),
            }
        )

    class_audit_rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        mask = train_labels == class_id
        if not mask.any():
            continue
        class_audit_rows.append(
            {
                "class_name": class_name,
                "support": int(mask.sum()),
                "anchor_accuracy": float((anchor_prediction[mask] == class_id).mean()),
                "new_accuracy": float((best_prediction[mask] == class_id).mean()),
                "delta": float(
                    (best_prediction[mask] == class_id).mean()
                    - (anchor_prediction[mask] == class_id).mean()
                ),
            }
        )

    configuration_table.to_csv(OUTPUT_DIR / "configuration_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(training_report).to_csv(OUTPUT_DIR / "head_training.csv", index=False)
    pd.DataFrame(pair_audit_rows).to_csv(OUTPUT_DIR / "pair_audit.csv", index=False)
    pd.DataFrame(class_audit_rows).to_csv(OUTPUT_DIR / "class_audit.csv", index=False)
    metric_helpers.probability_frame(
        pd.Index(ids[train_rows]), best_probabilities, class_names
    ).to_csv(OUTPUT_DIR / "oof_probabilities_best_exploratory.csv", index=False)

    success = {
        "accuracy_above_anchor": bool(float(best_row["accuracy"]) > anchor_metrics["accuracy"]),
        "net_corrections_positive": bool(int(best_row["net_corrections"]) > 0),
        "at_least_four_positive_folds": bool(int(best_row["positive_folds"]) >= 4),
        "macro_f1_not_lower": bool(float(best_row["macro_f1"]) >= anchor_metrics["macro_f1"]),
    }
    metrics = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(0),
            "outer_folds": 5,
            "architecture": "monotonic biology-axis pair heads with weak auxiliary anatomy terms",
            "strong_correction": "only Anchor Top-5 candidate pairs with head disagreement, high posterior confidence, and a directionally consistent biology-axis margin",
            "expanded_priors": "central glia plus microglia/endothelial/pericyte/Schwann/peripheral/meninges; weak oligo-1 vs progenitor-2 evidence is medium-tier only",
            "modes": MODES,
            "high_confidence_thresholds": HIGH_CONFIDENCE_THRESHOLDS,
            "high_main_margins": HIGH_MAIN_MARGINS,
            "medium_confidence_thresholds": MEDIUM_CONFIDENCE_THRESHOLDS,
            "medium_main_margins": MEDIUM_MAIN_MARGINS,
            "alphas": ALPHAS,
            "moderate_weights": MODERATE_WEIGHTS,
            "reference_weight_for_rare_classes": REFERENCE_WEIGHT,
            "selection_caveat": "The best configuration is exploratory because thresholds are selected on the same OOF predictions; fold consistency is reported separately.",
        },
        "anchor": anchor_metrics,
        "best_exploratory_configuration": best_name,
        "best_exploratory_metrics": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in best_row.to_dict().items()
        },
        "success_tests": success,
        "deployment": "No test probabilities are produced unless the biology head improves accuracy with positive net corrections in at least four folds and does not reduce macro-F1.",
        "pair_audit": pair_audit_rows,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
