from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token")
ANCHOR = ROOT / "outputs" / "external_reference_fusion"
MODES = ("categorical", "topology_full", "topology_incremental", "topology_incremental_shuffled")
WEIGHTS = (0.05, 0.10, 0.20)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p.astype(np.float64), 1e-12, None); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1); f1 = []; support = []
    for c in range(p.shape[1]):
        truth, guess = y == c, pred == c
        tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
        d = 2 * tp + fp + fn; f1.append(0.0 if d == 0 else 2 * tp / d); support.append(int(truth.sum()))
    f1, support = np.asarray(f1), np.asarray(support)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "weighted_f1": float((f1 * support).sum() / support.sum()),
            "log_loss": float(-np.log(p[np.arange(len(y)), y]).mean())}


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0: return 1.0
    logs = np.asarray([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2) for k in range(min(gained, lost) + 1)])
    top = float(logs.max())
    return min(1.0, 2 * math.exp(top) * float(np.exp(logs - top).sum()))


def paired(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    a, b = base.argmax(1), candidate.argmax(1); ac, bc = a == y, b == y
    gained, lost = int((~ac & bc).sum()), int((ac & ~bc).sum())
    am, bm = metrics(y, base), metrics(y, candidate)
    return {"accuracy_delta": bm["accuracy"] - am["accuracy"], "macro_f1_delta": bm["macro_f1"] - am["macro_f1"],
            "log_loss_delta": bm["log_loss"] - am["log_loss"], "gained": gained, "lost": lost,
            "net": gained - lost, "changed_predictions": int((a != b).sum()), "mcnemar_exact_p": mcnemar(gained, lost)}


def load_probabilities(path: Path, ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
    columns = [f"p__{c}" if f"p__{c}" in frame.columns else c for c in classes]
    p = frame.loc[ids, columns].to_numpy(np.float64); p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def save_probabilities(path: Path, ids: np.ndarray, p: np.ndarray, classes: np.ndarray) -> None:
    frame = pd.DataFrame(p, columns=[f"p__{c}" for c in classes]); frame.insert(0, "Cell_ID", ids); frame.to_csv(path, index=False)


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    id_column = "Cell_ID" if "Cell_ID" in frame.columns else frame.columns[0]
    frame[id_column] = frame[id_column].astype(str)
    return frame.set_index(id_column)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy()


def build_segment_graph(meta_all: pd.DataFrame, segments: np.ndarray, k: int = 2) -> tuple[np.ndarray, list[dict[str, object]]]:
    position = {int(value): index for index, value in enumerate(segments)}
    graph = np.zeros((len(segments), len(segments)), np.float64)
    edge_support = np.zeros_like(graph)
    frame = meta_all.copy()
    frame["_segment"] = pd.to_numeric(frame["Segment"], errors="coerce")
    frame["_x"] = pd.to_numeric(frame["center_x"], errors="coerce")
    frame["_y"] = pd.to_numeric(frame["center_y"], errors="coerce")
    frame = frame.dropna(subset=["_segment", "_x", "_y", "Section_ID"])
    for _, section in frame.groupby("Section_ID"):
        centers = section.groupby("_segment")[["_x", "_y"]].mean()
        available = [int(v) for v in centers.index if int(v) in position]
        if len(available) < 2: continue
        coords = centers.loc[available].to_numpy(np.float64)
        scale = np.maximum(coords.std(0), 1e-6); coords = (coords - coords.mean(0)) / scale
        distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)); np.fill_diagonal(distance, np.inf)
        for source in range(len(available)):
            for target in np.argsort(distance[source])[:min(k, len(available) - 1)]:
                i, j = position[available[source]], position[available[target]]
                weight = math.exp(-float(distance[source, target]))
                graph[i, j] += weight; graph[j, i] += weight
                edge_support[i, j] += 1; edge_support[j, i] += 1
    graph = (graph + graph.T) / 2
    row_sum = graph.sum(1, keepdims=True)
    graph = np.divide(graph, row_sum, out=np.zeros_like(graph), where=row_sum > 0)
    rows = []
    for i, source in enumerate(segments):
        for j, target in enumerate(segments):
            if graph[i, j] > 0:
                rows.append({"source_segment": int(source), "target_segment": int(target),
                             "normalized_weight": float(graph[i, j]), "section_edge_support": int(edge_support[i, j])})
    return graph.astype(np.float64), rows


def ap_graph(ap_values: np.ndarray) -> np.ndarray:
    graph = np.zeros((len(ap_values), len(ap_values)), np.float64)
    for i, value in enumerate(ap_values):
        for j, other in enumerate(ap_values):
            if abs(int(value) - int(other)) == 1: graph[i, j] = 1.0
    graph /= np.maximum(graph.sum(1, keepdims=True), 1)
    return graph


def fit_counts(meta: pd.DataFrame, y: np.ndarray, segments: np.ndarray, aps: np.ndarray, classes: int) -> dict[str, np.ndarray]:
    segment_pos = {int(v): i for i, v in enumerate(segments)}; ap_pos = {int(v): i for i, v in enumerate(aps)}
    state = np.zeros((len(segments), len(aps), classes), np.float64); ap_count = np.zeros((len(aps), classes), np.float64)
    segment_values, ap_values = numeric(meta, "Segment"), numeric(meta, "AP_position")
    for row, label in enumerate(y):
        a = ap_pos.get(int(ap_values[row])) if np.isfinite(ap_values[row]) else None
        if a is None: continue
        ap_count[a, label] += 1
        s = segment_pos.get(int(segment_values[row])) if np.isfinite(segment_values[row]) else None
        if s is not None: state[s, a, label] += 1
    global_count = np.bincount(y, minlength=classes).astype(np.float64)
    return {"state": state, "ap": ap_count, "global": global_count}


def priors(meta: pd.DataFrame, fitted: dict[str, np.ndarray], segments: np.ndarray, aps: np.ndarray,
           segment_graph: np.ndarray, ap_edges: np.ndarray, tau: float,
           segment_strength: float, ap_strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_pos = {int(v): i for i, v in enumerate(segments)}; ap_pos = {int(v): i for i, v in enumerate(aps)}
    state, ap_count, global_count = fitted["state"], fitted["ap"], fitted["global"]
    global_p = (global_count + 1.0) / (global_count.sum() + len(global_count))
    categorical = np.empty((len(meta), len(global_count)), np.float64); topology = np.empty_like(categorical)
    segment_values, ap_values = numeric(meta, "Segment"), numeric(meta, "AP_position")
    for row in range(len(meta)):
        a = ap_pos.get(int(ap_values[row])) if np.isfinite(ap_values[row]) else None
        s = segment_pos.get(int(segment_values[row])) if np.isfinite(segment_values[row]) else None
        if a is None:
            cat_count = global_count.copy(); topo_count = cat_count.copy()
        elif s is None:
            cat_count = ap_count[a].copy()
            topo_count = cat_count + ap_strength * np.tensordot(ap_edges[a], ap_count, axes=(0, 0))
        else:
            cat_count = state[s, a].copy()
            segment_neighbor = np.tensordot(segment_graph[s], state[:, a, :], axes=(0, 0))
            ap_neighbor = np.tensordot(ap_edges[a], state[s, :, :], axes=(0, 0))
            topo_count = cat_count + segment_strength * segment_neighbor + ap_strength * ap_neighbor
        categorical[row] = (cat_count + tau * global_p) / (cat_count.sum() + tau)
        topology[row] = (topo_count + tau * global_p) / (topo_count.sum() + tau)
    return categorical, topology, np.repeat(global_p[None, :], len(meta), axis=0)


def residual(mode: str, categorical: np.ndarray, topology: np.ndarray, global_p: np.ndarray,
             shuffled_topology: np.ndarray) -> np.ndarray:
    if mode == "categorical": value = np.log(categorical) - np.log(global_p)
    elif mode == "topology_full": value = np.log(topology) - np.log(global_p)
    elif mode == "topology_incremental": value = np.log(topology) - np.log(categorical)
    else: value = np.log(shuffled_topology) - np.log(categorical)
    value -= value.mean(1, keepdims=True)
    return np.clip(value, -3.0, 3.0)


def inject(anchor: np.ndarray, value: np.ndarray, weight: float) -> np.ndarray:
    logits = np.log(np.clip(anchor, 1e-12, None)) + weight * value; logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); return p / p.sum(1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "anatomical_topology_prior")
    parser.add_argument("--tau", type=float, default=20.0); parser.add_argument("--segment-strength", type=float, default=.75)
    parser.add_argument("--ap-strength", type=float, default=.35); parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)

    ids = np.load(DATA / "ids.npy").astype(str); labels = np.load(DATA / "labels.npy").astype(np.int64)
    fold_all = np.load(DATA / "folds.npy").astype(np.int64); train_pos = np.load(DATA / "train_positions.npy")
    test_pos = np.load(DATA / "test_positions.npy"); classes = np.load(DATA / "class_names.npy").astype(str)
    train_ids, test_ids, y, folds = ids[train_pos], ids[test_pos], labels[train_pos], fold_all[train_pos]
    meta_train = load_metadata(ROOT / "data" / "meta_train.csv").reindex(train_ids)
    meta_test = load_metadata(ROOT / "data" / "meta_test.csv").reindex(test_ids)
    meta_all = pd.concat([meta_train, meta_test], axis=0)
    segments = np.sort(pd.to_numeric(meta_all["Segment"], errors="coerce").dropna().astype(int).unique())
    aps = np.sort(pd.to_numeric(meta_all["AP_position"], errors="coerce").dropna().astype(int).unique())
    segment_edges, edge_rows = build_segment_graph(meta_all, segments); ap_edges = ap_graph(aps)
    rng = np.random.default_rng(args.seed); permutation = rng.permutation(len(segments))
    shuffled_edges = segment_edges[np.ix_(permutation, permutation)]
    anchor_oof = load_probabilities(ANCHOR / "oof_probabilities_external_primary_crossfit.csv", train_ids, classes)
    anchor_test = load_probabilities(ANCHOR / "test_probabilities_external_primary_crossfit.csv", test_ids, classes)
    oof = {(mode, weight): np.zeros_like(anchor_oof) for mode in MODES for weight in WEIGHTS}
    test_fold = {(mode, weight): [] for mode in MODES if "shuffled" not in mode for weight in WEIGHTS}

    for fold in np.sort(np.unique(folds)):
        held = folds == fold; fitted = fit_counts(meta_train.loc[~held], y[~held], segments, aps, len(classes))
        cat, topo, global_p = priors(meta_train.loc[held], fitted, segments, aps, segment_edges, ap_edges, args.tau, args.segment_strength, args.ap_strength)
        _, shuffled, _ = priors(meta_train.loc[held], fitted, segments, aps, shuffled_edges, ap_edges, args.tau, args.segment_strength, args.ap_strength)
        cat_test, topo_test, global_test = priors(meta_test, fitted, segments, aps, segment_edges, ap_edges, args.tau, args.segment_strength, args.ap_strength)
        for mode in MODES:
            value = residual(mode, cat, topo, global_p, shuffled)
            for weight in WEIGHTS: oof[(mode, weight)][held] = inject(anchor_oof[held], value, weight)
            if "shuffled" not in mode:
                test_value = residual(mode, cat_test, topo_test, global_test, topo_test)
                for weight in WEIGHTS: test_fold[(mode, weight)].append(inject(anchor_test, test_value, weight))

    anchor_metrics = metrics(y, anchor_oof); rows = [{"configuration": "anchor", "mode": "anchor", "weight": 0, "control": False, **anchor_metrics,
                                                     "accuracy_delta": 0., "macro_f1_delta": 0., "log_loss_delta": 0., "gained": 0, "lost": 0,
                                                     "net": 0, "changed_predictions": 0, "mcnemar_exact_p": 1.}]
    for mode in MODES:
        for weight in WEIGHTS:
            candidate = oof[(mode, weight)]
            rows.append({"configuration": f"{mode}_{weight:g}", "mode": mode, "weight": weight,
                         "control": "shuffled" in mode, **metrics(y, candidate), **paired(y, anchor_oof, candidate)})

    topology_rows = []
    for weight in WEIGHTS:
        category, topo = oof[("categorical", weight)], oof[("topology_full", weight)]
        topology_rows.append({"weight": weight, **paired(y, category, topo)})
    known = pd.to_numeric(meta_train["Segment"], errors="coerce").notna().to_numpy()
    stratum_rows = []
    for stratum, mask in [("segment_known", known), ("segment_missing", ~known)]:
        for mode in ("categorical", "topology_full", "topology_incremental"):
            for weight in WEIGHTS:
                stratum_rows.append({"stratum": stratum, "n_cells": int(mask.sum()), "configuration": f"{mode}_{weight:g}",
                                     **metrics(y[mask], oof[(mode, weight)][mask])})

    for mode in ("categorical", "topology_full", "topology_incremental"):
        for weight in WEIGHTS:
            name = f"{mode}_{weight:g}"; test_p = np.stack(test_fold[(mode, weight)]).mean(0)
            save_probabilities(output / f"oof_probabilities_{name}.csv", train_ids, oof[(mode, weight)], classes)
            save_probabilities(output / f"test_probabilities_{name}.csv", test_ids, test_p, classes)
            pd.DataFrame({"Cell_ID": test_ids, "CellType": classes[test_p.argmax(1)]}).to_csv(output / f"submission_{name}.csv", index=False)

    pd.DataFrame(rows).to_csv(output / "configuration_metrics.csv", index=False)
    pd.DataFrame(topology_rows).to_csv(output / "topology_vs_categorical.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output / "segment_presence_metrics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output / "segment_topology_edges.csv", index=False)
    report = {"protocol": {"topology": "empirical k=2 Segment-centroid adjacency within Section plus ordered adjacent AP nodes",
                           "labels_for_prior": "outer-fold competition training labels only", "unlabeled_graph_population": "train + test metadata",
                           "modes": MODES, "fixed_weights": WEIGHTS, "tau": args.tau, "segment_strength": args.segment_strength,
                           "ap_strength": args.ap_strength, "n_segment_nodes": len(segments), "n_ap_nodes": len(aps),
                           "segment_known_fraction": float(known.mean()), "selection": "none"},
              "anchor": anchor_metrics, "configurations": rows, "topology_vs_categorical": topology_rows}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

