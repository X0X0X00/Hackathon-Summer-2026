"""Dress rehearsals for the frozen pipeline on synthetic 'validation' sets with known labels.

  A  : 5000 deposit cells held out of the reference at train time (same sections/mice as train)
  A2 : same cells, Region / E-I / Segment blanked (metadata-missing scenario)
  C  : 5000 SNI-dataset cells (new mice, new sections, no Region/E-I/Segment); labels = 'voting'
       consensus of 5 transfer methods (noisy ground truth -> treat accuracy as a lower bound)
Usage: python rehearse.py [A|A2|C ...]
"""
import sys, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from final_features import WORK, BASE, ART, DEPOSIT, norm_label
from predict_final import run_predict

OUT = WORK / "rehearsal"; OUT.mkdir(exist_ok=True)
genes = json.load(open(ART / "genes.json")); prep = json.load(open(ART / "prep.json"))
inv = lambda m: {v: k for k, v in m.items()}


def write_set(name, ids, counts, meta, labels):
    d = OUT / name; d.mkdir(exist_ok=True)
    pd.DataFrame(counts, index=ids, columns=genes).to_csv(d / "counts_test.csv")
    meta.index = ids; meta.to_csv(d / "meta_test.csv")
    pd.Series(labels, index=ids, name="label").to_csv(d / "labels.csv")
    return d


def scenario_A(blank_meta=False):
    hold = json.load(open(ART / "holdout_A_ids.json"))
    a = ad.read_h5ad(DEPOSIT); a.obs.index = a.obs.index.astype(str)
    sub = a[hold]
    X = sub[:, genes].X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(int)
    o = sub.obs
    meta = pd.DataFrame({
        "Datasets": o["Datasets"].astype(str).values, "volume": o["volume"].astype(float).values,
        "center_x": o["center_x"].astype(float).values, "center_y": o["center_y"].astype(float).values,
        "MERFISH_cell_type_annotation": np.nan,
        "Region": o["Region"].astype(str).map(prep["region_map"]).values,
        "Excitatory_vs_Inhibitory": o["Excitatory_vs_Inhibitory"].astype(str).replace("nan", np.nan).values,
        "Segment": o["Laminae"].astype(str).map(prep["segment_map"]).values,
        "Gender": o["Gender"].astype(str).values, "Mouse_ID": o["Mouse ID"].astype(str).values,
        "AP_position": o["Axial level"].astype(str).map(prep["ap_map"]).values,
        "Section_ID": o["Section ID"].astype(str).values})
    if blank_meta:
        meta[["Region", "Excitatory_vs_Inhibitory", "Segment"]] = np.nan
    labels = o["MERFISH cell type annotation"].astype(str).map(norm_label).values
    return write_set("A2" if blank_meta else "A", np.array(hold, dtype=str), X, meta, labels)


def scenario_C(n=5000, seed=0):
    a = ad.read_h5ad(WORK / "external/SNI_merged_0917.h5ad"); a.obs.index = a.obs.index.astype(str)
    rng = np.random.default_rng(seed); idx = rng.choice(a.n_obs, n, replace=False)
    sub = a[idx]
    X = sub[:, genes].X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(int)
    o = sub.obs
    mouse = o["Mouse ID"].astype(str).values
    meta = pd.DataFrame({
        "Datasets": o["batch"].astype(str).values, "volume": o["volume"].astype(float).values,
        "center_x": o["center_x"].astype(float).values, "center_y": o["center_y"].astype(float).values,
        "MERFISH_cell_type_annotation": np.nan, "Region": np.nan, "Excitatory_vs_Inhibitory": np.nan, "Segment": np.nan,
        "Gender": ["male" if "_M" in m else "female" for m in mouse], "Mouse_ID": mouse,
        "AP_position": o["Axial level"].astype(str).map(prep["ap_map"]).values,
        "Section_ID": o["Custom Cells groups"].astype(str).values})
    labels = o["voting"].astype(str).map(norm_label).values
    ids = np.array(["SNI_" + str(i) for i in sub.obs.index], dtype=str)   # guaranteed not in deposit
    return write_set("C", ids, X, meta, labels)


def score(d):
    res = run_predict(d / "meta_test.csv", d / "counts_test.csv", d / "prediction.csv")
    lab = pd.read_csv(d / "labels.csv", index_col=0).label.astype(str).values
    ok = lab != "nan"
    acc = (res["labels_pred"][ok] == lab[ok]).mean()
    tiers = pd.Series(res["tier"]).value_counts().to_dict()
    print(f"### {d.name}: accuracy={acc:.4f} on {ok.sum()} labelled cells  tiers={tiers}")
    for t in sorted(tiers):
        m = (res["tier"] == t) & ok
        if m.sum(): print(f"     tier {t}: acc={np.mean(res['labels_pred'][m] == lab[m]):.4f} (n={m.sum()})")
    return acc


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["A", "A2", "C"]):
        d = {"A": lambda: scenario_A(False), "A2": lambda: scenario_A(True), "C": scenario_C}[s]()
        score(d)
