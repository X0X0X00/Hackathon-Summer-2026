"""Score ANY test set with the frozen models in final_artifacts/.

Usage: python predict_final.py [--meta PATH] [--counts PATH] [--out PATH]
Defaults: ../data/meta_test.csv, ../data/counts_test.csv, ../prediction/prediction.csv
Rows are written in the order of the meta file, header identical to the official template.
"""
import sys, json, time
import numpy as np, pandas as pd
from final_features import WORK, BASE, ART, build_universe, featurize
from final_blend import blend_predict, load_and_predict


def run_predict(meta_path=None, counts_path=None, out_path=None, verbose=True):
    t0 = time.time()
    genes = json.load(open(ART / "genes.json"))
    prep = json.load(open(ART / "prep.json"))
    hold = json.load(open(ART / "holdout_A_ids.json"))
    U = build_universe(genes, meta_path, counts_path, exclude_ids=hold); U["genes"] = genes
    # Integrity check: cells that were LABELLED REFERENCE rows when the models were trained must not be
    # scored by those models (train_final.py excludes every competition id, so the fix is to re-run it).
    trained_test = set(json.load(open(ART / "trained_with_test_ids.json")))
    train_ids = set(U["mtr"].index.astype(str)); cur_test = set(U["mte"].index.astype(str))
    dep_ids = set(U["ids"][:U["n_dep"]])
    overlap = (cur_test & dep_ids) - trained_test - train_ids - set(hold)
    if overlap and "--allow-overlap" not in sys.argv:
        raise SystemExit(f"{len(overlap)} test cells were reference (labelled) rows when the frozen models were trained. "
                         "Re-run `python train_final.py` (it excludes all current competition ids) and predict again.")
    F = featurize(U, prep)
    known = np.where(U["is_ref"] | U["is_train"], U["y"], -1).astype(np.int64)
    te = np.where(U["is_test"])[0]
    probs = load_and_predict(F, known, te)
    res = blend_predict(probs, F, known, te, prep)
    ids_te = U["ids"][te]
    mte = U["mte"]
    order = pd.Series(np.arange(len(ids_te)), index=ids_te).loc[mte.index.astype(str)].values
    labels = np.array(prep["labels"])
    out = pd.DataFrame({"Cell_ID": mte.index.values, "MERFISH_cell_type_annotation.y": labels[res["pred"][order]]})
    if out_path:
        out.to_csv(out_path, index=False)
    if verbose:
        tiers = {k: int(v) for k, v in zip(*np.unique(res["tier"], return_counts=True))}
        new_secs = len(set(mte.Section_ID.astype(str)) - set(U["mtr"].Section_ID.astype(str)))
        print(f"test cells={len(te)}  in deposit={U['test_in_deposit']}  new={U['test_new']}  "
              f"sections not in train={new_secs}  labelled-spatial-nbr median={np.median(res['n_sp']):.0f}  "
              f"meta_frac={res['meta_frac']:.2f}  tiers={tiers}  ({time.time()-t0:.0f}s)", flush=True)
        if out_path: print("wrote", out_path)
    return {"ids": mte.index.values, "labels_pred": out.iloc[:, 1].values, "prob": res["prob"][order],
            "tier": res["tier"][order], "classes": labels}


if __name__ == "__main__":
    args = sys.argv[1:]
    def opt(flag, default):
        return args[args.index(flag) + 1] if flag in args else default
    run_predict(opt("--meta", BASE / "data/meta_test.csv"), opt("--counts", BASE / "data/counts_test.csv"),
                opt("--out", BASE / "prediction/prediction.csv"))
