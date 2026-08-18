"""Post-processing shared by evaluation and submission (competition row order):
  1. E/I hard constraint (common.apply_ei)
  2. Segment (=Laminae) hard mask: a cell with observed Segment s may only take labels that
     occur with s among REFERENCE cells (table built from oof-free reference labels only;
     0 true-label violations on the 5000 competition-train cells).
"""
import numpy as np
import pandas as pd
from common import load, apply_ei
from common_ext import load_ext

_cache = {}


def _tables():
    if "tab" in _cache:
        return _cache["tab"]
    D = load(); E = load_ext(); names = E["names"]
    seg = E["X"][:, names.index("Segment")]; ref = E["is_ref"]; ys = E["y"]
    tab = {}
    for s in np.unique(seg[ref & ~np.isnan(seg)]):
        tab[int(s)] = np.unique(ys[ref & (seg == s)]).tolist()
    ids_e = pd.Series(np.arange(len(E["ids"])), index=E["ids"].astype(str))
    tr = np.where(D["is_train"])[0]; te = np.where(~D["is_train"])[0]
    seg_tr = seg[ids_e.loc[D["ids"][tr].astype(str)].values]
    seg_te = seg[ids_e.loc[D["ids"][te].astype(str)].values]
    _cache["tab"] = (D, tab, seg_tr, seg_te)
    return _cache["tab"]


def segment_mask(segv, tab):
    M = np.ones((len(segv), 60), bool)
    for i, s in enumerate(segv):
        if np.isnan(s):
            continue
        M[i] = False; M[i, tab[int(s)]] = True
    return M


def postprocess(probs, which="test"):
    """probs: (n,60) competition-order probs for 'train' (OOF) or 'test'. Returns masked probs."""
    D, tab, seg_tr, seg_te = _tables()
    if which == "test":
        rows = np.where(~D["is_train"])[0]; segv = seg_te
    else:
        rows = np.where(D["is_train"])[0]; segv = seg_tr
    p = apply_ei(probs, D["ei_known"][rows], D["ei_of_label"])
    M = segment_mask(segv, tab)
    p2 = np.where(M, p, 0.0)
    bad = p2.sum(1) <= 0
    p2[bad] = p[bad]
    return p2 / p2.sum(1, keepdims=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else "final_probs.npz")
    D, tab, seg_tr, seg_te = _tables()
    tr = np.where(D["is_train"])[0]; y = D["y"][tr]; folds = D["folds"][tr]
    p0 = apply_ei(z["oof"], D["ei_known"][tr], D["ei_of_label"]).argmax(1)
    p1 = postprocess(z["oof"], "train").argmax(1)
    print(f"OOF  EI-only={np.mean(p0==y):.4f}  +Segment mask={np.mean(p1==y):.4f}   f34 {np.mean((p0==y)[folds>=3]):.4f} -> {np.mean((p1==y)[folds>=3]):.4f}")
