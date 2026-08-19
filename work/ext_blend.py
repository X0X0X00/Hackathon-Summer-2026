"""Align + evaluate + blend competition-order members (oof/*.npz) with universe-order members
(oof_ext/*.npz).  Usage: python ext_blend.py [member ...]  (default: a standard set)
Prints per-member acc/+EI overall, folds0-2, folds3-4 and a few fixed (no-search) blends.
Writes final_probs.npz for a chosen blend if --save NAME=members,comma,list is given.
"""
import sys
import numpy as np
from pathlib import Path
from common import load, apply_ei, OOF
from common_ext import load_ext, OOF as OOF_EXT

D = load()
tr = np.where(D["is_train"])[0]; te = np.where(~D["is_train"])[0]
y = D["y"][tr]; folds = D["folds"][tr]
ids_tr = D["ids"][tr].astype(str); ids_te = D["ids"][te].astype(str)
E = load_ext()
e_ids_tr = E["ids"][E["is_train"]].astype(str); e_ids_te = E["ids"][E["is_test"]].astype(str)
pos_tr = {i: k for k, i in enumerate(e_ids_tr)}; pos_te = {i: k for k, i in enumerate(e_ids_te)}
map_tr = np.array([pos_tr[i] for i in ids_tr]); map_te = np.array([pos_te[i] for i in ids_te])


def load_member(name):
    """Return (oof, test) in competition order."""
    if (OOF / f"{name}.npz").exists():
        z = np.load(OOF / f"{name}.npz"); return z["oof"], z["test"]
    z = np.load(OOF_EXT / f"{name}.npz", allow_pickle=True)
    if name.startswith("yhh"):        # already competition order
        return z["oof"], z["test"]
    return z["oof"][map_tr], z["test"][map_te]


def aei(p, rows=slice(None)):
    return (apply_ei(p[rows], D["ei_known"][tr][rows], D["ei_of_label"]).argmax(1) == y[rows]).mean()


def report(name, p):
    print(f"{name:34s} all={aei(p):.4f}  f012={aei(p, folds<=2):.4f}  f34={aei(p, folds>=3):.4f}  "
          f"perfold={np.round([aei(p, folds==f) for f in range(5)], 3).tolist()}")


args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = [a for a in sys.argv[1:] if a.startswith("--save")]
members = args or ["poolAll", "ext_comp", "ext_all25", "ext_mlp"]
M = {}
for m in members:
    try:
        M[m] = load_member(m)
    except FileNotFoundError:
        print(f"(missing {m})")
print("== members ==")
for m, (o, t) in M.items(): report(m, o)

def blend(names, w=None):
    w = w or [1] * len(names)
    o = sum(M[n][0] * wi for n, wi in zip(names, w)) / sum(w)
    t = sum(M[n][1] * wi for n, wi in zip(names, w)) / sum(w)
    return o, t

print("== fixed blends ==")
avail = [m for m in members if m in M]
combos = []
if "poolAll" in M:
    for m in avail:
        if m != "poolAll": combos.append(([ "poolAll", m], None))
    ext = [m for m in avail if m != "poolAll"]
    if len(ext) >= 2: combos.append((["poolAll"] + ext, None))
    if len(ext) >= 2: combos.append((["poolAll"] + ext, [len(ext)] + [1] * len(ext)))
if len(avail) >= 2: combos.append((avail, None))
for names, w in combos:
    o, t = blend(names, w)
    report("+".join(names) + (f" w={w}" if w else ""), o)

for s in save:
    spec = s.split("=", 1)[1]
    nm, lst = spec.split(":", 1) if ":" in spec else ("final", spec)
    names = lst.split(",")
    o, t = blend(names)
    np.savez_compressed("final_probs.npz", oof=o, test=t, weights=np.array([f"{n}:1" for n in names]))
    print(f"saved final_probs.npz = mean({names})  all={aei(o):.4f} f34={aei(o, folds>=3):.4f}")
