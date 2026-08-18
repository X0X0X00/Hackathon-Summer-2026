"""Ensemble all OOF prob files: greedy weight search on OOF acc, then apply
E/I (and optionally Region) constraints. Writes final test probs.

Usage: python ensemble.py [--exclude name1,name2]
"""
import sys
import numpy as np
from pathlib import Path
from common import load, apply_ei, OOF

D = load()
tr = np.where(D["is_train"])[0]
te = np.where(~D["is_train"])[0]
y_tr = D["y"][tr]

exclude = set()
if "--exclude" in sys.argv:
    exclude = set(sys.argv[sys.argv.index("--exclude") + 1].split(","))

files = sorted(OOF.glob("*.npz"))
models = {}
for f in files:
    if f.stem in exclude:
        continue
    z = np.load(f)
    models[f.stem] = {"oof": z["oof"], "test": z["test"], "acc": float(z["acc"])}
    print(f"{f.stem:16s} acc={float(z['acc']):.4f}")
assert models, "no oof files"

names = sorted(models, key=lambda k: -models[k]["acc"])


def acc_of(p):
    return (p.argmax(1) == y_tr).mean()


def acc_ei_of(p):
    return (apply_ei(p, D["ei_known"][tr], D["ei_of_label"]).argmax(1) == y_tr).mean()


# ---- greedy forward ensemble with replacement (Caruana) ----
pool = np.zeros_like(models[names[0]]["oof"])
picks = []
best_hist = []
cur = 0.0
for step in range(30):
    best_gain, best_name = -1, None
    for nm in names:
        cand = (pool * len(picks) + models[nm]["oof"]) / (len(picks) + 1)
        a = acc_ei_of(cand)
        if a > best_gain:
            best_gain, best_name = a, nm
    if picks and best_gain <= cur + 1e-6:
        break
    picks.append(best_name)
    pool = (pool * (len(picks) - 1) + models[best_name]["oof"]) / len(picks)
    cur = best_gain
    best_hist.append((best_name, cur))
print("\ngreedy picks:", [f"{n}({a:.4f})" for n, a in best_hist])

w = {nm: picks.count(nm) / len(picks) for nm in set(picks)}
print("weights:", {k: round(v, 3) for k, v in sorted(w.items(), key=lambda x: -x[1])})

oof_ens = sum(models[nm]["oof"] * wt for nm, wt in w.items())
test_ens = sum(models[nm]["test"] * wt for nm, wt in w.items())
print(f"\nensemble OOF acc      = {acc_of(oof_ens):.4f}")
print(f"ensemble OOF acc +EI  = {acc_ei_of(oof_ens):.4f}")

# ---- optional Region hard-constraint check (classes absent from a region) ----
reg = D["region_known"]
reg_tr, y_full = reg[tr], D["y"][tr]
n_reg = int(np.nanmax(reg)) + 1
present = np.zeros((n_reg, 60), bool)
for r in range(n_reg):
    m = reg_tr == r
    present[r, np.unique(y_full[m])] = True
    print(f"Region {r}: {m.sum()} labeled cells, {present[r].sum()}/60 classes present")

oof_reg = oof_ens.copy()
for r in range(n_reg):
    rows = np.where(reg_tr == r)[0]
    sub = oof_reg[rows].copy()
    sub[:, ~present[r]] = 0
    z = sub.sum(1)
    ok = z > 0
    oof_reg[rows[ok]] = sub[ok] / z[ok, None]
a_reg = acc_ei_of(oof_reg)
print(f"ensemble OOF acc +EI +RegionMask = {a_reg:.4f}")

np.savez_compressed(OOF.parent / "final_probs.npz",
                    oof=oof_ens, test=test_ens,
                    weights=np.array([f"{k}:{v}" for k, v in w.items()]))
print("\nsaved work/final_probs.npz")
