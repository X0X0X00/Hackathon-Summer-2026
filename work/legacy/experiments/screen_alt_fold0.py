"""Screen diverse alt models on fold 0 only (baseline features sp_k=15, ex_k=25).

Leakage-free: known labels exclude fold 0 when building neighbor histograms,
model fits on folds 1-4, evaluates on fold 0. Reuses common.build_X.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load, build_X

D = load()
y, folds, is_tr = D["y"], D["folds"], D["is_train"]
tr = np.where(is_tr)[0]
F = 0
known = np.where(is_tr & (folds != F), y, -1).astype(np.int64)
X, names = build_X(D, known, sp_k=15, ex_k=25)
X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)  # inf -> NaN
fit = tr[folds[tr] != F]
va = tr[folds[tr] == F]
Xtr, ytr, Xva, yva = X[fit], y[fit], X[va], y[va]
print(f"fold0 screen: fit={len(fit)} va={len(va)} d={X.shape[1]}", flush=True)

# histogram-only column block for the kNN-style model
hist_cols = np.array([i for i, nm in enumerate(names)
                      if nm.startswith(("sp_h", "ex_h", "sp_n", "ex_n", "sp_d", "ex_d"))])
print(f"hist cols: {len(hist_cols)}", flush=True)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def ev(tag, model, cols=None):
    t0 = time.time()
    a, b = (Xtr, Xva) if cols is None else (Xtr[:, cols], Xva[:, cols])
    model.fit(a, ytr)
    acc = (model.predict_proba(b).argmax(1) == yva).mean()
    print(f"{tag:44s} acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return acc


which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "hgb"):
    for lr, it in [(0.1, 200), (0.06, 400)]:
        ev(f"hgb lr={lr} iters={it}",
           HistGradientBoostingClassifier(learning_rate=lr, max_iter=it,
                                          early_stopping=False, random_state=0))

if which in ("all", "logreg"):
    for C in [0.03, 0.1, 0.3, 1.0]:
        ev(f"logreg C={C}",
           make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, n_jobs=4)))

if which in ("all", "mlp"):
    for hid in [(256,), (512, 256)]:
        ev(f"mlp hidden={hid}",
           make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=hid, early_stopping=True,
                                       n_iter_no_change=12, max_iter=300,
                                       random_state=0)))

if which in ("all", "knn"):
    for C in [0.3, 1.0, 3.0, 10.0]:
        ev(f"knn-hist logreg C={C}",
           make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, n_jobs=4)),
           cols=hist_cols)
