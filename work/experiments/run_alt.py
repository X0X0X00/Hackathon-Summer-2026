"""Diverse alt models on baseline features (sp_k=15, ex_k=25), full 5-fold CV.

Configs picked by fold-0 screening (experiments/screen_alt_fold0.py):
  alt_hgb    HistGradientBoosting lr=0.1 iters=60 l2=1.0     (fold0 0.754)
             NOTE: sklearn HGB default l2_regularization=0 DIVERGES on this
             60-class problem (va acc decays from 0.62@iter1 to 0.158);
             l2=1.0 fixes it.
  alt_logreg LogisticRegression C=0.1 on imputed+standardized (fold0 0.572)
  alt_mlp    MLP 256, alpha=1.0, adam lr 1e-3, early stopping (fold0 0.597)
  alt_knn    LogisticRegression C=0.01 on ONLY the 124 neighbor-label
             histogram cols (kNN-style; fold0 0.475 - weak by nature:
             plain neighbor-vote argmax is only ~0.45 on 60 classes)
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load, run_cv, save_oof, N_CLASSES

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

D = load()


def full60(model, Xva):
    """predict_proba padded to all 60 classes (rare classes can miss a fit set)."""
    p = model.predict_proba(Xva)
    if p.shape[1] == N_CLASSES:
        return p
    out = np.zeros((len(Xva), N_CLASSES), p.dtype)
    cls = model[-1].classes_ if hasattr(model, "steps") else model.classes_
    out[:, np.asarray(cls, int)] = p
    return out


def clean(X):
    return np.where(np.isfinite(X), X, np.nan)


def fp_hgb(Xtr, ytr, Xva, names):
    m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=60,
                                       l2_regularization=1.0, early_stopping=False,
                                       random_state=0)
    m.fit(clean(Xtr), ytr)
    return full60(m, clean(Xva))


def fp_logreg(Xtr, ytr, Xva, names):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=3000, n_jobs=4))
    m.fit(clean(Xtr), ytr)
    return full60(m, clean(Xva))


def fp_mlp(Xtr, ytr, Xva, names):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0,
                                    learning_rate_init=1e-3, batch_size=256,
                                    early_stopping=True, tol=0, n_iter_no_change=30,
                                    max_iter=500, random_state=0))
    m.fit(clean(Xtr), ytr)
    return full60(m, clean(Xva))


def make_fp_knn():
    def fp(Xtr, ytr, Xva, names):
        hc = np.array([i for i, nm in enumerate(names)
                       if nm.startswith(("sp_h", "ex_h", "sp_n", "ex_n"))
                       or nm in ("sp_d", "ex_d")])
        assert len(hc) == 124, len(hc)
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.01, max_iter=3000, n_jobs=4))
        m.fit(clean(Xtr[:, hc]), ytr)
        return full60(m, clean(Xva[:, hc]))
    return fp


JOBS = [("alt_knn", make_fp_knn()), ("alt_logreg", fp_logreg),
        ("alt_mlp", fp_mlp), ("alt_hgb", fp_hgb)]
only = sys.argv[1:] if len(sys.argv) > 1 else None

for name, fp in JOBS:
    if only and name not in only:
        continue
    print(f"=== {name} ===", flush=True)
    res = run_cv(D, fp, sp_k=15, ex_k=25)
    save_oof(name, res)
