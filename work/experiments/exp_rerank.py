"""Top-2 RERANKER post-processor (generalizes the glial-pair specialist).

For every cell take the post-EI top-1/top-2 classes (c1, c2) of a base prob
matrix. Train a per-fold binary LGBM  "is the true label c2 rather than c1?"
on the train cells whose true label is in {c1, c2}  (label = y == c2), with
features = common.build_X (fold-aware, fold-f labels masked -> leakage-free,
same X run_cv uses) + [p1, p2, margin, p3, c1, c2 (categorical), family codes
of c1/c2 (categorical), same-family flag] and, in the 'rel' variants, the
pair-relative neighbor-hist values sp_h[c1], sp_h[c2], ex_h[c1], ex_h[c2].
At inference the c1/c2 probabilities are swapped where P(swap) > tau (and
margin < cap).  tau/cap are tuned on folds 0-2 ONLY; folds 3-4 are the honest
holdout number.

Usage:
  python experiments/exp_rerank.py explore [base.npz]      # all variants + specialist comparison
  python experiments/exp_rerank.py run <base.npz> [--variant meta|base|rel|metapure]
        [--tau T] [--cap C] [--top6] [--force] [--out path]
      refit against the given RAW base probs (npz with oof/test), tune tau/cap
      on folds 0-2, print holdout 3-4, and save
      experiments/rerank_correction.npz {oof, test, tau, cap, variant, note}
      (post-EI corrected probs, D-order train rows) -- ONLY if the holdout
      3-4 delta is > +0.001 (or --force).

RESULT on oof/bag8mix.npz (round 3, see rerank_explore.log / rerank_explore2.log):
  the reranker has NO information beyond the base model's own margin
  (AUC for "y==c2 | y in top-2": margin-only 0.8145, reranker 0.805-0.817 for
  every variant).  Every variant's holdout 3-4 delta is in [-0.003, +0.001]
  (a handful of cells), tau=0.5 fixed is net NEGATIVE in every fold, and the
  reverse split (tune 3-4, eval 0-2) is also ~0.  Not adopted.
  The 5-pair specialist re-tuned on folds 0-2 against bag8mix is -0.0030 on
  folds 3-4 (per-fold [2, 7, 7, -2, -4]); its earlier +0.0017 (tune 0-1, eval
  2-4) replicates only because folds 2 responds positively; folds 3-4 do not.

Reranker probs are cached per (variant, base file hash) in
experiments/rerank_cache/; delete to force retraining.
"""
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, build_X, apply_ei, N_CLASSES, WORK  # noqa: E402

N_JOBS = 4
SEEDS = (0, 1, 2)
TUNE_FOLDS = (0, 1, 2)
HOLD_FOLDS = (3, 4)
TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CAP_GRID = [0.10, 0.20, 0.30, 0.50, 0.75, 1.01]
PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=15,
    min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, max_bin=127, verbosity=-1,
    num_threads=N_JOBS, seed=0,
)
ROUNDS = 300
CACHE_DIR = WORK / "experiments" / "rerank_cache"
FAM_NAMES = ["other", "oligo", "astro", "OPC", "endothelial", "neuronal", "meninges"]


# --------------------------------------------------------------------------- utils
def family_codes(D):
    fam = np.zeros(N_CLASSES, np.int64)
    for i, lab in enumerate(D["labels"]):
        s = str(lab).lower()
        if s.startswith("oligodendrocyte_progenitor") or s == "oligodendrocyte_precursor_cell":
            fam[i] = 3
        elif s.startswith("oligodendrocyte"):
            fam[i] = 1
        elif s.startswith("astrocyte"):
            fam[i] = 2
        elif s == "endothelial":
            fam[i] = 4
        elif D["ei_of_label"][i] >= 0 or "motoneuron" in s or "interneuron" in s:
            fam[i] = 5
        elif s.startswith("meninges"):
            fam[i] = 6
    return fam


def top2(P):
    """(c1, c2, p1, p2, p3) per row."""
    order = np.argsort(-P, axis=1)[:, :3]
    rows = np.arange(len(P))
    c1, c2, c3 = order[:, 0], order[:, 1], order[:, 2]
    return c1, c2, P[rows, c1], P[rows, c2], P[rows, c3]


def load_base(path, D, tr, te):
    z = np.load(path)
    oof, test = z["oof"].astype(np.float64), z["test"].astype(np.float64)
    assert oof.shape == (len(tr), N_CLASSES) and test.shape == (len(te), N_CLASSES)
    assert np.allclose(oof.sum(1), 1, atol=1e-3) and np.allclose(test.sum(1), 1, atol=1e-3)
    P = apply_ei(oof, D["ei_known"][tr], D["ei_of_label"])
    P_te = apply_ei(test, D["ei_known"][te], D["ei_of_label"])
    h = hashlib.md5(np.ascontiguousarray(oof).tobytes()).hexdigest()[:10]
    return P, P_te, h


def confusion_pairs(c1, y, mask, k):
    """Top-k symmetric confused (true, pred) pairs computed on rows in mask."""
    C = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(C, (y[mask], c1[mask]), 1)
    pairs = []
    for a in range(N_CLASSES):
        for b in range(a + 1, N_CLASSES):
            m = C[a, b] + C[b, a]
            if m:
                pairs.append((m, a, b))
    pairs.sort(reverse=True)
    return [(a, b) for _, a, b in pairs[:k]]


def pair_mask(c1, c2, pairs):
    m = np.zeros(len(c1), bool)
    for a, b in pairs:
        m |= ((c1 == a) & (c2 == b)) | ((c1 == b) & (c2 == a))
    return m


# --------------------------------------------------------------------------- features
def meta_block(P, fam):
    c1, c2, p1, p2, p3 = top2(P)
    return c1, c2, np.column_stack([
        p1, p2, p1 - p2, p3, c1, c2, fam[c1], fam[c2], (fam[c1] == fam[c2]),
    ]).astype(np.float64)


META_NAMES = ["p1", "p2", "margin", "p3", "c1", "c2", "fam1", "fam2", "samefam"]
CAT_META = ["c1", "c2", "fam1", "fam2"]


def rel_block(X, names, c1, c2):
    sp = np.array([names.index(f"sp_h{c}") for c in range(N_CLASSES)])
    ex = np.array([names.index(f"ex_h{c}") for c in range(N_CLASSES)])
    r = np.arange(len(c1))
    return np.column_stack([X[r, sp[c1]], X[r, sp[c2]], X[r, ex[c1]], X[r, ex[c2]],
                            X[r, sp[c1]] - X[r, sp[c2]], X[r, ex[c1]] - X[r, ex[c2]]])


REL_NAMES = ["sp_c1", "sp_c2", "ex_c1", "ex_c2", "sp_diff", "ex_diff"]


def assemble(X, names, meta, c1, c2, variant):
    """Feature matrix for a set of rows (X already row-subset). Returns (F, fnames, cat_idx)."""
    parts, fn = [], []
    if not variant.startswith("meta"):
        parts.append(X); fn += [n.replace(" ", "_") for n in names]
    parts.append(meta); fn += META_NAMES
    if "rel" in variant or variant == "meta":
        parts.append(rel_block(X, names, c1, c2)); fn += REL_NAMES
    F = np.hstack(parts)
    cat = [fn.index(c) for c in CAT_META]
    return F, fn, cat


# --------------------------------------------------------------------------- reranker
def fit_predict(F_fit, y_fit, F_pred, cat, seeds=SEEDS, rounds=ROUNDS, override=None):
    out = np.zeros(len(F_pred))
    for s in seeds:
        p = dict(PARAMS, seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
        if override:
            p.update(override)
        ds = lgb.Dataset(F_fit, y_fit, categorical_feature=cat, free_raw_data=False)
        m = lgb.train(p, ds, num_boost_round=rounds)
        out += m.predict(F_pred)
    return out / len(seeds)


def train_reranker(D, tr, te, P, P_te, variant, elig_extra=None, base_hash="x",
                   rounds=ROUNDS, verbose=True, override=None, fit_cap=None):
    """Leakage-free per-fold P(swap) for every train row (+ full-train model
    for test rows). elig_extra: optional bool mask over train rows further
    restricting the training set / application (e.g. top-k pairs)."""
    CACHE_DIR.mkdir(exist_ok=True)
    tag = variant + ("" if elig_extra is None else f"_e{int(elig_extra.sum())}")
    if override:
        tag += "_" + "_".join(f"{k[:3]}{v}" for k, v in sorted(override.items()))
    if fit_cap is not None:
        tag += f"_fc{fit_cap}"
    cache_f = CACHE_DIR / f"{tag}_r{rounds}_{base_hash}.npz"
    if cache_f.exists():
        z = np.load(cache_f)
        if verbose:
            print(f"  (loaded reranker probs from {cache_f.name})")
        return z["pswap"], z["pswap_te"]
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    y_tr, fold_tr = y[tr], folds[tr]
    fam = family_codes(D)
    c1, c2, meta = meta_block(P, fam)
    c1t, c2t, meta_te = meta_block(P_te, fam)
    elig = (y_tr == c1) | (y_tr == c2)
    if elig_extra is not None:
        elig &= elig_extra
    if fit_cap is not None:                     # train only on ambiguous rows
        elig &= (meta[:, 2] < fit_cap)
    lab = (y_tr == c2).astype(np.int8)
    pswap = np.full(len(tr), np.nan)
    pswap_te = np.full(len(te), np.nan)
    for f in range(5):
        t0 = time.time()
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build_X(D, known)
        F, fn, cat = assemble(X[tr], names, meta, c1, c2, variant)
        fit = (fold_tr != f) & elig
        va = fold_tr == f
        pswap[va] = fit_predict(F[fit], lab[fit], F[va], cat, rounds=rounds, override=override)
        if verbose:
            print(f"  fold{f}: fit n={fit.sum()} pos={lab[fit].sum()} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, names = build_X(D, known)
    F, fn, cat = assemble(X[tr], names, meta, c1, c2, variant)
    Ft, _, _ = assemble(X[te], names, meta_te, c1t, c2t, variant)
    pswap_te = fit_predict(F[elig], lab[elig], Ft, cat, rounds=rounds, override=override)
    np.savez_compressed(cache_f, pswap=pswap, pswap_te=pswap_te)
    return pswap, pswap_te


def apply_swap(P, pswap, tau, cap=1.01, mask=None):
    """Swap c1/c2 probs where P(swap) > tau and margin < cap. Returns (P', n_swaps)."""
    c1, c2, p1, p2, _ = top2(P)
    do = (pswap > tau) & ((p1 - p2) < cap)
    if mask is not None:
        do &= mask
    out = P.copy()
    r = np.where(do)[0]
    out[r, c1[r]], out[r, c2[r]] = P[r, c2[r]], P[r, c1[r]]
    return out, len(r)


def tune(P, pswap, y_tr, fold_tr, mask=None, tune_folds=TUNE_FOLDS, use_cap=True):
    """(tau, cap) maximizing net correct-count gain on tune folds."""
    c1, c2, p1, p2, _ = top2(P)
    in_t = np.isin(fold_tr, tune_folds)
    if mask is not None:
        in_t &= mask
    gain = np.where(y_tr == c2, 1, np.where(y_tr == c1, -1, 0))
    best = (0, 0.5, 1.01)
    for cap in (CAP_GRID if use_cap else [1.01]):
        for tau in TAU_GRID:
            do = in_t & (pswap > tau) & ((p1 - p2) < cap)
            net = int(gain[do].sum())
            if net > best[0]:
                best = (net, tau, cap)
    return best


def evaluate(P, P_new, y_tr, fold_tr, label):
    a0 = P.argmax(1) == y_tr
    a1 = P_new.argmax(1) == y_tr
    per = [int(a1[fold_tr == f].sum() - a0[fold_tr == f].sum()) for f in range(5)]
    hold = np.isin(fold_tr, HOLD_FOLDS)
    tun = np.isin(fold_tr, TUNE_FOLDS)
    h0, h1 = a0[hold].mean(), a1[hold].mean()
    print(f"  {label:38s} full {a0.mean():.4f}->{a1.mean():.4f} ({a1.mean()-a0.mean():+.4f}) | "
          f"tune0-2 {a0[tun].mean():.4f}->{a1[tun].mean():.4f} ({a1[tun].mean()-a0[tun].mean():+.4f}) | "
          f"HOLDOUT3-4 {h0:.4f}->{h1:.4f} ({h1-h0:+.4f}) | per-fold {per}")
    return h1 - h0, a1.mean() - a0.mean(), per


# --------------------------------------------------------------------------- specialist baseline
def specialist_holdout(D, tr, P, y_tr, fold_tr):
    """Existing pair-specialist (cached specialist probs) re-tuned on folds
    0-2 against THIS base, evaluated on 3-4 -- apples-to-apples comparison."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exp_spec  # noqa: E402
    cache_f = WORK / "experiments" / "spec_cache.npz"
    if not cache_f.exists():
        print("  (spec_cache.npz missing -> specialist comparison skipped)")
        return None
    z = np.load(cache_f)
    pairs = [tuple(int(v) for v in p) for p in z["pairs"]]
    spec = {p: z[f"oof_{p[0]}_{p[1]}"] for p in pairs}
    thrs, confs = exp_spec.tune_thresholds(P, spec, pairs, y_tr, fold_tr,
                                           tune_folds=TUNE_FOLDS)
    C, n = exp_spec.apply_correction(P, spec, pairs, thrs, confs)
    print(f"  pair-specialist (5 pairs, tuned 0-2): overrides={n}")
    return C


# --------------------------------------------------------------------------- commands
def explore(base_path):
    D = load()
    tr = np.where(D["is_train"])[0]
    te = np.where(~D["is_train"])[0]
    y_tr, fold_tr = D["y"][tr], D["folds"][tr]
    P, P_te, h = load_base(base_path, D, tr, te)
    print(f"base {Path(base_path).name} hash={h}: +EI OOF acc={(P.argmax(1)==y_tr).mean():.4f}")
    c1, c2, p1, p2, _ = top2(P)
    elig = (y_tr == c1) | (y_tr == c2)
    print(f"cells with y in top-2: {elig.sum()}  (top1 correct {(y_tr==c1).sum()}, "
          f"top2 correct {(y_tr==c2).sum()}, neither {(~elig).sum()})")
    tune_m = np.isin(fold_tr, TUNE_FOLDS)
    pairs6 = confusion_pairs(c1, y_tr, tune_m, 6)
    print("top-6 confused pairs (folds 0-2):",
          [(a, b, D['labels'][a][:14], D['labels'][b][:14]) for a, b in pairs6])
    m6 = pair_mask(c1, c2, pairs6)
    print(f"cells whose top-2 is one of the 6 pairs: {m6.sum()} (y in top2: {(m6&elig).sum()})")

    results = {}
    print("\n=== specialist baseline (same base, tuned on 0-2) ===")
    C_spec = specialist_holdout(D, tr, P, y_tr, fold_tr)
    if C_spec is not None:
        results["specialist"] = evaluate(P, C_spec, y_tr, fold_tr, "pair-specialist")

    variants = [("meta", None), ("base", None), ("rel", None),
                ("base", m6), ("rel", m6)]
    for variant, extra in variants:
        vname = variant + ("" if extra is None else "_top6")
        print(f"\n=== variant {vname} ===", flush=True)
        t0 = time.time()
        pswap, pswap_te = train_reranker(D, tr, te, P, P_te, variant, extra, h)
        print(f"  trained in {time.time()-t0:.0f}s")
        for use_cap in (False, True):
            net, tau, cap = tune(P, pswap, y_tr, fold_tr, mask=extra, use_cap=use_cap)
            P_new, n = apply_swap(P, pswap, tau, cap, mask=extra)
            lab = f"{vname} tau={tau:.2f} cap={cap:.2f} (n={n})"
            results[lab] = evaluate(P, P_new, y_tr, fold_tr, lab)
        # fixed tau=0.5 for reference (no tuning at all)
        P_new, n = apply_swap(P, pswap, 0.5, 1.01, mask=extra)
        results[f"{vname} tau=0.5 fixed"] = evaluate(P, P_new, y_tr, fold_tr,
                                                     f"{vname} tau=0.50 fixed (n={n})")
        # robustness: reverse protocol (tune on 3-4, eval on 0-2)
        net_r, tau_r, cap_r = tune(P, pswap, y_tr, fold_tr, mask=extra,
                                   tune_folds=HOLD_FOLDS, use_cap=False)
        P_r, n = apply_swap(P, pswap, tau_r, cap_r, mask=extra)
        a0 = P.argmax(1) == y_tr; a1 = P_r.argmax(1) == y_tr
        print(f"  [reverse check] tuned on 3-4 tau={tau_r:.2f}: eval on 0-2 "
              f"{a0[tune_m].mean():.4f}->{a1[tune_m].mean():.4f} ({a1[tune_m].mean()-a0[tune_m].mean():+.4f})")
        if C_spec is not None and extra is None:
            # stacked: specialist first, then reranker (both tuned on 0-2)
            net, tau, cap = tune(P, pswap, y_tr, fold_tr, use_cap=False)
            P_new, n = apply_swap(C_spec, pswap, tau, cap)
            results[f"spec+{vname}"] = evaluate(P, P_new, y_tr, fold_tr,
                                                f"specialist then {vname} tau={tau:.2f} (n={n})")

    print("\n=== SUMMARY (holdout 3-4 delta, full-OOF delta) ===")
    for k, (hd, fd, per) in results.items():
        print(f"  {k:52s} holdout {hd:+.4f}  full {fd:+.4f}  per-fold {per}")


def run(base_path, variant="meta", tau=None, cap=None, top6=False, force=False,
        out=None, min_delta=0.001):
    D = load()
    tr = np.where(D["is_train"])[0]
    te = np.where(~D["is_train"])[0]
    y_tr, fold_tr = D["y"][tr], D["folds"][tr]
    P, P_te, h = load_base(base_path, D, tr, te)
    acc0 = (P.argmax(1) == y_tr).mean()
    print(f"base {Path(base_path).name} hash={h}: +EI OOF acc={acc0:.4f}")
    extra = extra_te = None
    if top6:
        c1, c2, _, _, _ = top2(P)
        pairs6 = confusion_pairs(c1, y_tr, np.isin(fold_tr, TUNE_FOLDS), 6)
        extra = pair_mask(c1, c2, pairs6)
        c1t, c2t, _, _, _ = top2(P_te)
        extra_te = pair_mask(c1t, c2t, pairs6)
        print("top-6 pairs (folds 0-2):", pairs6)
    pswap, pswap_te = train_reranker(D, tr, te, P, P_te, variant, extra, h)
    if tau is None:
        net, tau, cap_t = tune(P, pswap, y_tr, fold_tr, mask=extra, use_cap=(cap is None))
        if cap is None:
            cap = cap_t
        print(f"tuned on folds 0-2: tau={tau:.2f} cap={cap:.2f} (net {net:+d} cells on 0-2)")
    else:
        cap = 1.01 if cap is None else cap
        print(f"fixed tau={tau:.2f} cap={cap:.2f}")
    P_new, n = apply_swap(P, pswap, tau, cap, mask=extra)
    hd, fd, per = evaluate(P, P_new, y_tr, fold_tr, f"rerank[{variant}] tau={tau:.2f} cap={cap:.2f}")
    C_spec = specialist_holdout(D, tr, P, y_tr, fold_tr)
    if C_spec is not None:
        evaluate(P, C_spec, y_tr, fold_tr, "pair-specialist (ref, tuned 0-2)")
    P_te_new, n_te = apply_swap(P_te, pswap_te, tau, cap, mask=extra_te)
    out = Path(out) if out else WORK / "experiments" / "rerank_correction.npz"
    n_hold = int(np.isin(fold_tr, HOLD_FOLDS).sum())
    net_hold = int(round(hd * n_hold))            # gain in cells, avoids fp ties
    if net_hold > min_delta * n_hold or force:
        np.savez_compressed(
            out, oof=P_new.astype(np.float32), test=P_te_new.astype(np.float32),
            tau=float(tau), cap=float(cap), variant=variant, top6=bool(top6),
            base=str(base_path), base_hash=h,
            holdout_delta=float(hd), full_delta=float(fd),
            note=("post-EI top-2 reranker corrected probs (train rows D-order); "
                  f"tau/cap tuned on folds 0-2; holdout 3-4 delta {hd:+.4f}; "
                  f"full-OOF delta {fd:+.4f}; train swaps {n}; test swaps {n_te}"))
        print(f"saved {out}  train swaps={n} test swaps={n_te}  "
              f"HOLDOUT delta={hd:+.4f}  full delta={fd:+.4f}")
    else:
        print(f"NOT saved: holdout 3-4 delta {hd:+.4f} ({net_hold:+d} cells) <= +{min_delta} "
              f"(use --force to save anyway). train swaps={n} test swaps={n_te}")
    return hd, fd


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "explore"
    args = sys.argv[2:]
    if cmd == "explore":
        base = args[0] if args else str(WORK / "oof" / "bag8mix.npz")
        explore(base)
    elif cmd == "run":
        assert args, ("usage: run <base.npz> [--variant meta|base|rel|metapure] [--tau T] "
                      "[--cap C] [--top6] [--force] [--out path]")
        base = args[0]
        opt = {"--variant": "meta", "--tau": None, "--cap": None, "--out": None}
        for k in list(opt):
            if k in args:
                opt[k] = args[args.index(k) + 1]
        tau = None if opt["--tau"] is None else float(opt["--tau"])
        cap = None if opt["--cap"] is None else float(opt["--cap"])
        run(base, opt["--variant"], tau, cap, top6="--top6" in args,
            force="--force" in args, out=opt["--out"])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
