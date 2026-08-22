"""Train the FROZEN final model set and save every artifact needed by predict_final.py.

Members (all trained with known labels = reference + competition train):
  full  : lgb_ref (reference only), lgb_all25 (25% reference + train), mlp_s0..2 (reference + train)
  nosp  : same feature set WITHOUT spatial neighbour-label histograms  -> used for test cells that
          have no labelled spatial neighbours (new sections)
  nometa: nosp AND Region/EI/Segment blanked -> used if the test metadata lacks those columns
5000 random deposit cells are held out of the reference as a labelled rehearsal set (scenario A).
Usage: python train_final.py [threads]
"""
import sys, json, time
import scipy.spatial, sklearn.decomposition  # import order: scipy/sklearn before lightgbm (segfault otherwise)
import numpy as np, pandas as pd, torch
import lightgbm as lgb
from final_features import (WORK, ART, build_universe, fit_prep, featurize, make_X, META_NAMES)
from final_blend import MEMBERS, mlp_prep_stats, mlp_transform, Net, blend_predict

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ART.mkdir(exist_ok=True)
genes = pd.read_csv(WORK.parent / "data/counts_train.csv", index_col=0, nrows=1).columns.tolist()
json.dump(genes, open(ART / "genes.json", "w"))

t0 = time.time()
U = build_universe(genes); U["genes"] = genes
rng = np.random.default_rng(2026)
cand = np.where(U["is_ref"])[0]
hold = rng.choice(cand, 5000, replace=False)
U["is_ref"][hold] = False
json.dump(U["ids"][hold].tolist(), open(ART / "holdout_A_ids.json", "w"))
json.dump(sorted(set(U["test_uids"].tolist())), open(ART / "trained_with_test_ids.json", "w"))   # test cells (universe ids) unlabelled at train time
print(f"universe={len(U['ids'])} ref={U['is_ref'].sum()} train={U['is_train'].sum()} test={U['is_test'].sum()} holdoutA=5000  ({time.time()-t0:.0f}s)", flush=True)

prep = fit_prep(U)
json.dump(prep, open(ART / "prep.json", "w"))
F = featurize(U, prep)
np.save(ART / "feature_names_static.npy", np.array(F["names"]))
known = np.where(U["is_ref"] | U["is_train"], U["y"], -1).astype(np.int64)
y = U["y"].astype(np.int64)
ref = np.where(U["is_ref"])[0]; tr = np.where(U["is_train"])[0]; te = np.where(U["is_test"])[0]
print(f"features built ({time.time()-t0:.0f}s)", flush=True)

LGB_REF = dict(objective="multiclass", num_class=60, learning_rate=0.08, num_leaves=63, min_data_in_leaf=30,
               feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127,
               verbosity=-1, num_threads=threads, seed=0, bagging_seed=100, feature_fraction_seed=200)
LGB_ALL = dict(LGB_REF, learning_rate=0.06, min_data_in_leaf=20, feature_fraction=0.7)
ROUNDS = {"lgb_ref": 700, "lgb_all25": 400}

def train_lgb(name, X, names, rows, params, rounds):
    t = time.time()
    m = lgb.train(params, lgb.Dataset(X[rows], y[rows], feature_name=names), rounds)
    m.save_model(str(ART / f"{name}.txt"))
    json.dump(names, open(ART / f"{name}.features.json", "w"))
    print(f"  {name}: rows={len(rows)} rounds={rounds} ({time.time()-t:.0f}s)", flush=True)
    return m.predict(X[te])

def train_mlp(name, X, names, rows, seed):
    t = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    stats = mlp_prep_stats(X[rows], names)
    A = torch.tensor(mlp_transform(X[rows], names, stats)); yy = torch.tensor(y[rows])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = Net(A.shape[1]).to(dev)
    epochs = 30; steps = epochs * ((len(A) + 511) // 512)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(A))
        for i in range(0, len(A), 512):
            idx = perm[i:i + 512]; xb = A[idx].to(dev); yb = yy[idx].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); sched.step()
    torch.save(net.state_dict(), ART / f"{name}.pt")
    json.dump({"names": names, "mu": stats["mu"].tolist(), "sd": stats["sd"].tolist(), "d_in": int(A.shape[1])},
              open(ART / f"{name}.meta.json", "w"))
    net.eval()
    with torch.no_grad():
        B = torch.tensor(mlp_transform(X[te], names, stats)).to(dev)
        out = torch.softmax(net(B), 1).cpu().numpy()
    print(f"  {name}: rows={len(rows)} ({time.time()-t:.0f}s)", flush=True)
    return out

test_probs = {}
rng25 = np.random.default_rng(0)
ref25 = ref[rng25.random(len(ref)) < 0.25]
for tier, kw in [("full", {}), ("nosp", {"drop_sp": True}), ("nometa", {"drop_sp": True, "drop_meta": True})]:
    X, names = make_X(F, known, **kw)
    print(f"== tier {tier}: X={X.shape}", flush=True)
    for name in MEMBERS[tier]:
        if name.startswith("lgb_ref"):
            test_probs[name] = train_lgb(name, X, names, ref, LGB_REF, ROUNDS["lgb_ref"])
        elif name.startswith("lgb_all25"):
            test_probs[name] = train_lgb(name, X, names, np.concatenate([ref25, tr]), LGB_ALL, ROUNDS["lgb_all25"])
        elif name.startswith("mlp"):
            seed = int(name.split("_s")[-1])
            test_probs[name] = train_mlp(name, X, names, np.concatenate([ref, tr]), seed)

np.savez_compressed(ART / "train_time_test_probs.npz", ids=U["ids"][te], **test_probs)
# final prediction on the CURRENT test set via the same frozen blend/gating code
ids_te = U["ids"][te]
res = blend_predict(test_probs, F, known, te, prep)
order = pd.Series(np.arange(len(ids_te)), index=ids_te).loc[U["test_uids"]].values
out = pd.DataFrame({"Cell_ID": U["mte"].index.values, "MERFISH_cell_type_annotation.y": np.array(prep["labels"])[res["pred"][order]]})
out.to_csv(ART / "prediction_train_time.csv", index=False)
print("tiers:", {k: int(v) for k, v in zip(*np.unique(res["tier"], return_counts=True))})
print(f"DONE in {time.time()-t0:.0f}s -> {ART}")
