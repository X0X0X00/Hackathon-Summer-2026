"""Train the (sp yes, meta no) tier members with the SAME universe/prep/known labels as train_final.py."""
import sys, json, time
import scipy.spatial, sklearn.decomposition
import numpy as np, pandas as pd, torch
import lightgbm as lgb
from final_features import WORK, ART, build_universe, featurize, make_X
from final_blend import mlp_prep_stats, mlp_transform, Net
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
genes = json.load(open(ART / "genes.json")); prep = json.load(open(ART / "prep.json")); hold = json.load(open(ART / "holdout_A_ids.json"))
U = build_universe(genes, exclude_ids=hold); U["genes"] = genes
print("reference rows:", int(U["is_ref"].sum()), flush=True)
F = featurize(U, prep)
known = np.where(U["is_ref"] | U["is_train"], U["y"], -1).astype(np.int64); y = U["y"].astype(np.int64)
ref = np.where(U["is_ref"])[0]; tr = np.where(U["is_train"])[0]
X, names = make_X(F, known, drop_meta=True)
P = dict(objective="multiclass", num_class=60, learning_rate=0.08, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbosity=-1, num_threads=threads, seed=0,
         bagging_seed=100, feature_fraction_seed=200)
t = time.time(); m = lgb.train(P, lgb.Dataset(X[ref], y[ref], feature_name=names), 700)
m.save_model(str(ART / "lgb_ref_nometa_sp.txt")); json.dump(names, open(ART / "lgb_ref_nometa_sp.features.json", "w"))
print(f"lgb_ref_nometa_sp done ({time.time()-t:.0f}s)", flush=True)
torch.manual_seed(0); np.random.seed(0); rows = np.concatenate([ref, tr])
stats = mlp_prep_stats(X[rows], names); A = torch.tensor(mlp_transform(X[rows], names, stats)); yy = torch.tensor(y[rows])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); net = Net(A.shape[1]).to(dev)
epochs = 30; opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(A) + 511) // 512)); lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
for ep in range(epochs):
    net.train(); perm = torch.randperm(len(A))
    for i in range(0, len(A), 512):
        idx = perm[i:i + 512]; opt.zero_grad(); loss = lossf(net(A[idx].to(dev)), yy[idx].to(dev)); loss.backward(); opt.step(); sched.step()
torch.save(net.state_dict(), ART / "mlp_nometa_sp_s0.pt")
json.dump({"names": names, "mu": stats["mu"].tolist(), "sd": stats["sd"].tolist(), "d_in": int(A.shape[1])}, open(ART / "mlp_nometa_sp_s0.meta.json", "w"))
print("mlp_nometa_sp_s0 done; EXTRA_TIER_DONE", flush=True)
