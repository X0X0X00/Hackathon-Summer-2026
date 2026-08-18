"""MLP member on the extended universe (reference + competition-train), torch/MPS.
Usage: python ext_mlp.py [name] [seed] [ref_frac]
Features: build_X_ext (static + dense neighbour-label hists), standardised on the fit rows,
NaN -> 0 after standardisation plus a missing-indicator for the metadata cols.
"""
import sys, time
import numpy as np
import torch, torch.nn as nn
from common_ext import load_ext, run_cv_ext, save_oof_ext

name = sys.argv[1] if len(sys.argv) > 1 else "ext_mlp"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ref_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(seed); np.random.seed(seed)


def prep(Xtr, Xva):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6
    def t(X):
        Z = (X - mu) / sd
        miss = np.isnan(X[:, -19:]).astype(np.float32)   # meta block indicators
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.hstack([Z, miss]).astype(np.float32)
    return t(Xtr), t(Xva)


class Net(nn.Module):
    def __init__(self, d, h=512, p=0.3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(h // 2, 60))
    def forward(self, x): return self.f(x)


def fit_predict(Xtr, ytr, Xva, names):
    t0 = time.time()
    A, B = prep(Xtr, Xva)
    A = torch.tensor(A); y = torch.tensor(ytr.astype(np.int64)); B = torch.tensor(B).to(dev)
    net = Net(A.shape[1]).to(dev)
    epochs = 30 if len(A) > 20000 else 120
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(A) + 511) // 512))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(A)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            xb = A[idx].to(dev); yb = y[idx].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(B), 4096):
            out.append(torch.softmax(net(B[i:i + 4096]), 1).cpu().numpy())
    print(f"    mlp fit {Xtr.shape} epochs={epochs} in {time.time()-t0:.0f}s", flush=True)
    return np.vstack(out)


D = load_ext()
res = run_cv_ext(D, fit_predict, mode="all", ref_frac=ref_frac, seed=seed)
save_oof_ext(name, res)
