"""Frozen member list, MLP helpers and the gated blend used by train_final.py / predict_final.py."""
import json
import numpy as np
import torch, torch.nn as nn
from final_features import ART, META_NAMES, sp_labeled_count, postprocess

MEMBERS = {                      # 2 x 2 tiers: (spatial labels available?) x (Region/EI/Segment available?)
    "full":      ["lgb_ref", "lgb_all25", "mlp_s0", "mlp_s1", "mlp_s2"],      # sp yes, meta yes
    "nosp":      ["lgb_ref_nosp", "mlp_nosp_s0", "mlp_nosp_s1"],             # sp no,  meta yes
    "nometa_sp": ["lgb_ref_nometa_sp", "mlp_nometa_sp_s0"],                  # sp yes, meta no
    "nometa":    ["lgb_ref_nometa", "mlp_nometa_s0"],                        # sp no,  meta no
}
TIER_KW = {"full": {}, "nosp": {"drop_sp": True}, "nometa_sp": {"drop_meta": True},
           "nometa": {"drop_sp": True, "drop_meta": True}}
SP_MIN_LABELED = 3        # a test cell with fewer labelled spatial neighbours uses the nosp tier
META_MIN_FRAC = 0.02      # if fewer test cells than this have any Region/EI/Segment -> nometa tier


class Net(nn.Module):
    def __init__(self, d, h=512, p=0.3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(h // 2, 60))
    def forward(self, x): return self.f(x)


def mlp_prep_stats(Xfit, names):
    with np.errstate(all="ignore"):
        mu = np.nanmean(Xfit, 0); sd = np.nanstd(Xfit, 0) + 1e-6
    mu[~np.isfinite(mu)] = 0.0; sd[~np.isfinite(sd)] = 1.0      # all-NaN columns (nometa tier)
    return {"mu": mu, "sd": sd}


def mlp_transform(X, names, stats):
    Z = (X - stats["mu"]) / stats["sd"]
    meta_idx = [names.index(c) for c in META_NAMES if c in names]
    miss = np.isnan(X[:, meta_idx]).astype(np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([Z, miss]).astype(np.float32)


def tier_blend(probs, names):
    ps = [probs[n] for n in names if n in probs]
    mlps = [p for n, p in zip([n for n in names if n in probs], ps) if n.startswith("mlp")]
    others = [p for n, p in zip([n for n in names if n in probs], ps) if not n.startswith("mlp")]
    parts = others + ([np.mean(mlps, 0)] if mlps else [])
    return np.mean(parts, 0)


def blend_predict(probs, F, known, te, prep):
    """probs: {member: (n_te, 60)}; returns dict(prob, pred, tier, n_sp)."""
    n_sp = sp_labeled_count(F, known, 15)[te]
    seg = F["segment"][te]; ei = F["ei_known"][te]
    region = F["X"][te, F["names"].index("Region")]
    meta_frac = np.mean(~np.isnan(seg) | (ei >= 0) | ~np.isnan(region))
    sp_ok = n_sp >= SP_MIN_LABELED
    meta_ok = meta_frac >= META_MIN_FRAC
    tier = np.array(["full" if (s and meta_ok) else "nosp" if (not s and meta_ok) else "nometa_sp" if s else "nometa"
                     for s in sp_ok], dtype=object)
    out = np.zeros((len(te), 60))
    for t in set(tier):
        names = MEMBERS[t]
        if not all(n in probs for n in names):             # member set not trained -> nearest tier
            names = MEMBERS["nometa" if t == "nometa_sp" else "nosp"]
        out[tier == t] = tier_blend(probs, names)[tier == t]
    out = postprocess(out, ei, seg, prep)
    return {"prob": out, "pred": out.argmax(1), "tier": tier, "n_sp": n_sp, "meta_frac": float(meta_frac)}


def load_and_predict(F, known, te):
    """Load every saved member and predict the test rows. Returns {member: probs}."""
    import lightgbm as lgb
    from final_features import make_X
    probs = {}
    for tier, kw in TIER_KW.items():
        X, names = make_X(F, known, **kw)
        for name in MEMBERS[tier]:
            if not (ART / f"{name}.features.json").exists() and not (ART / f"{name}.meta.json").exists():
                continue                                       # tier not trained
            if name.startswith("lgb"):
                feats = json.load(open(ART / f"{name}.features.json"))
                assert feats == names, f"feature layout mismatch for {name}"
                mf = ART / f"{name}.txt"
                if mf.exists():
                    m = lgb.Booster(model_file=str(mf))
                else:                                      # packaged form: gzipped booster (maybe split)
                    import gzip, io
                    gz = ART / f"{name}.txt.gz"
                    raw = gz.read_bytes() if gz.exists() else b"".join(p.read_bytes() for p in sorted(ART.glob(f"{name}.txt.gz.part*")))
                    m = lgb.Booster(model_str=gzip.decompress(raw).decode())
                probs[name] = m.predict(X[te], num_threads=4)
            else:
                meta = json.load(open(ART / f"{name}.meta.json"))
                assert meta["names"] == names, f"feature layout mismatch for {name}"
                stats = {"mu": np.array(meta["mu"]), "sd": np.array(meta["sd"])}
                net = Net(meta["d_in"]); net.load_state_dict(torch.load(ART / f"{name}.pt", map_location="cpu")); net.eval()
                with torch.no_grad():
                    B = torch.tensor(mlp_transform(X[te], names, stats))
                    probs[name] = torch.softmax(net(B), 1).numpy()
    return probs
