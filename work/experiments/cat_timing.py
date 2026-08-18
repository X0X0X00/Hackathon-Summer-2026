import sys, time
sys.path.insert(0, "/Users/zzh/Visual Studio Code/Hackathon_26/work")
import numpy as np
from common import load, build_X
from catboost import CatBoostClassifier
D = load()
tr = np.where(D['is_train'])[0]; y=D['y']; folds=D['folds']
known = np.where(D['is_train'] & (folds!=0), y, -1).astype(np.int64)
X, names = build_X(D, known)
fit = tr[folds[tr]!=0]; va = tr[folds[tr]==0]
def t(**kw):
    p = dict(loss_function="MultiClass", iterations=100, depth=6, learning_rate=0.08, l2_leaf_reg=3, random_strength=1,
             bootstrap_type="Bernoulli", subsample=0.8, rsm=0.6, thread_count=4, verbose=0, allow_writing_files=False, classes_count=60, random_seed=0)
    p.update(kw)
    t0=time.time(); m=CatBoostClassifier(**p); m.fit(X[fit], y[fit].astype(np.int64))
    pr=m.predict_proba(X[va]); acc=(pr.argmax(1)==y[va]).mean()
    print(kw, f"{time.time()-t0:.1f}s acc100={acc:.4f}", "boosting_type=", m.get_all_params().get('boosting_type'), "border_count=", m.get_all_params().get('border_count'), flush=True)
t()
t(boosting_type='Plain')
t(boosting_type='Plain', border_count=64)
t(boosting_type='Plain', border_count=64, rsm=0.3)
t(boosting_type='Plain', border_count=64, depth=8)
