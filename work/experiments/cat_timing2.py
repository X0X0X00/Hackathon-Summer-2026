import sys, time
sys.path.insert(0, "/Users/zzh/Visual Studio Code/Hackathon_26/work")
import numpy as np
from common import load, build_X
from catboost import CatBoostClassifier, Pool
D = load()
tr = np.where(D['is_train'])[0]; y=D['y']; folds=D['folds']
known = np.where(D['is_train'] & (folds!=0), y, -1).astype(np.int64)
X, names = build_X(D, known)
fit = tr[folds[tr]!=0]; va = tr[folds[tr]==0]
ptr = Pool(X[fit], y[fit].astype(np.int64)); pva = Pool(X[va], y[va].astype(np.int64))
def t(**kw):
    p = dict(loss_function="MultiClass", iterations=600, depth=6, learning_rate=0.08, l2_leaf_reg=3, random_strength=1,
             bootstrap_type="Bernoulli", subsample=0.8, rsm=0.3, border_count=64, thread_count=4, verbose=0,
             allow_writing_files=False, classes_count=60, random_seed=0, eval_metric='Accuracy', metric_period=50)
    p.update(kw)
    t0=time.time(); m=CatBoostClassifier(**p); m.fit(ptr, eval_set=pva, use_best_model=False)
    ev = m.get_evals_result()['validation']['Accuracy']
    curve = {i*50: round(ev[i*50],4) for i in range(1, len(ev)//50+1) if i*50 < len(ev)}
    curve[len(ev)-1]=round(ev[-1],4)
    print(kw, f"{time.time()-t0:.0f}s", curve, flush=True)
t(learning_rate=0.15, iterations=600)
t(learning_rate=0.15, iterations=600, grow_policy='Depthwise', depth=6)
t(learning_rate=0.15, iterations=600, grow_policy='Lossguide', max_leaves=63, depth=10, min_data_in_leaf=10)
