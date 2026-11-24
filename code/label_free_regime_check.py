import json, os
import numpy as np
import imm_label_free as E
import topology_experiments as T

K = 9
rng = np.random.default_rng(2); X, Y, tid, tt = [], [], [], []; lab = []
for i in range(2000):
    o, m, x, kinds, nodes, x0 = T.simulate(E.TOPO, rng=rng, dynamics="linear")
    for t in range(64): lab.append(T.regime_index(E.TOPO, kinds[t], nodes[t]))
X, Y, tid, tt, Rhat = E.collect()
lab = np.array(lab)
B0, *_ = np.linalg.lstsq(X, Y, rcond=None); R0 = Y - X @ B0; Z = R0 / R0.std(0); best = None
for r in range(3):
    z0 = E.kmeans(Z, K, np.random.default_rng(0 + r)); params, z, ll = E.hard_em(X, Y, z0, K)
    if best is None or ll > best[0]: best = (ll, z)
z = best[1]
conf = np.zeros((K, 7), int)
for g, l in zip(z, lab): conf[g, l] += 1
big = [g for g in range(K) if conf[g].sum() >= 100]
purity = float(sum(conf[g].max() for g in big) / sum(conf[g].sum() for g in big))
out = {"confusion_rows_groups_cols_true_regimes(calm,tripT1,tripT2,spikeJ1..J4)": conf.tolist(), "groups_with_>=100_transitions": big,
       "purity_over_large_groups": purity, "true_regime_counts": np.bincount(lab, minlength=7).tolist()}
json.dump(out, open(os.path.join("..", "results", "label_free_regime_check.json"), "w"), indent=1); print(json.dumps(out))
