import json, os, sys
import numpy as np
from multiprocessing import Pool
import imm_label_free as E
import topology_experiments as T

KS = [3, 5, 7, 8, 9, 10, 11, 12]
SEEDS = [0, 1, 2, 3, 4]
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def collect2(n_train, seed):
    rng = np.random.default_rng(seed); X, Y, tid, lab = [], [], [], []; res_obs = []
    for i in range(n_train):
        o, m, x, kinds, nodes, x0 = T.simulate(E.TOPO, rng=rng, dynamics="linear"); prev = x0
        for t in range(64):
            ang = 2 * np.pi * t / 48
            X.append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Y.append(x[t]); tid.append(i); lab.append(T.regime_index(E.TOPO, kinds[t], nodes[t])); prev = x[t]
            for k in E.TOPO.tanks:
                if m[t, k]: res_obs.append(o[t, k] - x[t, k])
    return np.array(X), np.array(Y), np.array(tid), np.array(lab), float(np.var(res_obs))


def ari(a, b):
    from math import comb
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b, return_inverse=True)[1]
    M = np.zeros((ca.max() + 1, cb.max() + 1), dtype=np.int64)
    for i, j in zip(ca, cb): M[i, j] += 1
    s_ij = sum(comb(int(x), 2) for x in M.ravel()); s_a = sum(comb(int(x), 2) for x in M.sum(1)); s_b = sum(comb(int(x), 2) for x in M.sum(0))
    n = int(M.sum()); expct = s_a * s_b / comb(n, 2); mx = 0.5 * (s_a + s_b)
    return float((s_ij - expct) / (mx - expct)) if mx != expct else 1.0


def job(args):
    K, seed = args
    X, Y, tid, lab, Rhat = collect2(2000, 100 + seed)
    B0, *_ = np.linalg.lstsq(X, Y, rcond=None); R0 = Y - X @ B0; Z = R0 / R0.std(0); best = None
    for r in range(3):
        z0 = E.kmeans(Z, K, np.random.default_rng(3 * seed + r)); params, z, ll = E.hard_em(X, Y, z0, K)
        if best is None or ll > best[0]: best = (ll, params, z)
    ll, params, z = best; N = len(X)
    sizes = np.bincount(z, minlength=K); large = [g for g in range(K) if sizes[g] >= 100]
    conf = np.zeros((K, 7), int)
    for g, l in zip(z, lab): conf[g, l] += 1
    mapped = {g: int(conf[g].argmax()) for g in large}
    train_purity = float(sum(conf[g].max() for g in large) / sum(sizes[g] for g in large))
    rng = np.random.default_rng(777); Xh, Yh, lh = [], [], []
    for _ in range(100):
        o, m, x, kinds, nodes, x0 = T.simulate(E.TOPO, rng=rng, dynamics="linear"); prev = x0
        for t in range(64):
            ang = 2 * np.pi * t / 48; Xh.append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Yh.append(x[t]); lh.append(T.regime_index(E.TOPO, kinds[t], nodes[t])); prev = x[t]
    Xh, Yh, lh = np.array(Xh), np.array(Yh), np.array(lh)
    pi = np.maximum(sizes / N, 1e-4); L = np.zeros((len(Xh), K))
    for k, (B, Q) in enumerate(params):
        r = Yh - Xh @ B; Qi = np.linalg.inv(Q)
        L[:, k] = -0.5 * (np.linalg.slogdet(Q)[1] + np.einsum("ni,ij,nj->n", r, Qi, r)) + np.log(pi[k])
    zh = L.argmax(1); in_large = np.isin(zh, large)
    heldout_purity = float(np.mean([mapped[g] == l for g, l in zip(zh[in_large], lh[in_large])])) if in_large.any() else float("nan")
    out = {"K": K, "seed": seed, "train_loglik": ll, "bic": float(-2 * ll + 75 * K * np.log(N)), "n_large_groups": len(large),
           "train_purity": train_purity, "heldout_purity": heldout_purity, "heldout_share_in_large_groups": float(in_large.mean()),
           "distinct_true_regimes_represented": len(set(mapped.values())), "heldout_ari": ari(zh, lh)}
    trans = np.zeros((K, K))
    for i in range(1, N):
        if tid[i] == tid[i - 1]: trans[z[i - 1], z[i]] += 1
    PI = (trans + 1e-3) / (trans + 1e-3).sum(1, keepdims=True); mu0 = sizes / sizes.sum(); n = E.TOPO.n; models = []
    for B, Q in params:
        c0, c1, c2 = B[n], B[n + 1], B[n + 2]
        models.append((B[:n].T, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)), Q))
    pk = T.prior_known(E.TOPO); se, an, cnt = 0.0, 0.0, 0
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        for _ in range(50):
            o, m, x, *_ = T.simulate(E.TOPO, rng=rng)
            Er = T.imm(o, m, models, PI, mu0 / mu0.sum(), pk, Rhat) - x
            se += float((Er ** 2).mean()); an += float(((Er @ V) ** 2).mean()); cnt += 1
    out["imm_mse"] = se / cnt; out["imm_antisym"] = an / cnt
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with Pool(nproc) as p: res = p.map(job, [(K, s) for s in SEEDS for K in KS], chunksize=1)
    json.dump(res, open(os.path.join("..", "results", "imm_label_free_validation.json"), "w"), indent=1)
