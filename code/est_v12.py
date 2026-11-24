import json, os, sys
import numpy as np
from multiprocessing import Pool
import topo_exp_v11 as T

TOPO = T.Topo("paper")
N_TEST = 100


def collect(n_train=2000, seed=2):
    rng = np.random.default_rng(seed); X, Y, tid, tt = [], [], [], []; res_obs = []
    for i in range(n_train):
        o, m, x, kinds, nodes, x0 = T.simulate(TOPO, rng=rng, dynamics="linear"); prev = x0
        for t in range(64):
            ang = 2 * np.pi * t / 48
            X.append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Y.append(x[t]); tid.append(i); tt.append(t); prev = x[t]
            for k in TOPO.tanks:
                if m[t, k]: res_obs.append(o[t, k] - x[t, k])
    return np.array(X), np.array(Y), np.array(tid), np.array(tt), float(np.var(res_obs))


def kmeans(Z, K, rng, iters=30):
    C = [Z[rng.integers(len(Z))]]
    for _ in range(K - 1):
        d = np.min(((Z[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1); C.append(Z[rng.choice(len(Z), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        z = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            if (z == k).sum() > 0: C[k] = Z[z == k].mean(0)
    return z


def fit_group(X, Y):
    B, *_ = np.linalg.lstsq(X, Y, rcond=None); Q = np.cov((Y - X @ B).T) + 1e-6 * np.eye(Y.shape[1]); return B, Q


def hard_em(X, Y, z, K, iters=25):
    n = len(X); d = Y.shape[1]
    for _ in range(iters):
        params = []
        for k in range(K):
            idx = np.where(z == k)[0]
            if len(idx) < 60: idx = np.random.default_rng(k).choice(n, 200, replace=False)
            params.append(fit_group(X[idx], Y[idx]))
        pi = np.maximum(np.bincount(z, minlength=K) / n, 1e-4)
        L = np.zeros((n, K))
        for k, (B, Q) in enumerate(params):
            r = Y - X @ B; Qi = np.linalg.inv(Q)
            L[:, k] = -0.5 * (np.linalg.slogdet(Q)[1] + np.einsum("ni,ij,nj->n", r, Qi, r)) + np.log(pi[k])
        z_new = L.argmax(1)
        if (z_new != z).mean() < 1e-3: z = z_new; break
        z = z_new
    return params, z, float(L.max(1).sum())


def identify(K, restarts=3, seed=0):
    X, Y, tid, tt, Rhat = collect()
    B0, *_ = np.linalg.lstsq(X, Y, rcond=None); R0 = Y - X @ B0; Z = R0 / R0.std(0)
    best = None
    for r in range(restarts):
        z0 = kmeans(Z, K, np.random.default_rng(seed + r))
        params, z, ll = hard_em(X, Y, z0, K)
        if best is None or ll > best[0]: best = (ll, params, z)
    ll, params, z = best
    trans = np.zeros((K, K))
    for i in range(1, len(z)):
        if tid[i] == tid[i - 1]: trans[z[i - 1], z[i]] += 1
    PI = (trans + 1e-3) / (trans + 1e-3).sum(1, keepdims=True); mu0 = np.bincount(z, minlength=K) / len(z)
    n = TOPO.n; models = []
    for B, Q in params:
        c0, c1, c2 = B[n], B[n + 1], B[n + 2]
        models.append((B[:n].T, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)), Q))
    sizes = np.bincount(z, minlength=K).tolist()
    return models, PI, mu0 / mu0.sum(), Rhat, sizes, ll


def run(K):
    models, PI, mu0, Rhat, sizes, ll = identify(K)
    pk = T.prior_known(TOPO); V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2); se, anti, n = 0.0, 0.0, 0
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        for _ in range(N_TEST):
            o, m, x, *_ = T.simulate(TOPO, rng=rng)
            E = T.imm(o, m, models, PI, mu0, pk, Rhat) - x
            se += float((E ** 2).mean()); anti += float(((E @ V) ** 2).mean()); n += 1
    r = {"K": K, "mse": se / n, "antisym": anti / n, "n": n, "group_sizes": sizes, "train_loglik": ll}
    print(r, flush=True); return r


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(nproc) as p: res = p.map(run, [3, 5, 7, 9])
    json.dump(res, open(os.path.join("..", "results", "est_v12.json"), "w"), indent=1)
