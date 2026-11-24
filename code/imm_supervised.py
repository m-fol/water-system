import json, os, sys
import numpy as np
from multiprocessing import Pool
import imm_true_regimes as E
from simulator import simulate
from data import MEASUREMENT_NOISE_STD
from network import STATE_DIM

V = E.V
N_PER_SET = 100


def imm(obs, mask, models, PI, mu0, prior, R):
    n = len(models)
    xs = [prior[0].copy() for _ in range(n)]; Ps = [prior[1].copy() for _ in range(n)]
    mu = mu0.copy(); out = np.zeros((obs.shape[0], STATE_DIM))
    E.R = R
    for t in range(obs.shape[0]):
        cbar = PI.T @ mu + 1e-300
        xm, Pm = [], []
        for j in range(n):
            w = PI[:, j] * mu / cbar[j]
            if w.sum() < 1e-9:
                xm.append(xs[j].copy()); Pm.append(Ps[j].copy()); continue
            xj = sum(w[i] * xs[i] for i in range(n))
            Pj = np.zeros((STATE_DIM, STATE_DIM))
            for i in range(n):
                if w[i] > 1e-12:
                    Pj += w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj))
            xm.append(xj); Pm.append(Pj)
        lik = np.zeros(n)
        for j in range(n):
            A, bf, Q = models[j]
            xp = A @ xm[j] + bf(t); Pp = A @ Pm[j] @ A.T + Q
            xs[j], Ps[j], ll = E.update(xp, Pp, obs[t], mask[t]); lik[j] = ll
        lik = np.nan_to_num(lik, nan=-1e9, neginf=-1e9)
        post = cbar * np.exp(lik - lik.max()) + 1e-300; mu = post / post.sum()
        out[t] = sum(mu[j] * xs[j] for j in range(n))
    return out


def true_models():
    return [(E.A_R[j], (lambda t, j=j: E.b_regime(t, 0.0, *E.REGIMES[j])), E.Q_R[j]) for j in range(7)]


def true_PI(onset_scale=1.0, exit_p=1.0 / 6.0):
    PI = np.zeros((7, 7))
    PI[0, 1:3] = 0.08 / 3 / 2 * onset_scale; PI[0, 3:] = 0.08 / 3 / 4 * onset_scale; PI[0, 0] = 1 - PI[0, 1:].sum()
    for r in range(1, 7):
        PI[r, 0] = exit_p; PI[r, r] = 1 - exit_p
    return PI


def exact_duration():
    dyn_index = [1, 2, 3, 4, 5, 6, 0, 0]
    onset = [0.08 / 3 / 2] * 2 + [0.08 / 3 / 4] * 4 + [0.08 / 3 / 2] * 2
    models = [true_models()[0]]
    for k in range(8):
        for a in range(1, 7):
            models.append(true_models()[dyn_index[k]])
    n = len(models); PI = np.zeros((n, n))
    idx = lambda k, a: 1 + 6 * k + (a - 1)
    start = [0] + [idx(k, 6) for k in range(8)]
    for s in start:
        for k in range(8):
            PI[s, idx(k, 1)] = onset[k]
        PI[s, 0] = 1 - sum(onset)
    for k in range(8):
        for a in range(1, 6):
            PI[idx(k, a), idx(k, a + 1)] = 1.0
    mu0 = np.zeros(n); mu0[0] = 1 - sum(onset)
    for k in range(8): mu0[idx(k, 1)] = onset[k]
    return models, PI, mu0 / mu0.sum()


def regime_index(kind, node):
    if kind == "pump_trip": return 1 + (0 if node == "T1" else 1)
    if kind == "demand_spike": return 3 + int(node[1]) - 1
    return 0


def identify(n_train=2000, seed=2):
    rng = np.random.default_rng(seed)
    Xr = [[] for _ in range(7)]; Yr = [[] for _ in range(7)]; trans = np.zeros((7, 7)); res_obs = []
    for _ in range(n_train):
        o, m, x, kinds, ph, x0, nodes = simulate(T=64, rng=rng, p_event=0.08, return_nodes=True)
        prev, prev_r = x0, None
        for t in range(64):
            r = regime_index(kinds[t], nodes[t]); ang = 2 * np.pi * t / 48
            Xr[r].append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Yr[r].append(x[t]); prev = x[t]
            if prev_r is not None: trans[prev_r, r] += 1
            prev_r = r
            for i in (0, 1):
                if m[t, i]: res_obs.append(o[t, i] - x[t, i])
    models = []
    for r in range(7):
        X = np.array(Xr[r]); Y = np.array(Yr[r])
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Q = np.cov((Y - X @ B).T) + 1e-6 * np.eye(STATE_DIM)
        A = B[:STATE_DIM].T; c0, c1, c2 = B[STATE_DIM], B[STATE_DIM + 1], B[STATE_DIM + 2]
        models.append((A, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)), Q))
    PI = (trans + 1e-3) / (trans + 1e-3).sum(axis=1, keepdims=True)
    mu0 = np.array([trans[:, r].sum() for r in range(7)]); mu0 = mu0 / mu0.sum()
    return models, PI, mu0, float(np.var(res_obs)), [len(x) for x in Xr]


CONFIGS = {}
_ID = None
_CACHE = {}


def setup(name):
    prior_known = E.start_prior("known", False)
    if name == "true_base":    return true_models(), true_PI(), E_mu(), prior_known, E.R
    if name == "onset_x0.5":   return true_models(), true_PI(onset_scale=0.5), E_mu(), prior_known, E.R
    if name == "onset_x2":     return true_models(), true_PI(onset_scale=2.0), E_mu(), prior_known, E.R
    if name == "exit_1_3":     return true_models(), true_PI(exit_p=1 / 3), E_mu(), prior_known, E.R
    if name == "exit_1_12":    return true_models(), true_PI(exit_p=1 / 12), E_mu(), prior_known, E.R
    if name == "exact_duration":
        m, PI, mu0 = exact_duration(); return m, PI, mu0, prior_known, E.R
    if name == "identified":
        global _ID
        if _ID is None: _ID = identify()
        m, PI, mu0, Rhat, _ = _ID; return m, PI, mu0, prior_known, Rhat


def E_mu():
    mu = np.array([1 - 0.0533] + [0.0133] * 2 + [0.00667] * 4); return mu / mu.sum()


def one(job):
    name, seed, i = job
    R0 = 0.03 ** 2; E.R = R0
    if seed not in _CACHE:
        rng = np.random.default_rng(seed); irng = np.random.default_rng(seed + 10_000)
        _CACHE[seed] = [simulate(T=64, rng=rng, random=False, init_rng=irng) for _ in range(N_PER_SET)]
    o, m, x, k, ph, x0 = _CACHE[seed][i]
    models, PI, mu0, prior, Rv = setup(name)
    est = imm(o, m, models, PI, mu0, prior, Rv)
    Err = est - x
    return name, float((Err ** 2).mean()), float(((Err @ V) ** 2).mean())


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    names = ["true_base", "onset_x0.5", "onset_x2", "exit_1_3", "exit_1_12", "exact_duration", "identified"]
    jobs = [(nm, s, i) for nm in names for s in (777, 888, 999) for i in range(N_PER_SET)]
    with Pool(nproc) as p:
        res = p.map(one, jobs, chunksize=4)
    out = {}
    for nm in names:
        r = [x for x in res if x[0] == nm]
        out[nm] = {"mse": float(np.mean([a for _, a, _ in r])), "antisym": float(np.mean([b for _, _, b in r])), "n": len(r)}
        print(nm, out[nm], flush=True)
    _, _, _, Rhat, counts = identify()
    out["identified_info"] = {"measurement_variance_hat": Rhat, "training_steps_per_regime": counts}
    json.dump(out, open(os.path.join("..", "results", "imm_supervised.json"), "w"), indent=1)
