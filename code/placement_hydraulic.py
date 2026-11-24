import itertools, json, os, sys
import numpy as np
from multiprocessing import Pool
import epanet_simulator as H
import topologies as TD
from placement_kalman import gram, Cmat, SV2

V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2); W6 = 6
RULES = ("random", "symmetry", "gram_E", "gram_trace", "bayes", "bayes_weak")


def make_config(i):
    r = np.random.default_rng(700 + i)
    het = None if i < 3 else r.uniform(0.8, 1.25, 6)
    return {"hetero": None if het is None else het.tolist(), "build_kw": {"hw": float(r.uniform(80, 140)), "pipe_len": float(r.uniform(400, 900))}}


def gen(cfg, seed, n):
    rng = np.random.default_rng(seed); O, M, X = [], [], []
    het = None if cfg["hetero"] is None else np.array(cfg["hetero"])
    for _ in range(n):
        o, m, x, _ = H.simulate(rng, hetero=het, build_kw=cfg["build_kw"]); O.append(o); M.append(m); X.append(x)
    return np.array(O), np.array(M), np.array(X)


def features(O, M, X, sensors, rng):
    n, T, _ = O.shape; chans = [O[:, :, 0], O[:, :, 1]] + [X[:, :, 2 + j] + rng.normal(0, H.MEAS_STD, (n, T)) for j in sensors]
    masks = [M[:, :, 0].astype(float), M[:, :, 1].astype(float)]
    base = np.stack(chans + masks, axis=2); F = []
    for lag in range(W6):
        sh = np.zeros_like(base); sh[:, lag:] = base[:, :T - lag]; F.append(sh)
    t = np.arange(T)[None, :, None] * np.ones((n, 1, 1)); ph = 2 * np.pi * t / 48
    return np.concatenate(F + [np.sin(ph), np.cos(ph)], axis=2).reshape(n * T, -1)


def ridge_eval(tr, te, sensors, seed):
    rng = np.random.default_rng(seed)
    Ftr = features(*tr, sensors, rng); Fte = features(*te, sensors, rng)
    Ytr = tr[2][:, :, 2:].reshape(-1, 4); Yte = te[2][:, :, 2:].reshape(-1, 4)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9; Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd; ym = Ytr.mean(0)
    Wt = np.linalg.solve(Ftr.T @ Ftr + 1.0 * np.eye(Ftr.shape[1]), Ftr.T @ (Ytr - ym)); E = Fte @ Wt + ym - Yte
    v4 = np.array([0, 1, 0, -1.0]) / np.sqrt(2)
    return float((E ** 2).mean()), float(((E @ v4) ** 2).mean())


def fit_linear(X):
    x0, x1 = X[:, :-1].reshape(-1, 6), X[:, 1:].reshape(-1, 6); t = np.tile(np.arange(63), len(X)); ph = 2 * np.pi * t / 48
    D = np.column_stack([x0, np.ones(len(x0)), np.sin(ph), np.cos(ph)]); Bm = np.linalg.lstsq(D, x1, rcond=None)[0]
    return Bm[:6].T


def criteria(A, P0, sets, tanks, U):
    out = {}
    for s in sets:
        Wg = gram(A, Cmat(6, list(tanks) + list(s))); Pp = np.linalg.inv(np.linalg.inv(P0) + Wg / SV2)
        out[s] = (np.linalg.eigvalsh(Wg)[0], np.trace(Wg), np.trace(Pp), np.trace(U.T @ Pp @ U))
    return out


def job(i):
    cfg = make_config(i); tr = gen(cfg, 5000 + i, 1000); te = gen(cfg, 9000 + i, 300); tanks = [0, 1]; Jn = [2, 3, 4, 5]
    n0, e0, w0 = None, None, None
    base = ridge_eval(tr, te, [], i)
    nom = TD.TOPOS["paper (T1-J1, T2-J3, loop J1-J2-J3-J4)"]; A_nom = TD.build(6, nom[1], nom[2], np.full(6, TD.K0)); P_nom = 0.25 * np.eye(6)
    A_fit = fit_linear(tr[2]); Xc = tr[2].reshape(-1, 6) - tr[2].reshape(-1, 6).mean(0); P_fit = np.cov(Xc.T) + 1e-6 * np.eye(6)
    res = {"config": i, "hetero": cfg["hetero"], "tank_only": base}
    for b in (1, 2):
        sets = list(itertools.combinations(Jn, b)); real = {s: ridge_eval(tr, te, [j - 2 for j in s], 100 * i + b) for s in sets}
        res["b%d" % b] = {"real": {",".join(map(str, s)): v for s, v in real.items()}}
        for mname, A, P0 in (("nominal", A_nom, P_nom), ("fitted", A_fit, P_fit)):
            w, Vv = np.linalg.eigh(gram(A, Cmat(6, tanks))); U = Vv[:, :1]
            cr = criteria(A, P0, sets, tanks, U); out = {}
            moved = [3, 5]; sym = [s for s in sets if all(j in moved for j in s)] or sets
            for rule in RULES:
                if rule == "random": chosen = sets
                elif rule == "symmetry": chosen = sym
                else:
                    idx, sign = {"gram_E": (0, -1), "gram_trace": (1, -1), "bayes": (2, 1), "bayes_weak": (3, 1)}[rule]
                    vals = {s: round(sign * cr[s][idx], 9) for s in sets}; best = min(vals.values()); chosen = [s for s in sets if vals[s] == best]
                out[rule] = {"total": float(np.mean([real[s][0] for s in chosen])), "weak": float(np.mean([real[s][1] for s in chosen])), "sets": [",".join(map(str, s)) for s in chosen]}
            res["b%d" % b][mname] = out
    print("config", i, "done", flush=True)
    return res


if __name__ == "__main__":
    n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    with Pool(min(15, n_cfg)) as p: runs = p.map(job, range(n_cfg), chunksize=1)
    json.dump({"runs": runs}, open(os.path.join("..", "results", "placement_hydraulic.json"), "w"), indent=1); print("done")
