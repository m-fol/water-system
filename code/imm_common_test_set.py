import json, os
import numpy as np
from multiprocessing import Pool
import imm_label_free as E12
import imm_supervised as E10
import topology_experiments as T
import imm_label_free_validation as E13

V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def label_free_models(K, seed):
    X, Y, tid, lab, Rhat = E13.collect2(2000, 100 + seed)
    B0, *_ = np.linalg.lstsq(X, Y, rcond=None); R0 = Y - X @ B0; Z = R0 / R0.std(0); best = None
    for r in range(3):
        z0 = E12.kmeans(Z, K, np.random.default_rng(3 * seed + r)); params, z, ll = E12.hard_em(X, Y, z0, K)
        if best is None or ll > best[0]: best = (ll, params, z)
    ll, params, z = best; N = len(X); sizes = np.bincount(z, minlength=K); trans = np.zeros((K, K))
    for i in range(1, N):
        if tid[i] == tid[i - 1]: trans[z[i - 1], z[i]] += 1
    PI = (trans + 1e-3) / (trans + 1e-3).sum(1, keepdims=True); n = E12.TOPO.n; models = []
    for B, Q in params:
        c0, c1, c2 = B[n], B[n + 1], B[n + 2]
        models.append((B[:n].T, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)), Q))
    return models, PI, sizes / sizes.sum(), Rhat


def run(spec):
    name = spec[0]
    if name == "supervised":
        models, PI, mu0, Rhat, _ = E10.identify()
    else:
        models, PI, mu0, Rhat = label_free_models(spec[1], spec[2])
    pk = T.prior_known(E12.TOPO); se, an, n = 0.0, 0.0, 0
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        for _ in range(300):
            o, m, x, *_ = T.simulate(E12.TOPO, rng=rng)
            Er = T.imm(o, m, models, PI, mu0 / mu0.sum(), pk, Rhat) - x
            se += float((Er ** 2).mean()); an += float(((Er @ V) ** 2).mean()); n += 1
    r = {"name": name if name == "supervised" else "label_free_seed%d_K%d" % (spec[2], spec[1]), "mse": se / n, "antisym": an / n, "n": n}
    print(r, flush=True); return r


if __name__ == "__main__":
    d = json.load(open(os.path.join("..", "results", "imm_label_free_validation.json")))
    specs = [("supervised",)]
    for s in sorted(set(r["seed"] for r in d)):
        b = min([r for r in d if r["seed"] == s], key=lambda r: r["bic"]); specs.append(("lf", b["K"], s))
    with Pool(len(specs)) as p: res = p.map(run, specs)
    json.dump(res, open(os.path.join("..", "results", "imm_common_test_set.json"), "w"), indent=1)
