import json, os
import numpy as np
from tune_temporal_networks import gen

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def design(U, P):
    n, T, d = U.shape
    pad = np.concatenate([np.zeros((n, P - 1, d), np.float32), U], axis=1)
    X = np.stack([pad[:, t:t + P].reshape(n, -1) for t in range(T)], axis=1)
    X = np.concatenate([X, np.ones((n, T, 1), np.float32)], axis=2)
    return X.reshape(n * T, -1)


Utr, Xtr = gen(1000, 8000)
Uv, Xv = gen(123, 500)
tests = {s: gen(s, 1000) for s in (777, 888, 999)}
out = {"n_train": 8000}
for P in (6, 16, 32, 64):
    A = design(Utr, P).astype(np.float64); Y = Xtr.reshape(-1, 6).astype(np.float64)
    G = A.T @ A; B = A.T @ Y
    Av = design(Uv, P); Yv = Xv.reshape(-1, 6)
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B)
        v = float(((Av @ W - Yv) ** 2).mean())
        if best is None or v < best[0]:
            best = (v, lam, W)
    v, lam, W = best
    res = []
    for s, (Ut, Xt) in tests.items():
        E = design(Ut, P) @ W - Xt.reshape(-1, 6)
        res.append({"seed_eval": s, "mse": float((E ** 2).mean()), "mse_antisym": float(((E @ V) ** 2).mean())})
    out[f"P{P}"] = {"lambda": lam, "val_mse": v, "mse_mean": float(np.mean([r["mse"] for r in res])),
                    "antisym_mean": float(np.mean([r["mse_antisym"] for r in res])), "test": res}
    print(P, lam, round(v, 4), round(out[f"P{P}"]["mse_mean"], 4), round(out[f"P{P}"]["antisym_mean"], 4), flush=True)
with open(os.path.join(RES, "ridge_equal_training.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
