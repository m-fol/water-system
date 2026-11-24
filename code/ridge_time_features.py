import json, os
import numpy as np
from tune_temporal_networks import gen

V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def design(U, P, clock, inter):
    n, T, d = U.shape
    pad = np.concatenate([np.zeros((n, P - 1, d), np.float32), U], axis=1)
    X = np.stack([pad[:, t:t + P].reshape(n, -1) for t in range(T)], axis=1)
    cols = [X, np.ones((n, T, 1), np.float32)]
    if clock:
        ph = 2 * np.pi * np.arange(T) / 48
        c = np.stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)], 1).astype(np.float32)
        cols.append(np.broadcast_to(c, (n, T, 4)))
        if inter:
            cur = U[:, :, :2]
            cols.append((cur[:, :, :, None] * c[None, :, None, :]).reshape(n, T, -1))
    return np.concatenate(cols, axis=2).reshape(n * T, -1)


Utr, Xtr = gen(1000, 8000); Uv, Xv = gen(123, 500)
tests = {s: gen(s, 1000) for s in (777, 888, 999)}
out = {}
for name, clock, inter in (("no_clock", False, False), ("clock", True, False), ("clock_x_readings", True, True)):
    P = 64
    A = design(Utr, P, clock, inter).astype(np.float64); Y = Xtr.reshape(-1, 6).astype(np.float64)
    G = A.T @ A; B = A.T @ Y
    Av = design(Uv, P, clock, inter); Yv = Xv.reshape(-1, 6); best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B)
        v = float(((Av @ W - Yv) ** 2).mean())
        if best is None or v < best[0]: best = (v, lam, W)
    v, lam, W = best; res = []
    for s, (Ut, Xt) in tests.items():
        E = design(Ut, P, clock, inter) @ W - Xt.reshape(-1, 6)
        res.append((float((E ** 2).mean()), float(((E @ V) ** 2).mean())))
    out[name] = {"lambda": lam, "val_mse": v, "mse_mean": float(np.mean([r[0] for r in res])), "antisym_mean": float(np.mean([r[1] for r in res]))}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join("..", "results", "ridge_time_features.json"), "w"), indent=1)
