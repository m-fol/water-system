import json, os
import numpy as np, torch, torch.nn as nn
from tune_v8 import gen, build, STEPS, BATCH, N_TRAIN
def design(U, P):
    n, T, d = U.shape
    pad = np.concatenate([np.zeros((n, P - 1, d), np.float32), U], axis=1)
    X = np.stack([pad[:, t:t + P].reshape(n, -1) for t in range(T)], axis=1)
    X = np.concatenate([X, np.ones((n, T, 1), np.float32)], axis=2)
    return X.reshape(n * T, -1)
from data import generate_trajectory

torch.set_num_threads(8); torch.manual_seed(0)
Utr, Xtr = gen(1000, N_TRAIN)
net = build("mlp", 256); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); rng = np.random.default_rng(0)
for step in range(STEPS):
    idx = rng.integers(0, N_TRAIN, BATCH)
    loss = ((net(torch.tensor(Utr[idx])) - torch.tensor(Xtr[idx])) ** 2).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
P = 64
A = design(Utr, P).astype(np.float64); Y = Xtr.reshape(-1, 6).astype(np.float64)
W = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ Y)
acc = {}
for s in (777, 888, 999):
    r = np.random.default_rng(s); U, X, K = [], [], []
    for _ in range(1000):
        o, m, x, ev = generate_trajectory(T=64, rng=r, p_event=0.08)
        U.append(np.concatenate([o[:, :2], m[:, :2].astype(np.float32)], 1)); X.append(x)
        k = np.full(64, "calm", dtype=object)
        for (t0, t1, kind, node) in ev: k[t0:t1] = kind
        K.append(k)
    U = np.stack(U).astype(np.float32); X = np.stack(X); K = np.stack(K)
    with torch.no_grad(): Em = (net(torch.tensor(U)).numpy() - X) ** 2
    Er = (design(U, P) @ W).reshape(X.shape) - X; Er = Er ** 2
    for kind in ("calm", "dropout", "pump_trip", "demand_spike"):
        sel = K == kind
        a = acc.setdefault(kind, {"mlp": [], "ridge64": [], "n": 0})
        a["mlp"].append(float(Em[sel].mean())); a["ridge64"].append(float(Er[sel].mean())); a["n"] += int(sel.sum())
out = {k: {"mlp": float(np.mean(v["mlp"])), "ridge64": float(np.mean(v["ridge64"])), "steps": v["n"]} for k, v in acc.items()}
json.dump(out, open(os.path.join("..", "results", "regime_v8.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
