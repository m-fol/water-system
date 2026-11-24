import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
from sim_v9 import simulate
from tune_v8 import build as build_temporal
from models import _normalized_adjacency, GraphConv
from data import TANK_IDX

STEPS, BATCH, N_TRAIN = 4000, 32, 8000
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def gen(seed, n, random):
    rng = np.random.default_rng(seed); irng = np.random.default_rng(seed + 10_000)
    U, X = [], []
    for _ in range(n):
        o, m, x, k, ph, x0 = simulate(T=64, rng=rng, random=random, init_rng=irng)
        U.append(np.concatenate([o[:, :2], m[:, :2].astype(np.float32)], axis=1)); X.append(x)
    return np.stack(U).astype(np.float32), np.stack(X).astype(np.float32)


class GraphLSTM(nn.Module):
    def __init__(self, gh, lh):
        super().__init__()
        A = _normalized_adjacency()
        self.embed = nn.Linear(2, gh)
        self.g1 = GraphConv(gh, gh, A); self.g2 = GraphConv(gh, gh, A)
        self.proj = nn.Linear(6 * gh, 32)
        self.lstm = nn.LSTM(32, lh, batch_first=True)
        self.out = nn.Linear(lh, 6)

    def forward(self, u):
        B, T, _ = u.shape
        node = torch.zeros(B, T, 6, 2)
        node[:, :, TANK_IDX[0], 0] = u[:, :, 0]; node[:, :, TANK_IDX[1], 0] = u[:, :, 1]
        node[:, :, TANK_IDX[0], 1] = u[:, :, 2]; node[:, :, TANK_IDX[1], 1] = u[:, :, 3]
        h = torch.relu(self.embed(node.reshape(B * T, 6, 2)))
        h = torch.relu(self.g1(h)); h = torch.relu(self.g2(h))
        z = torch.relu(self.proj(h.reshape(B * T, -1))).reshape(B, T, -1)
        z, _ = self.lstm(z)
        return self.out(z)


def make(kind, hp):
    if kind == "graph":
        return GraphLSTM(*hp)
    return build_temporal(kind, hp)


def mse(net, U, X):
    with torch.no_grad():
        E = (net(torch.tensor(U)) - torch.tensor(X)).numpy()
    return float((E ** 2).mean()), float(((E @ V) ** 2).mean())


def run(job):
    kind, hp, lr, seed, random, final = job
    torch.set_num_threads(1); torch.manual_seed(seed)
    Utr, Xtr = gen(1000 + seed, N_TRAIN, random); Uv, Xv = gen(123, 500, random)
    net = make(kind, hp); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); rng = np.random.default_rng(seed)
    best, state, t0 = 1e9, None, time.time()
    for step in range(STEPS):
        idx = rng.integers(0, N_TRAIN, BATCH)
        loss = ((net(torch.tensor(Utr[idx])) - torch.tensor(Xtr[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            v, _ = mse(net, Uv, Xv)
            if v < best:
                best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    res = {"kind": kind, "hp": hp, "lr": lr, "seed": seed, "random": random, "val_mse": best, "seconds": time.time() - t0,
           "params": sum(p.numel() for p in net.parameters())}
    if final:
        res["test"] = []
        for s in (777, 888, 999):
            Ut, Xt = gen(s, 1000, random); m, a = mse(net, Ut, Xt)
            res["test"].append({"seed_eval": s, "mse": m, "mse_antisym": a})
    return res


def window_design(U, P=64):
    n, T, d = U.shape
    pad = np.concatenate([np.zeros((n, P - 1, d), np.float32), U], axis=1)
    X = np.stack([pad[:, t:t + P].reshape(n, -1) for t in range(T)], axis=1)
    return np.concatenate([X, np.ones((n, T, 1), np.float32)], axis=2).reshape(n * T, -1)


def ridge_random():
    out = {}
    for random in (False, True):
        Utr, Xtr = gen(1000, 8000, random); Uv, Xv = gen(123, 500, random)
        A = window_design(Utr).astype(np.float64); G = A.T @ A; B = A.T @ Xtr.reshape(-1, 6).astype(np.float64)
        Av = window_design(Uv); best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B)
            v = float(((Av @ W - Xv.reshape(-1, 6)) ** 2).mean())
            if best is None or v < best[0]:
                best = (v, lam, W)
        v, lam, W = best; r = []
        for s in (777, 888, 999):
            Ut, Xt = gen(s, 1000, random); E = window_design(Ut) @ W - Xt.reshape(-1, 6)
            r.append((float((E ** 2).mean()), float(((E @ V) ** 2).mean())))
        out["random" if random else "fixed"] = {"lambda": lam, "mse": float(np.mean([a for a, _ in r])), "antisym": float(np.mean([b for _, b in r]))}
        print("ridge", random, out["random" if random else "fixed"], flush=True)
    return out


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ridge = ridge_random()
    grid = [("graph", (16, 64), 3e-3), ("graph", (32, 64), 3e-3), ("graph", (32, 128), 3e-3), ("graph", (32, 128), 1e-3)]
    with Pool(nproc) as p:
        phase1 = p.map(run, [(k, hp, lr, 0, False, False) for k, hp, lr in grid])
        bestg = min(phase1, key=lambda r: r["val_mse"]); print("graph selected", bestg["hp"], bestg["lr"], bestg["val_mse"], flush=True)
        models = [("gru", 64, 3e-3), ("lstm", 128, 3e-3), ("mlp", 256, 3e-3), ("graph", tuple(bestg["hp"]), bestg["lr"])]
        jobs = [(k, hp, lr, s, rnd, True) for rnd in (False, True) for (k, hp, lr) in models for s in range(3)]
        phase2 = p.map(run, jobs)
    summ = {}
    for rnd in (False, True):
        for k, _, _ in models:
            rs = [r for r in phase2 if r["kind"] == k and r["random"] == rnd]
            ps = [float(np.mean([t["mse"] for t in r["test"]])) for r in rs]
            an = [float(np.mean([t["mse_antisym"] for t in r["test"]])) for r in rs]
            summ[f"{k}_{'random' if rnd else 'fixed'}"] = {"mse_mean": float(np.mean(ps)), "mse_sd": float(np.std(ps, ddof=1)), "antisym": float(np.mean(an)), "per_seed": ps}
    json.dump({"ridge": ridge, "phase1": phase1, "summary": summ, "phase2": phase2}, open(os.path.join(RES, "neural_v9.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
