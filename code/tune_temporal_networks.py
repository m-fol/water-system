import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool

from benchmark import generate_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
STEPS = 4000
BATCH = 32
N_TRAIN = 8000


def gen(seed, n):
    rng = np.random.default_rng(seed)
    U, X = [], []
    for _ in range(n):
        o, m, x, _ = generate_trajectory(T=64, rng=rng, p_event=0.08)
        U.append(np.concatenate([o[:, :2], m[:, :2].astype(np.float32)], axis=1)); X.append(x)
    return np.stack(U).astype(np.float32), np.stack(X).astype(np.float32)


class RNNModel(nn.Module):
    def __init__(self, kind, h):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.rnn = nn.LSTM(4, h, batch_first=True)
        elif kind == "gru":
            self.rnn = nn.GRU(4, h, batch_first=True)
        self.out = nn.Linear(h, 6)

    def forward(self, u):
        z, _ = self.rnn(u)
        return self.out(z)


class LinearRecurrent(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.A = nn.Parameter(0.9 * torch.eye(h) + 0.01 * torch.randn(h, h))
        self.B = nn.Linear(4, h)
        self.C = nn.Linear(h, 6)
        self.h = h

    def forward(self, u):
        h = torch.zeros(u.shape[0], self.h)
        outs = []
        for t in range(u.shape[1]):
            h = h @ self.A.T + self.B(u[:, t])
            outs.append(self.C(h))
        return torch.stack(outs, 1)


class WindowMLP(nn.Module):
    def __init__(self, h, win=16):
        super().__init__()
        self.win = win
        self.net = nn.Sequential(nn.Linear(4 * win, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 6))

    def forward(self, u):
        B, T, _ = u.shape
        pad = torch.cat([torch.zeros(B, self.win - 1, 4), u], dim=1)
        win = torch.stack([pad[:, t:t + self.win].reshape(B, -1) for t in range(T)], 1)
        return self.net(win)


def build(kind, h):
    if kind in ("lstm", "gru"):
        return RNNModel(kind, h)
    if kind == "linrec":
        return LinearRecurrent(h)
    if kind == "mlp":
        return WindowMLP(h)


def mse(net, U, X):
    with torch.no_grad():
        E = (net(torch.tensor(U)) - torch.tensor(X)).numpy()
    return float((E ** 2).mean()), float(((E @ V) ** 2).mean())


def run(job):
    kind, h, lr, seed, final = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    Utr, Xtr = gen(1000 + seed, N_TRAIN)
    Uv, Xv = gen(123, 500)
    net = build(kind, h)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(seed)
    best, best_state = 1e9, None
    t0 = time.time()
    for step in range(STEPS):
        idx = rng.integers(0, N_TRAIN, BATCH)
        u = torch.tensor(Utr[idx]); x = torch.tensor(Xtr[idx])
        loss = ((net(u) - x) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            v, _ = mse(net, Uv, Xv)
            if v < best:
                best = v; best_state = {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    res = {"kind": kind, "hidden": h, "lr": lr, "seed": seed, "val_mse": best, "seconds": time.time() - t0,
           "params": sum(p.numel() for p in net.parameters())}
    if final:
        tests = []
        for s in (777, 888, 999):
            Ut, Xt = gen(s, 1000)
            m_, a_ = mse(net, Ut, Xt)
            tests.append({"seed_eval": s, "mse": m_, "mse_antisym": a_})
        res["test"] = tests
    return res


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    grid = [("lstm", h, lr) for h in (32, 128) for lr in (1e-3, 3e-3)] + \
           [("gru", 64, 3e-3), ("gru", 128, 1e-3)] + \
           [("linrec", h, lr) for h in (16, 64) for lr in (3e-3, 1e-2)] + \
           [("mlp", h, lr) for h in (64, 256) for lr in (1e-3, 3e-3)]
    with Pool(nproc) as p:
        phase1 = p.map(run, [(k, h, lr, 0, False) for k, h, lr in grid])
        print("phase 1 done", flush=True)
        best = {}
        for r in phase1:
            if r["kind"] not in best or r["val_mse"] < best[r["kind"]]["val_mse"]:
                best[r["kind"]] = r
        jobs = [(r["kind"], r["hidden"], r["lr"], s, True) for r in best.values() for s in range(5)]
        phase2 = p.map(run, jobs)
    out = {"phase1": phase1, "selected": {k: {"hidden": v["hidden"], "lr": v["lr"], "val_mse": v["val_mse"]} for k, v in best.items()},
           "phase2": phase2, "steps": STEPS, "batch": BATCH, "n_train": N_TRAIN}
    summ = {}
    for k in best:
        rs = [r for r in phase2 if r["kind"] == k]
        per_seed = [float(np.mean([t["mse"] for t in r["test"]])) for r in rs]
        anti = [float(np.mean([t["mse_antisym"] for t in r["test"]])) for r in rs]
        summ[k] = {"mse_mean": float(np.mean(per_seed)), "mse_sd": float(np.std(per_seed, ddof=1)),
                   "mse_best_seed": float(min(per_seed)), "antisym_mean": float(np.mean(anti)), "per_seed": per_seed}
    out["summary"] = summ
    with open(os.path.join(RES, "tuned_temporal_networks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summ, indent=1))
