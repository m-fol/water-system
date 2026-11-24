import json, os, sys, time
import numpy as np
from benchmark import FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, generate_trajectory

DT, SIGMA_D = 0.5, 0.05
TOPOS = {
    "paper": (6, [0, 1], [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]),
    "ring_opposite": (8, [0, 1], [(0, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]),
    "ring_adjacent": (8, [0, 1], [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]),
}
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


class Topo:
    def __init__(self, name):
        n, tanks, edges = TOPOS[name]
        self.name, self.n, self.tanks, self.edges = name, n, tanks, edges
        self.junctions = [i for i in range(n) if i not in tanks]
        self.adj = np.zeros((n, n))
        for a, b in edges:
            self.adj[a, b] = self.adj[b, a] = 1.0
        self.L = np.diag(self.adj.sum(1)) - self.adj
        self.mask = np.array([i in tanks for i in range(n)])

    def unobservable(self):
        A = self.A_regime(None)
        C = np.zeros((len(self.tanks), self.n)); C[np.arange(len(self.tanks)), self.tanks] = 1.0
        O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(self.n)])
        _, sv, Vt = np.linalg.svd(O)
        return Vt[sv < 1e-9 * sv[0]] if (sv < 1e-9 * sv[0]).any() else np.zeros((0, self.n))

    def A_regime(self, trip):
        damp = np.zeros((self.n, self.n))
        for k, i in enumerate(self.tanks):
            if trip != k: damp[i, i] = PUMP_GAIN
        return np.eye(self.n) + DT * (-FLOW_GAIN * self.L - damp)

    def b_regime(self, t, trip=None, spike=None):
        b = np.zeros(self.n)
        for k, i in enumerate(self.tanks):
            if trip != k: b[i] = DT * PUMP_GAIN * TANK_SETPOINT
        for k, i in enumerate(self.junctions):
            b[i] = -DT * BASE_DEMAND * (1 + 0.5 * np.sin(2 * np.pi * t / 48)) * (4.0 if spike == k else 1.0)
        return b

    def regimes(self):
        return [(None, None)] + [(k, None) for k in range(len(self.tanks))] + [(None, j) for j in range(len(self.junctions))]


def simulate(topo, T=64, rng=None, p_event=0.08, event_len=6, dynamics="linear"):
    rng = rng or np.random.default_rng(); n = topo.n
    tnames = ["T%d" % (i + 1) for i in range(len(topo.tanks))]; jnames = ["J%d" % (i + 1) for i in range(len(topo.junctions))]
    x = np.zeros(n, dtype=np.float32); x[topo.tanks] = TANK_SETPOINT; x[topo.junctions] = TANK_SETPOINT - 1.0; x0 = x.copy()
    state = np.zeros((T, n), np.float32); obs = np.zeros((T, n), np.float32); mask = np.tile(topo.mask, (T, 1)).copy()
    kinds = np.full(T, "calm", dtype=object); nodes = np.full(T, "", dtype=object); active = None
    for t in range(T):
        if active is not None and t >= active[0]: active = None
        if active is None and rng.random() < p_event:
            kind = rng.choice(["dropout", "pump_trip", "demand_spike"])
            node = rng.choice(tnames) if kind in ("dropout", "pump_trip") else rng.choice(jnames)
            active = (t + event_len, kind, node)
        if active is not None: kinds[t], nodes[t] = active[1], active[2]
        dh = x[None, :] - x[:, None]
        edge = np.sign(dh) * np.sqrt(np.abs(dh) + 1e-6) if dynamics == "nonlinear" else dh
        flow = FLOW_GAIN * (topo.adj * edge).sum(axis=1)
        pump = np.zeros(n, np.float32)
        for idx, name in zip(topo.tanks, tnames):
            g = 0.0 if (active is not None and active[1] == "pump_trip" and active[2] == name) else PUMP_GAIN
            pump[idx] = g * (TANK_SETPOINT - x[idx])
        demand = np.zeros(n, np.float32)
        for idx, name in zip(topo.junctions, jnames):
            d = BASE_DEMAND * (1.0 + 0.5 * np.sin(2 * np.pi * t / 48)) + rng.normal(0, 0.05)
            if active is not None and active[1] == "demand_spike" and active[2] == name: d *= 4.0
            demand[idx] = max(d, 0.0)
        x = x + DT * (flow + pump - demand); state[t] = x
        o = np.zeros(n, np.float32); m = topo.mask.copy()
        for idx, name in zip(topo.tanks, tnames):
            if active is not None and active[1] == "dropout" and active[2] == name: m[idx] = False
            else: o[idx] = x[idx] + rng.normal(0, MEASUREMENT_NOISE_STD)
        obs[t], mask[t] = o, m
    return obs, mask, state, kinds, nodes, x0


def regime_index(topo, kind, node):
    if kind == "pump_trip": return 1 + int(node[1]) - 1
    if kind == "demand_spike": return 1 + len(topo.tanks) + int(node[1]) - 1
    return 0


R_MEAS = MEASUREMENT_NOISE_STD ** 2


def update(x, P, z, mask, R):
    av = np.where(mask)[0]
    if len(av) == 0: return x, P, 0.0
    n = len(x); H = np.zeros((len(av), n)); H[np.arange(len(av)), av] = 1.0
    P = 0.5 * (P + P.T); S = H @ P @ H.T + R * np.eye(len(av)); K = P @ H.T @ np.linalg.inv(S)
    inn = z[av] - H @ x; ll = -0.5 * (inn @ np.linalg.solve(S, inn) + np.linalg.slogdet(S)[1])
    Pn = (np.eye(n) - K @ H) @ P
    return x + K @ inn, 0.5 * (Pn + Pn.T), ll


def imm(obs, mask, models, PI, mu0, prior, R):
    n = len(models); d = len(prior[0])
    xs = [prior[0].copy() for _ in range(n)]; Ps = [prior[1].copy() for _ in range(n)]
    mu = mu0.copy(); out = np.zeros((obs.shape[0], d))
    for t in range(obs.shape[0]):
        cbar = PI.T @ mu + 1e-300; xm, Pm = [], []
        for j in range(n):
            w = PI[:, j] * mu / cbar[j]
            if w.sum() < 1e-9: xm.append(xs[j].copy()); Pm.append(Ps[j].copy()); continue
            xj = sum(w[i] * xs[i] for i in range(n)); Pj = np.zeros((d, d))
            for i in range(n):
                if w[i] > 1e-12: Pj += w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj))
            xm.append(xj); Pm.append(Pj)
        lik = np.zeros(n)
        for j in range(n):
            A, bf, Q = models[j]
            xs[j], Ps[j], lik[j] = update(A @ xm[j] + bf(t), A @ Pm[j] @ A.T + Q, obs[t], mask[t], R)
        lik = np.nan_to_num(lik, nan=-1e9, neginf=-1e9)
        post = cbar * np.exp(lik - lik.max()) + 1e-300; mu = post / post.sum()
        out[t] = sum(mu[j] * xs[j] for j in range(n))
    return out


def true_models(topo):
    d = topo.n; models = []
    for (trip, spike) in topo.regimes():
        q = np.full(d, 1e-6)
        for k, i in enumerate(topo.junctions): q[i] = (DT * SIGMA_D * (4.0 if spike == k else 1.0)) ** 2 + 1e-6
        models.append((topo.A_regime(trip), (lambda t, trip=trip, spike=spike: topo.b_regime(t, trip, spike)), np.diag(q)))
    return models


def true_PI(topo, exit_p=1 / 6):
    nt, nj = len(topo.tanks), len(topo.junctions); R = 1 + nt + nj; PI = np.zeros((R, R))
    PI[0, 1:1 + nt] = 0.08 / 3 / nt; PI[0, 1 + nt:] = 0.08 / 3 / nj; PI[0, 0] = 1 - PI[0, 1:].sum()
    for r in range(1, R): PI[r, 0] = exit_p; PI[r, r] = 1 - exit_p
    mu = PI[0].copy(); return PI, mu / mu.sum()


def prior_known(topo):
    m = np.zeros(topo.n); m[topo.tanks] = TANK_SETPOINT; m[topo.junctions] = TANK_SETPOINT - 1.0
    return m, np.eye(topo.n) * 1e-4


def kf_models(topo, Q):
    return [(topo.A_regime(None), (lambda t: topo.b_regime(t)), Q)]


def calibrate_Q(topo, dynamics, n=200, seed=4242):
    rng = np.random.default_rng(seed); res = []
    A = topo.A_regime(None)
    for _ in range(n):
        o, m, x, k, nd, x0 = simulate(topo, rng=rng, dynamics=dynamics); prev = x0
        for t in range(64): res.append(x[t] - (A @ prev + topo.b_regime(t))); prev = x[t]
    return np.cov(np.array(res).T)


def identify(topo, dynamics, n_train=2000, seed=2):
    rng = np.random.default_rng(seed); R = len(topo.regimes()); Xr = [[] for _ in range(R)]; Yr = [[] for _ in range(R)]
    trans = np.zeros((R, R)); res_obs = []
    for _ in range(n_train):
        o, m, x, kinds, nodes, x0 = simulate(topo, rng=rng, dynamics=dynamics); prev, pr = x0, None
        for t in range(64):
            r = regime_index(topo, kinds[t], nodes[t]); ang = 2 * np.pi * t / 48
            Xr[r].append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Yr[r].append(x[t]); prev = x[t]
            if pr is not None: trans[pr, r] += 1
            pr = r
            for i in topo.tanks:
                if m[t, i]: res_obs.append(o[t, i] - x[t, i])
    models = []
    for r in range(R):
        X, Y = np.array(Xr[r]), np.array(Yr[r]); B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Q = np.cov((Y - X @ B).T) + 1e-6 * np.eye(topo.n); c0, c1, c2 = B[topo.n], B[topo.n + 1], B[topo.n + 2]
        models.append((B[:topo.n].T, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)), Q))
    PI = (trans + 1e-3) / (trans + 1e-3).sum(axis=1, keepdims=True)
    mu0 = trans.sum(axis=0); return models, PI, mu0 / mu0.sum(), float(np.var(res_obs))


def test_sets(topo, dynamics, n):
    out = []
    for s in (777, 888, 999):
        rng = np.random.default_rng(s); out.append([simulate(topo, rng=rng, dynamics=dynamics) for _ in range(n)])
    return out


def classical(topo_name, dynamics):
    topo = Topo(topo_name); N = topo.unobservable()
    rng = np.random.default_rng(1); train = [simulate(topo, rng=rng, dynamics=dynamics) for _ in range(300)]
    tm = np.stack([x for _, _, x, *_ in train]).mean(0)
    Q = calibrate_Q(topo, dynamics); PI, mu0 = true_PI(topo); pk = prior_known(topo)
    ident = identify(topo, dynamics)
    acc = {}
    for si, trajs in enumerate(test_sets(topo, dynamics, 100)):
        for o, m, x, k, nd, x0 in trajs:
            ests = {"time_only": np.broadcast_to(tm, x.shape),
                    "kf_known_start": imm(o, m, kf_models(topo, Q), np.ones((1, 1)), np.ones(1), pk, R_MEAS),
                    "imm_true_known_start": imm(o, m, true_models(topo), PI, mu0, pk, R_MEAS),
                    "imm_supervised_identified": imm(o, m, ident[0], ident[1], ident[2], pk, ident[3])}
            for name, e in ests.items():
                E = e - x; d = acc.setdefault(name, {"se": 0.0, "null": 0.0, "n": 0})
                d["se"] += float((E ** 2).mean()); d["null"] += float(((E @ N.T) ** 2).mean()) if len(N) else float("nan"); d["n"] += 1
    out = {k: {"mse": v["se"] / v["n"], "mse_null_dirs": v["null"] / v["n"]} for k, v in acc.items()}
    out["n_unobservable_dirs"] = int(len(N)); out["n_test_trajectories"] = 300
    json.dump(out, open(os.path.join(RES, f"topo_classical_{topo_name}_{dynamics}.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def neural_job(job):
    import torch, torch.nn as nn
    kind, hp, lr, seed, topo_name, dynamics = job
    torch.set_num_threads(1); torch.manual_seed(seed)
    topo = Topo(topo_name); n = topo.n

    def gen(s, cnt):
        rng = np.random.default_rng(s); U, X = [], []
        for _ in range(cnt):
            o, m, x, *_ = simulate(topo, rng=rng, dynamics=dynamics)
            U.append(np.concatenate([o[:, topo.tanks], m[:, topo.tanks].astype(np.float32)], axis=1)); X.append(x)
        return np.stack(U).astype(np.float32), np.stack(X).astype(np.float32)

    class GRU(nn.Module):
        def __init__(self, h):
            super().__init__(); self.rnn = nn.GRU(4, h, batch_first=True); self.out = nn.Linear(h, n)
        def forward(self, u): return self.out(self.rnn(u)[0])

    class GraphLSTM(nn.Module):
        def __init__(self, gh, lh):
            super().__init__()
            A = topo.adj + np.eye(n); d = A.sum(1) ** -0.5; An = torch.tensor(d[:, None] * A * d[None, :], dtype=torch.float32)
            self.register_buffer("An", An); self.embed = nn.Linear(2, gh); self.g1 = nn.Linear(gh, gh); self.g2 = nn.Linear(gh, gh)
            self.proj = nn.Linear(n * gh, 32); self.lstm = nn.LSTM(32, lh, batch_first=True); self.out = nn.Linear(lh, n)
        def forward(self, u):
            B, T, _ = u.shape; node = torch.zeros(B, T, n, 2)
            for k, i in enumerate(topo.tanks): node[:, :, i, 0] = u[:, :, k]; node[:, :, i, 1] = u[:, :, len(topo.tanks) + k]
            h = torch.relu(self.embed(node.reshape(B * T, n, 2)))
            h = torch.relu(self.g1(torch.einsum("ij,bjf->bif", self.An, h))); h = torch.relu(self.g2(torch.einsum("ij,bjf->bif", self.An, h)))
            z = torch.relu(self.proj(h.reshape(B * T, -1))).reshape(B, T, -1)
            return self.out(self.lstm(z)[0])

    Utr, Xtr = gen(1000 + seed, 8000); Uv, Xv = gen(123, 500)
    net = GRU(hp) if kind == "gru" else GraphLSTM(*hp)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000); rng = np.random.default_rng(seed)
    best, state = 1e9, None
    for step in range(4000):
        idx = rng.integers(0, 8000, 32)
        loss = ((net(torch.tensor(Utr[idx])) - torch.tensor(Xtr[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            with torch.no_grad(): v = float(((net(torch.tensor(Uv)) - torch.tensor(Xv)) ** 2).mean())
            if v < best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state); N = topo.unobservable(); res = []
    for s in (777, 888, 999):
        Ut, Xt = gen(s, 1000)
        with torch.no_grad(): E = (net(torch.tensor(Ut)) - torch.tensor(Xt)).numpy()
        res.append((float((E ** 2).mean()), float(((E @ N.T) ** 2).mean()) if len(N) else float("nan")))
    return {"kind": kind, "seed": seed, "val_mse": best, "mse": float(np.mean([a for a, _ in res])), "mse_null_dirs": float(np.mean([b for _, b in res]))}


def ridge_exp(topo_name, dynamics):
    topo = Topo(topo_name); N = topo.unobservable(); P = 64

    def gen(s, cnt):
        rng = np.random.default_rng(s); U, X = [], []
        for _ in range(cnt):
            o, m, x, *_ = simulate(topo, rng=rng, dynamics=dynamics)
            U.append(np.concatenate([o[:, topo.tanks], m[:, topo.tanks].astype(np.float32)], axis=1)); X.append(x)
        return np.stack(U).astype(np.float32), np.stack(X).astype(np.float32)

    def design(U):
        nn_, T, d = U.shape; pad = np.concatenate([np.zeros((nn_, P - 1, d), np.float32), U], axis=1)
        X = np.stack([pad[:, t:t + P].reshape(nn_, -1) for t in range(T)], axis=1)
        return np.concatenate([X, np.ones((nn_, T, 1), np.float32)], axis=2).reshape(nn_ * T, -1)

    Utr, Xtr = gen(1000, 8000); Uv, Xv = gen(123, 500)
    A = design(Utr).astype(np.float64); G = A.T @ A; B = A.T @ Xtr.reshape(-1, topo.n).astype(np.float64); Av = design(Uv); best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B); v = float(((Av @ W - Xv.reshape(-1, topo.n)) ** 2).mean())
        if best is None or v < best[0]: best = (v, lam, W)
    v, lam, W = best; r = []
    for s in (777, 888, 999):
        Ut, Xt = gen(s, 1000); E = design(Ut) @ W - Xt.reshape(-1, topo.n)
        r.append((float((E ** 2).mean()), float(((E @ N.T) ** 2).mean()) if len(N) else float("nan")))
    return {"lambda": lam, "mse": float(np.mean([a for a, _ in r])), "mse_null_dirs": float(np.mean([b for _, b in r]))}


def neural(topo_name, dynamics, nproc):
    from multiprocessing import Pool
    rid = ridge_exp(topo_name, dynamics); print("ridge", rid, flush=True)
    jobs = [(k, hp, 3e-3, s, topo_name, dynamics) for (k, hp) in (("gru", 64), ("graph", (32, 64))) for s in range(3)]
    with Pool(nproc) as p: res = p.map(neural_job, jobs)
    out = {"ridge": rid, "runs": res}
    for k in ("gru", "graph"):
        rs = [r for r in res if r["kind"] == k]
        out[k] = {"mse_mean": float(np.mean([r["mse"] for r in rs])), "mse_sd": float(np.std([r["mse"] for r in rs], ddof=1)),
                  "mse_null_dirs": float(np.mean([r["mse_null_dirs"] for r in rs])), "per_seed": [r["mse"] for r in rs]}
    json.dump(out, open(os.path.join(RES, f"topo_neural_{topo_name}_{dynamics}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("ridge", "gru", "graph")}, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        for dyn in ("linear", "nonlinear"):
            topo = Topo("paper")
            for s in (777, 888):
                a, b = np.random.default_rng(s), np.random.default_rng(s)
                for _ in range(3):
                    o1, m1, x1, _ = generate_trajectory(T=64, rng=a, p_event=0.08, dynamics=dyn)
                    o2, m2, x2, *_ = simulate(topo, rng=b, dynamics=dyn)
                    assert np.allclose(o1, o2, atol=1e-4) and np.array_equal(m1, m2) and np.allclose(x1, x2, atol=1e-4), "mismatch " + dyn
        print("generic simulator reproduces data.generate_trajectory (linear and nonlinear) for the paper topology")
        for nm in TOPOS: print(nm, "unobservable directions:", len(Topo(nm).unobservable()))
    elif mode == "classical":
        classical(sys.argv[2], sys.argv[3])
    elif mode == "neural":
        neural(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
