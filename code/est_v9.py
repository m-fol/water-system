import json, os, sys
import numpy as np
from data import (ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, TANK_IDX, JUNCTION_IDX)
from network import STATE_DIM
from sim_v9 import simulate, diurnal

DT = 0.5
R = MEASUREMENT_NOISE_STD ** 2
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
L = np.diag(ADJ.sum(axis=1)) - ADJ


def A_regime(trip_tank=None):
    damp = np.zeros((STATE_DIM, STATE_DIM))
    for k, idx in enumerate(TANK_IDX):
        if trip_tank != k:
            damp[idx, idx] = PUMP_GAIN
    return np.eye(STATE_DIM) + DT * (-FLOW_GAIN * L - damp)


def b_regime(t, phase, trip_tank=None, spike_j=None):
    b = np.zeros(STATE_DIM)
    for k, idx in enumerate(TANK_IDX):
        if trip_tank != k:
            b[idx] = DT * PUMP_GAIN * TANK_SETPOINT
    for k, idx in enumerate(JUNCTION_IDX):
        mult = 4.0 if spike_j == k else 1.0
        b[idx] = -DT * BASE_DEMAND * diurnal(t, phase) * mult
    return b


REGIMES = [(None, None)] + [(k, None) for k in range(2)] + [(None, j) for j in range(4)]
A_R = [A_regime(r[0]) for r in REGIMES]
SIGMA_D = 0.05


def Q_regime(r):
    q = np.full(STATE_DIM, 1e-6)
    for k, idx in enumerate(JUNCTION_IDX):
        q[idx] = (DT * SIGMA_D * (4.0 if r[1] == k else 1.0)) ** 2 + 1e-6
    return np.diag(q)


Q_R = [Q_regime(r) for r in REGIMES]
p_trip, p_spike, exit_p = 0.08 / 3 / 2, 0.08 / 3 / 4, 1.0 / 6.0
PI = np.zeros((7, 7))
PI[0, 1:3] = p_trip; PI[0, 3:] = p_spike; PI[0, 0] = 1 - PI[0, 1:].sum()
for r in range(1, 7):
    PI[r, 0] = exit_p; PI[r, r] = 1 - exit_p


def start_prior(mode, random):
    if mode == "diffuse":
        return np.full(STATE_DIM, TANK_SETPOINT - 0.5), np.eye(STATE_DIM) * 4.0
    m = np.zeros(STATE_DIM); m[TANK_IDX] = TANK_SETPOINT; m[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    return m, (np.eye(STATE_DIM) * 0.25 if random else np.eye(STATE_DIM) * 1e-4)


def update(x, P, z, mask):
    av = np.where(mask)[0]
    if len(av) == 0:
        return x, P, 0.0
    H = np.zeros((len(av), STATE_DIM)); H[np.arange(len(av)), av] = 1.0
    P = 0.5 * (P + P.T)
    S = H @ P @ H.T + R * np.eye(len(av))
    K = P @ H.T @ np.linalg.inv(S)
    innov = z[av] - H @ x
    ll = -0.5 * (innov @ np.linalg.solve(S, innov) + np.linalg.slogdet(S)[1])
    Pn = (np.eye(STATE_DIM) - K @ H) @ P
    return x + K @ innov, 0.5 * (Pn + Pn.T), ll


def run_kf(obs, mask, phase, Q, prior):
    x, P = prior; out = np.zeros((obs.shape[0], STATE_DIM))
    for t in range(obs.shape[0]):
        xp = A_R[0] @ x + b_regime(t, phase); Pp = A_R[0] @ P @ A_R[0].T + Q
        x, P, _ = update(xp, Pp, obs[t], mask[t]); out[t] = x
    return out


def run_imm(obs, mask, phase, prior):
    n = len(REGIMES)
    xs = [prior[0].copy() for _ in range(n)]; Ps = [prior[1].copy() for _ in range(n)]
    mu = np.array([1 - 0.0533] + [0.0133] * 2 + [0.00667] * 4); mu = mu / mu.sum()
    out = np.zeros((obs.shape[0], STATE_DIM))
    for t in range(obs.shape[0]):
        cbar = PI.T @ mu
        xm, Pm = [], []
        for j in range(n):
            w = PI[:, j] * mu / cbar[j]
            xj = sum(w[i] * xs[i] for i in range(n))
            Pj = sum(w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(n))
            xm.append(xj); Pm.append(Pj)
        lik = np.zeros(n)
        for j in range(n):
            xp = A_R[j] @ xm[j] + b_regime(t, phase, *REGIMES[j]); Pp = A_R[j] @ Pm[j] @ A_R[j].T + Q_R[j]
            xs[j], Ps[j], ll = update(xp, Pp, obs[t], mask[t]); lik[j] = ll
        lik = np.nan_to_num(lik, nan=-1e9, neginf=-1e9)
        post = cbar * np.exp(lik - lik.max()) + 1e-300; mu = post / post.sum()
        out[t] = sum(mu[j] * xs[j] for j in range(n))
    return out


def make_set(seed, n, random):
    rng = np.random.default_rng(seed); irng = np.random.default_rng(seed + 10_000)
    return [simulate(T=64, rng=rng, random=random, init_rng=irng) for _ in range(n)]


def calibrate_Q(random, n=200, seed=4242):
    rng = np.random.default_rng(seed); irng = np.random.default_rng(seed + 10_000); res = []
    for _ in range(n):
        o, m, x, k, ph, x0 = simulate(T=64, rng=rng, random=random, init_rng=irng)
        prev = x0
        for t in range(64):
            res.append(x[t] - (A_R[0] @ prev + b_regime(t, ph))); prev = x[t]
    return np.cov(np.array(res).T)


def mse(E):
    return float((E ** 2).mean()), float(((E @ V) ** 2).mean())


def evaluate(random, n_test=300):
    Q = calibrate_Q(random)
    train = make_set(1, 300, random)
    tm = np.stack([x for _, _, x, _, _, _ in train]).mean(axis=0) if not random else np.stack([x for _, _, x, _, _, _ in train]).mean(axis=(0, 1))
    res = {}
    kinds_all = ["calm", "dropout", "pump_trip", "demand_spike"]
    for s in (777, 888, 999):
        for o, m, x, k, ph, x0 in make_set(s, n_test, random):
            ests = {"time_only": np.broadcast_to(tm, x.shape),
                    "kf_prior": run_kf(o, m, ph, Q, start_prior("diffuse", random)),
                    "kf_known_start": run_kf(o, m, ph, Q, start_prior("known", random)),
                    "imm_diffuse": run_imm(o, m, ph, start_prior("diffuse", random)),
                    "imm_known_start": run_imm(o, m, ph, start_prior("known", random))}
            for name, e in ests.items():
                E = e - x
                d = res.setdefault(name, {"se": 0.0, "anti": 0.0, "n": 0, "by": {kk: [0.0, 0] for kk in kinds_all}})
                d["se"] += float((E ** 2).sum()); d["anti"] += float(((E @ V) ** 2).sum()); d["n"] += E.shape[0]
                for kk in kinds_all:
                    sel = k == kk
                    d["by"][kk][0] += float((E[sel] ** 2).sum()); d["by"][kk][1] += int(sel.sum())
        print(random, s, {n: round(v["se"] / (v["n"] * 6), 4) for n, v in res.items()}, flush=True)
    return {n: {"mse": v["se"] / (v["n"] * 6), "antisym": v["anti"] / v["n"],
                "by_regime": {kk: v["by"][kk][0] / (v["by"][kk][1] * 6) for kk in kinds_all if v["by"][kk][1]}} for n, v in res.items()}


if __name__ == "__main__":
    mode = sys.argv[1]
    out = evaluate(mode == "random")
    json.dump(out, open(os.path.join("..", "results", f"est_v9_{mode}.json"), "w"), indent=1)
    print(json.dumps({k: (round(v["mse"], 4), round(v["antisym"], 4)) for k, v in out.items()}))
