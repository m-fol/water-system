import json, os
import numpy as np
from benchmark import MEASUREMENT_NOISE_STD
from benchmark import STATE_DIM
import imm_true_regimes as E
from benchmark import simulate

R = MEASUREMENT_NOISE_STD ** 2
A0 = E.A_R[0]


def kf_rts(obs, mask, A, bfun, Q, prior):
    T = obs.shape[0]
    xf = np.zeros((T, STATE_DIM)); Pf = np.zeros((T, STATE_DIM, STATE_DIM))
    xp = np.zeros((T, STATE_DIM)); Pp = np.zeros((T, STATE_DIM, STATE_DIM))
    x, P = prior
    for t in range(T):
        xp[t] = A @ x + bfun(t); Pp[t] = A @ P @ A.T + Q
        x, P, _ = E.update(xp[t], Pp[t], obs[t], mask[t])
        xf[t], Pf[t] = x, P
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        G = Pf[t] @ A.T @ np.linalg.pinv(Pp[t + 1])
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
    return xf, xs


def calibrate_Q(p_event, seed, n=200):
    rng = np.random.default_rng(seed); res = []
    for _ in range(n):
        o, m, x, k, ph, x0 = simulate(T=64, rng=rng, p_event=p_event)
        prev = x0
        for t in range(64):
            res.append(x[t] - (A0 @ prev + E.b_regime(t, 0.0))); prev = x[t]
    return np.cov(np.array(res).T)


def traj(p_event, seed, n):
    rng = np.random.default_rng(seed)
    return [simulate(T=64, rng=rng, p_event=p_event)[:3] for _ in range(n)]


def mse(data, A, bfun, Q, prior):
    f, s = [], []
    for o, m, x in data:
        xf, xs = kf_rts(o, m, A, bfun, Q, prior)
        f.append(((xf - x) ** 2).mean()); s.append(((xs - x) ** 2).mean())
    return float(np.mean(f)), float(np.mean(s))


prior = E.start_prior("diffuse", False)
out = {}
Q4242 = calibrate_Q(0.08, 4242)
ev = traj(0.08, 777, 1000)
kf, rts = mse(ev, A0, lambda t: E.b_regime(t, 0.0), Q4242, prior)
out["oracle"] = {"kf": kf, "rts": rts}
print("oracle", kf, rts, flush=True)
tr = [simulate(T=64, rng=np.random.default_rng(1 + 0), p_event=0.08)]
rng = np.random.default_rng(1); Xs, Ys = [], []
for _ in range(300):
    o, m, x, k, ph, x0 = simulate(T=64, rng=rng, p_event=0.08)
    prev = x0
    for t in range(64):
        ang = 2 * np.pi * t / 48
        Xs.append(np.r_[prev, 1.0, np.sin(ang), np.cos(ang)]); Ys.append(x[t]); prev = x[t]
Xs, Ys = np.array(Xs), np.array(Ys)
B, *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
Af = B[:STATE_DIM].T; c0, c1, c2 = B[STATE_DIM], B[STATE_DIM + 1], B[STATE_DIM + 2]
Qf = np.cov((Ys - Xs @ B).T)
bf = lambda t: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)
kf, rts = mse(ev, Af, bf, Qf, prior)
out["fitted"] = {"kf": kf, "rts": rts}
print("fitted", kf, rts, flush=True)
Qc, Qe = calibrate_Q(0.0, 4242), calibrate_Q(0.08, 4343)
calm, events = traj(0.0, 777, 300), traj(0.08, 888, 300)
rows = []
for label, Q, d in (("Calm, Q from calm data", Qc, calm), ("Calm, Q from event data", Qe, calm),
                    ("Events, Q from event data", Qe, events), ("Events, Q from calm data", Qc, events)):
    kf, rts = mse(d, A0, lambda t: E.b_regime(t, 0.0), Q, prior)
    rows.append({"label": label, "kf": kf, "rts": rts, "ratio": rts / kf}); print(rows[-1], flush=True)
out["regimes"] = rows
json.dump(out, open(os.path.join("..", "results", "kalman_baselines_corrected.json"), "w"), indent=1)
json.dump({"description": 'Kalman filter and RTS smoother MSE by regime, 300 trajectories per regime, corrected timing (code/kalman_baselines_corrected.py). Q calibrated on seeds 4242 (calm) and 4343 (events), not on evaluation seeds.', "regimes": out["regimes"]}, open(os.path.join("..", "results", "smoother.json"), "w"), indent=1)
json.dump({"oracle": out["oracle"], "fitted": out["fitted"], "description": 'Kalman filter and RTS smoother MSE on the 1,000 evaluation trajectories (seed 777), corrected timing (code/kalman_baselines_corrected.py). Oracle Q calibrated on seed 4242, not an evaluation seed. The earlier values (an earlier results file, not included here) used the forcing one step late.'}, open(os.path.join("..", "results", "baselines.json"), "w"), indent=2)
