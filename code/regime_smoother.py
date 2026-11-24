import json, os
import numpy as np

from benchmark import generate_trajectory
from kalman import build_A, calibrate_Q, kalman_filter, rts_smoother

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "smoother_uncorrected_timing.json")
A = build_A()


def trajectories(p_event, seed, n=300, T=64):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o, m, x, _ = generate_trajectory(T=T, rng=rng, p_event=p_event)
        out.append((o, m, x))
    return out


def mse_pair(Q, data):
    f, s = [], []
    for o, m, x in data:
        xf, Pf, xp, Pp = kalman_filter(o, m, A, Q, None)
        xs = rts_smoother(xf, Pf, xp, Pp, A)
        f.append(((xf - x) ** 2).mean())
        s.append(((xs - x) ** 2).mean())
    return float(np.mean(f)), float(np.mean(s))


Q_calm, _ = calibrate_Q(A, p_event=0.0, seed=4242)
Q_event, _ = calibrate_Q(A, p_event=0.08, seed=4343)
calm = trajectories(0.0, 777)
events = trajectories(0.08, 888)

rows = [
    ("Calm, Q from calm data", Q_calm, calm),
    ("Calm, Q from event data", Q_event, calm),
    ("Events, Q from event data", Q_event, events),
    ("Events, Q from calm data", Q_calm, events),
]
regimes = []
for label, Q, data in rows:
    kf, rts = mse_pair(Q, data)
    regimes.append({"label": label, "kf": kf, "rts": rts, "ratio": rts / kf})
    print(f"{label:28s} filter {kf:.5f}  smoother {rts:.5f}  ratio {rts / kf:.2f}", flush=True)

out = {"description": "Kalman filter and RTS smoother MSE by regime, 300 trajectories per regime, "
                      "produced by code/regime_smoother.py. Q calibrated on seeds 4242 (calm) and 4343 "
                      "(events), not on evaluation seeds.",
       "regimes": regimes}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("written", OUT)
