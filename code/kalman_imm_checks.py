import json, os
import numpy as np
import kalman_smoother as K
from kalman_baselines import kf_general
from data import generate_trajectory, MEASUREMENT_NOISE_STD
import data as D
from network import STATE_DIM
import imm_true_regimes as E

R = MEASUREMENT_NOISE_STD ** 2
out = {}

A = K.build_A(); Q_orig, _ = K.calibrate_Q(A)
Q_new = E.calibrate_Q(False)
rng = np.random.default_rng(777); data = [generate_trajectory(T=64, rng=rng, p_event=0.08) for _ in range(1000)]
x0d = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5); P0d = np.eye(STATE_DIM) * 4.0


def mse_orig(n, Q):
    m = []
    for o, mk, x, ev in data[:n]:
        xf, *_ = kf_general(o, mk, A, K.exogenous_input, Q, R, x0d, P0d)
        m.append(((xf - x) ** 2).mean())
    return float(np.mean(m))


def mse_new(n, Q, prior):
    m = []
    for o, mk, x, ev in data[:n]:
        e = E.run_kf(o, mk, 0.0, Q, prior)
        m.append(((e - x) ** 2).mean())
    return float(np.mean(m))


out["original_filter_1000_traj_seed777"] = mse_orig(1000, Q_orig)
out["original_filter_first300"] = mse_orig(300, Q_orig)
out["new_filter_first300_Qnew"] = mse_new(300, Q_new, E.start_prior("diffuse", False))
out["new_filter_first300_Qorig"] = mse_new(300, Q_orig, E.start_prior("diffuse", False))
out["Q_diag_orig"] = [float(v) for v in np.diag(Q_orig)]
out["Q_diag_new"] = [float(v) for v in np.diag(Q_new)]
print(json.dumps({k: v for k, v in out.items() if not k.startswith("Q_")}, indent=1), flush=True)

rng = np.random.default_rng(5); viol = 0; checks = 0
for _ in range(20):
    o, mk, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
    base = E.run_imm(o, mk, 0.0, E.start_prior("known", False))
    for t0 in (10, 30, 50):
        o2 = o.copy(); o2[t0 + 1:] += rng.normal(0, 1.0, o2[t0 + 1:].shape)
        pert = E.run_imm(o2, mk, 0.0, E.start_prior("known", False))
        checks += 1; viol += int(not np.allclose(base[:t0 + 1], pert[:t0 + 1], atol=1e-9))
out["imm_causality_checks"] = checks; out["imm_causality_violations"] = viol

rng = np.random.default_rng(9); overlaps = 0; events_total = 0; max_active = 0
for _ in range(2000):
    _, _, _, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
    events_total += len(ev)
    ev_sorted = sorted(ev, key=lambda e: e[0])
    for a, b in zip(ev_sorted, ev_sorted[1:]):
        overlaps += int(b[0] < a[1])
out["events_simulated"] = events_total; out["overlapping_event_pairs"] = overlaps
print(json.dumps({k: v for k, v in out.items() if not k.startswith("Q_")}, indent=1))
json.dump(out, open(os.path.join("..", "results", "kalman_imm_checks.json"), "w"), indent=1)
