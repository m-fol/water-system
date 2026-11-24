import numpy as np
import torch

from data import generate_trajectory
from train_architectures import make
from network import STATE_DIM

N_TRAJ = 30
T = 64
rng = np.random.default_rng(12345)


def window_features(o, m, t, P):
    feats = []
    for s in range(t + 1 - P, t + 1):
        if s < 0:
            feats.append(np.zeros(4))
        else:
            feats.append(np.r_[o[s, :2], m[s, :2].astype(float)])
    return np.concatenate(feats + [np.ones(1)])


def perturb_future(o, m, t0, rng):
    o2, m2 = o.copy(), m.copy()
    o2[t0 + 1:, :2] = rng.normal(0.0, 20.0, size=(T - t0 - 1, 2)).astype(np.float32)
    m2[t0 + 1:, :2] = rng.random((T - t0 - 1, 2)) < 0.5
    return o2, m2


ridge_fail = {6: 0, 64: 0}
ridge_checks = 0
trajs = []
for _ in range(N_TRAJ):
    o, m, _, _ = generate_trajectory(T=T, rng=rng, p_event=0.08)
    trajs.append((o, m))
    for t0 in range(0, T - 1, 5):
        o2, m2 = perturb_future(o, m, t0, rng)
        for P in (6, 64):
            ridge_checks += 1
            a = window_features(o, m, t0, P)
            b = window_features(o2, m2, t0, P)
            if not np.array_equal(a, b):
                ridge_fail[P] += 1

print(f"ridge features: {ridge_checks} checks per window, failures P=6: {ridge_fail[6]}, P=64: {ridge_fail[64]}")

torch.manual_seed(0)
configs = ["gnn", "pointnet", "pointnet_id", "pointnetpp", "prob", "diag_hom"]
nn_fail = {c: 0 for c in configs}
nn_checks = {c: 0 for c in configs}
for c in configs:
    net = make(c)
    net.eval()
    for o, m in trajs[:10]:
        O = torch.tensor(o[None], dtype=torch.float32)
        with torch.no_grad():
            out1 = net(O)
        mu1 = out1[0] if isinstance(out1, tuple) else out1
        for t0 in range(0, T - 1, 9):
            O2 = O.clone()
            O2[0, t0 + 1:, :2] = torch.tensor(rng.normal(0.0, 20.0, size=(T - t0 - 1, 2)), dtype=torch.float32)
            with torch.no_grad():
                out2 = net(O2)
            mu2 = out2[0] if isinstance(out2, tuple) else out2
            nn_checks[c] += 1
            if not torch.equal(mu1[0, :t0 + 1], mu2[0, :t0 + 1]):
                nn_fail[c] += 1
    print(f"{c}: {nn_checks[c]} checks, failures {nn_fail[c]}")

ok = ridge_fail[6] == 0 and ridge_fail[64] == 0 and all(v == 0 for v in nn_fail.values())
print("ALL CAUSALITY CHECKS PASSED" if ok else "CAUSALITY FAILURE")
