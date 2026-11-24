import json, os
import numpy as np
from data import generate_trajectory, MEASUREMENT_NOISE_STD
import kalman_smoother as K
from kalman_baselines import kf_general
from baseline_linear import ridge
import data as D
from network import STATE_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
R = MEASUREMENT_NOISE_STD ** 2
TEST_SEEDS = (777, 888, 999)
N_TEST = 1000


def sample_full(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        out.append((o, m, x))
    return out


out = {}

train = sample_full(1, 300)
Xtr = np.stack([x for _, _, x in train])
proj_tr = Xtr @ V
const_mean = float(proj_tr.mean())
time_mean = proj_tr.mean(axis=0)
prior = {"train_variance_in_v": float(proj_tr.var()),
         "const_mean": const_mean, "per_test_set": []}
for s in TEST_SEEDS:
    te = sample_full(s, N_TEST)
    pj = np.stack([x for _, _, x in te]) @ V
    prior["per_test_set"].append({
        "seed_eval": s,
        "mse_const_mean": float(((pj - const_mean) ** 2).mean()),
        "mse_time_mean": float(((pj - time_mean[None, :]) ** 2).mean())})
prior["mse_const_mean_avg"] = float(np.mean([p["mse_const_mean"] for p in prior["per_test_set"]]))
prior["mse_time_mean_avg"] = float(np.mean([p["mse_time_mean"] for p in prior["per_test_set"]]))
out["prior_only_J2_minus_J4"] = prior
print("prior-only", prior["mse_const_mean_avg"], prior["mse_time_mean_avg"], flush=True)

edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
Ep = np.diag([1, 1, 0, 0, 0, 0.0])
C = np.zeros((2, 6)); C[0, 0] = 1.0; C[1, 1] = 1.0


def build_A(k_edges):
    L = np.zeros((6, 6))
    for e, (a, b) in enumerate(edges):
        L[a, a] += k_edges[e]; L[b, b] += k_edges[e]
        L[a, b] -= k_edges[e]; L[b, a] -= k_edges[e]
    return np.eye(6) - 0.5 * (L + 0.6 * Ep)


def gram(A, T, Cm=C):
    W = np.zeros((6, 6)); Ak = np.eye(6)
    for _ in range(T):
        W += Ak.T @ Cm.T @ Cm @ Ak
        Ak = A @ Ak
    return W


def weakest(W):
    ev, U = np.linalg.eigh(W / R)
    lam = ev[0]
    return float(lam), float((1.0 + lam) ** -0.5), float(abs(U[:, 0] @ V))


g = {}
A_hom = build_A(np.full(6, 0.35))
for T in (6, 64):
    lam, sd, al = weakest(gram(A_hom, T))
    g[f"homogeneous_T{T}"] = {"lambda_min_normalised": lam, "posterior_sd_unit_prior": sd, "alignment_with_v": al}
rng = np.random.default_rng(2024)
het = {6: [], 64: []}
for _ in range(200):
    A = build_A(rng.uniform(0.25, 0.45, 6))
    for T in (6, 64):
        het[T].append(weakest(gram(A, T))[:2])
for T in (6, 64):
    h = np.array(het[T])
    g[f"heterogeneous_T{T}"] = {
        "lambda_min_normalised_median": float(np.median(h[:, 0])),
        "lambda_min_normalised_min": float(h[:, 0].min()),
        "lambda_min_normalised_max": float(h[:, 0].max()),
        "posterior_sd_unit_prior_median": float(np.median(h[:, 1]))}
out["gramian_noise_normalised"] = g
print("gramian", json.dumps(g)[:300], flush=True)

A_or = K.build_A(); Q_or, _ = K.calibrate_Q(A_or)
bfun = K.exogenous_input
JUNCTIONS = {"J1": 2, "J2": 3, "J3": 4, "J4": 5}


def augment(data, j, rng):
    res = []
    for o, m, x in data:
        o2 = o.copy(); m2 = m.copy()
        if j is not None:
            o2[:, j] = x[:, j] + rng.normal(0, MEASUREMENT_NOISE_STD, x.shape[0])
            m2[:, j] = True
        res.append((o2, m2, x))
    return res


def run_kf(data):
    mse, anti, post_var = [], [], []
    for o, m, x in data:
        x0 = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5); P0 = np.eye(STATE_DIM) * 4.0
        xf, Pf, _, _ = kf_general(o, m, A_or, bfun, Q_or, R, x0, P0)
        E = xf - x
        mse.append((E ** 2).mean()); anti.append(((E @ V) ** 2).mean())
        post_var.append(np.einsum("i,tij,j->t", V, Pf, V).mean())
    return float(np.mean(mse)), float(np.mean(anti)), float(np.mean(post_var))


def feat(o, m, t, P, cols):
    f = []
    for s in range(t + 1 - P, t + 1):
        if s < 0:
            f.append(np.zeros(2 * len(cols)))
        else:
            f.append(np.r_[o[s, cols], m[s, cols].astype(float)])
    return np.concatenate(f + [np.ones(1)])


def design(data, P, cols):
    X, Y = [], []
    for o, m, x in data:
        for t in range(64):
            X.append(feat(o, m, t, P, cols)); Y.append(x[t])
    return np.array(X), np.array(Y)


placements = {"none": None, "J1": 2, "J2": 3, "J3": 4, "J4": 5}
sensor = {}
for name, j in placements.items():
    cols = [0, 1] + ([j] if j is not None else [])
    rng_tr = np.random.default_rng(11)
    tr = augment(train, j, rng_tr)
    Xtr_, Ytr_ = design(tr, 64, cols)
    W = ridge(Xtr_, Ytr_, lam=1e-3)
    kf_r, rg_r = [], []
    for s in TEST_SEEDS:
        te = augment(sample_full(s, N_TEST), j, np.random.default_rng(s + 5))
        kf_r.append(run_kf(te[:300]))
        Xe, Ye = design(te, 64, cols)
        E = Xe @ W - Ye
        rg_r.append((float((E ** 2).mean()), float(((E @ V) ** 2).mean())))
    kf_r = np.array(kf_r); rg_r = np.array(rg_r)
    sensor[name] = {"kf_mse": float(kf_r[:, 0].mean()), "kf_mse_antisym": float(kf_r[:, 1].mean()),
                    "kf_model_posterior_var_antisym": float(kf_r[:, 2].mean()),
                    "ridge64_mse": float(rg_r[:, 0].mean()), "ridge64_mse_antisym": float(rg_r[:, 1].mean())}
    print("sensor", name, sensor[name], flush=True)
out["sensor_placement"] = sensor
out["notes"] = ("KF rows use the first 300 trajectories of each of the three test sets. Extra sensor has the same "
                "noise std as the tanks and is never dropped. Ridge: 300 training trajectories (seed 1), lambda=1e-3.")
with open(os.path.join(RES, "weak_mode_recoverability.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("written weak_mode_recoverability.json")
