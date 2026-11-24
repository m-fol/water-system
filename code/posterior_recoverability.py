import json, os
import numpy as np
from benchmark import simulate

R = 0.03 ** 2
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
Ep = np.diag([1, 1, 0, 0, 0, 0.0]); C = np.zeros((2, 6)); C[0, 0] = 1.0; C[1, 1] = 1.0


def build_A(k):
    Lm = np.zeros((6, 6))
    for e, (a, b) in enumerate(edges):
        Lm[a, a] += k[e]; Lm[b, b] += k[e]; Lm[a, b] -= k[e]; Lm[b, a] -= k[e]
    return np.eye(6) - 0.5 * (Lm + 0.6 * Ep)


def gram(A, T):
    W = np.zeros((6, 6)); Ak = np.eye(6)
    for _ in range(T):
        W += Ak.T @ C.T @ C @ Ak; Ak = A @ Ak
    return W


rng = np.random.default_rng(1)
X = np.concatenate([simulate(T=64, rng=rng)[2] for _ in range(300)], axis=0)
P_A = np.cov(X.T) + 1e-8 * np.eye(6)
P_B = 0.25 * np.eye(6)


def reduction(P0, W):
    Pp = np.linalg.inv(np.linalg.inv(P0) + W / R)
    ev, U = np.linalg.eigh(W)
    u = U[:, 0]
    red_v = 1 - (V @ Pp @ V) / (V @ P0 @ V)
    red_u = 1 - (u @ Pp @ u) / (u @ P0 @ u)
    return float(red_v), float(red_u), float(abs(u @ V))


out = {"prior_A_variance_along_v": float(V @ P_A @ V), "prior_B_variance_along_v": float(V @ P_B @ V)}
k_hom = np.full(6, 0.35)
r2 = np.random.default_rng(2024); draws = [r2.uniform(0.25, 0.45, 6) for _ in range(200)]
for pname, P0 in (("A_empirical", P_A), ("B_independent", P_B)):
    for T in (6, 64):
        h = reduction(P0, gram(build_A(k_hom), T))
        het = np.array([reduction(P0, gram(build_A(k), T)) for k in draws])
        out[f"{pname}_T{T}"] = {
            "homogeneous": {"reduction_along_v": h[0], "reduction_along_weakest": h[1], "cos_weakest_v": h[2]},
            "heterogeneous": {"reduction_along_v_median": float(np.median(het[:, 0])), "reduction_along_v_min": float(het[:, 0].min()), "reduction_along_v_max": float(het[:, 0].max()),
                              "reduction_along_weakest_median": float(np.median(het[:, 1])), "reduction_along_weakest_max": float(het[:, 1].max()),
                              "cos_weakest_v_median": float(np.median(het[:, 2])), "cos_weakest_v_min": float(het[:, 2].min())}}
json.dump(out, open(os.path.join("..", "results", "posterior_recoverability.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
