import json, os
import numpy as np
from scipy.stats import t as tdist, ttest_1samp
from benchmark import generate_trajectory
from baseline_linear import sample, ridge, auc
from benchmark import STATE_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
V_ANTI = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def window_features(o, m, t, P):
    feats = []
    for s in range(t + 1 - P, t + 1):
        if s < 0:
            feats.append(np.zeros(4))
        else:
            feats.append(np.r_[o[s, :2], m[s, :2].astype(float)])
    return np.concatenate(feats + [np.ones(1)])


def design(data, P):
    X, Y = [], []
    for o, m, x, kinds in data:
        for t in range(64):
            X.append(window_features(o, m, t, P)); Y.append(x[t])
    return np.array(X), np.array(Y)


def evaluate_ridge(P, train, tests):
    Xtr, Ytr = design(train, P)
    W = ridge(Xtr, Ytr, lam=1e-3)
    out = []
    for test in tests:
        Xe, Ye = design(test, P)
        E = Xe @ W - Ye
        out.append({"mse": float((E ** 2).mean()), "mse_antisym": float(((E @ V_ANTI) ** 2).mean())})
    return out


train = sample(1, 300)
tests = [sample(s, 1000) for s in (777, 888, 999)]
windows = {}
for P in (6, 16, 32, 64):
    r = evaluate_ridge(P, train, tests)
    windows[P] = {"mse": [x["mse"] for x in r], "mse_mean": float(np.mean([x["mse"] for x in r])),
                  "mse_antisym_mean": float(np.mean([x["mse_antisym"] for x in r]))}
    print("window", P, round(windows[P]["mse_mean"], 4), flush=True)

edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
Ep = np.diag([1, 1, 0, 0, 0, 0.0])
C = np.zeros((2, 6)); C[0, 0] = 1.0; C[1, 1] = 1.0


def build_A(k_edges):
    L = np.zeros((6, 6))
    for e, (a, b) in enumerate(edges):
        L[a, a] += k_edges[e]; L[b, b] += k_edges[e]
        L[a, b] -= k_edges[e]; L[b, a] -= k_edges[e]
    return np.eye(6) - 0.5 * (L + 0.6 * Ep)


def gramian_stats(A, T):
    W = np.zeros((6, 6)); Ak = np.eye(6)
    for _ in range(T):
        W += Ak.T @ C.T @ C @ Ak
        Ak = A @ Ak
    ev = np.linalg.eigvalsh(W)
    return float(ev[0]), float(ev[-1]), float(ev[-1] / max(ev[0], 1e-300))


A_hom = build_A(np.full(6, 0.35))
gram = {"homogeneous": {}, "heterogeneous_draws": {}}
for T in (6, 64):
    lo, hi, cond = gramian_stats(A_hom, T)
    gram["homogeneous"][f"T{T}"] = {"lambda_min": lo, "lambda_max": hi, "condition": cond}
rng = np.random.default_rng(2024)
draws = {6: [], 64: []}
for _ in range(200):
    k = rng.uniform(0.25, 0.45, 6)
    A = build_A(k)
    for T in (6, 64):
        draws[T].append(gramian_stats(A, T)[0])
for T in (6, 64):
    d = np.array(draws[T])
    gram["heterogeneous_draws"][f"T{T}"] = {"lambda_min_median": float(np.median(d)),
                                            "lambda_min_min": float(d.min()),
                                            "lambda_min_max": float(d.max())}

R = [json.loads(l[7:]) for l in open(os.path.join(RES, "results_architecture_controls.txt"), encoding="utf-8") if l.startswith("RESULT ")]
prob = {r["seed"]: float(np.mean([e["mse"] for e in r["eval_sets"]])) for r in R if r["config"] == "prob"}


def paired_vs_prob(ridge_mse):
    d = np.array([ridge_mse - prob[s] for s in sorted(prob)])
    se = d.std(ddof=1) / np.sqrt(d.size)
    h = tdist.ppf(0.975, d.size - 1) * se
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "p_value": float(ttest_1samp(d, 0).pvalue),
            "seeds_where_ridge_lower": int((d < 0).sum()), "n": int(d.size),
            "per_seed": [float(x) for x in d]}


paired = {"ridge6_minus_prob": paired_vs_prob(windows[6]["mse_mean"]),
          "ridge64_minus_prob": paired_vs_prob(windows[64]["mse_mean"])}

out = {"ridge_window_length": windows, "observability_gramian": gram, "paired_ridge_vs_prob": paired}
with open(os.path.join(RES, "ridge_gramian_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("written ridge_gramian_analysis.json")
