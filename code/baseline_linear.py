import json, os
import numpy as np
from scipy.stats import rankdata
from benchmark import generate_trajectory

P = 6
V_ANTI = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "baseline_linear.json")


def sample(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        kinds = np.full(64, "calm", dtype=object)
        for (t0, t1, kind, node) in ev:
            kinds[t0:t1] = kind
        out.append((o, m, x, kinds))
    return out


def windows(o, m, t, use_current):
    end = t + 1 if use_current else t
    feats = []
    for s in range(end - P, end):
        if s < 0:
            feats.append(np.zeros(4))
        else:
            feats.append(np.r_[o[s, :2], m[s, :2].astype(float)])
    return np.concatenate(feats + [np.ones(1)])


def design(data, use_current):
    X, Y, Yt, Mt = [], [], [], []
    for o, m, x, kinds in data:
        for t in range(64):
            X.append(windows(o, m, t, use_current))
            Y.append(x[t])
            Yt.append(o[t, :2]); Mt.append(m[t, :2])
    return np.array(X), np.array(Y), np.array(Yt), np.array(Mt)


def ridge(X, Y, lam=1e-3):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def auc(scores, labels):
    r = rankdata(scores)
    n1 = labels.sum(); n0 = labels.size - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    train = sample(1, 300)
    Xc, Yc, _, _ = design(train, use_current=True)
    W_est = ridge(Xc, Yc)
    Xp, _, Yt_tr, Mt_tr = design(train, use_current=False)
    W_det = ridge(Xp, Yt_tr * Mt_tr)

    results = {"description": "equal-information linear baseline (tank readings and masks over the last 6 steps)",
               "test_sets": []}
    for s in (777, 888, 999):
        test = sample(s, 1000)
        Xe, Ye, _, _ = design(test, use_current=True)
        E = Xe @ W_est - Ye
        mse = float((E ** 2).mean())
        mse_tank = float((E[:, :2] ** 2).mean()); mse_junc = float((E[:, 2:] ** 2).mean())
        anti = float(((E @ V_ANTI) ** 2).mean())
        Xp_e, _, Yt_e, Mt_e = design(test, use_current=False)
        pred_t = Xp_e @ W_det
        inn = np.abs(Yt_e - pred_t) * Mt_e
        score = inn.sum(axis=1) / np.maximum(Mt_e.sum(axis=1), 1)
        score = score.reshape(1000, 64).ravel()
        kinds = np.stack([k for _, _, _, k in test]).ravel()
        active = kinds != "calm"
        res = {"seed_eval": s, "mse": mse, "mse_tank": mse_tank, "mse_junction": mse_junc, "mse_antisym": anti,
               "auc_innovation": auc(score, active)}
        for kind in ("dropout", "pump_trip", "demand_spike"):
            sel = (kinds == kind) | (kinds == "calm")
            res[f"auc_innovation_{kind}"] = auc(score[sel], (kinds == kind)[sel])
        results["test_sets"].append(res)
        print(s, round(mse, 4), round(anti, 4), round(res["auc_innovation"], 3), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("written", OUT)
