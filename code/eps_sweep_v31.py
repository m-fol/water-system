import json, os, numpy as np
dt, k, kp = 0.5, 0.35, 0.6; sv2 = 0.03 ** 2; P0v = 0.25
edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
N = np.zeros((6, 6))
for e, (a, b) in enumerate(edges): N[a, e] = 1; N[b, e] = -1
E = np.diag([1, 1, 0, 0, 0, 0.]); C = np.eye(6)[:2]; v = np.array([0, 0, 0, 1, 0, -1.]) / np.sqrt(2)
swap = {2: 5, 5: 2, 3: 4, 4: 3, 0: 0, 1: 1}


def A_of(kv): return np.eye(6) - dt * (N @ np.diag(kv) @ N.T + kp * E)


def gram(A, T):
    W = np.zeros((6, 6)); M = np.eye(6)
    for _ in range(T): W += M.T @ C.T @ C @ M; M = A @ M
    return W


def stats(A, T):
    W = gram(A, T); w, V = np.linalg.eigh(W); Pp = np.linalg.inv(np.eye(6) / P0v + W / sv2)
    return w[0], 1 - (v @ Pp @ v) / P0v


A0 = A_of(k * np.ones(6)); print("homog: rank", np.linalg.matrix_rank(np.vstack([C @ np.linalg.matrix_power(A0, i) for i in range(6)])), "v eigen", np.linalg.norm(A0 @ v - 0.65 * v))
rng = np.random.default_rng(0); eps = np.logspace(-4, -0.5, 10); out = {"eps": eps.tolist()}
for kind in ("random", "antisym", "sym"):
    for T in (6, 64):
        lam = np.zeros(len(eps)); red = np.zeros(len(eps))
        for i, e in enumerate(eps):
            ls = []; rs = []
            for _ in range(200):
                u = rng.uniform(-1, 1, 6)
                if kind == "antisym": u = u - u[[swap[j] for j in range(6)]]; u /= max(1e-12, np.abs(u).max())
                if kind == "sym": u = u + u[[swap[j] for j in range(6)]]; u /= max(1e-12, np.abs(u).max())
                l, r = stats(A_of(k * (1 + e * u)), T); ls.append(l); rs.append(r)
            lam[i] = np.median(ls); red[i] = np.median(rs)
        sl = np.polyfit(np.log(eps[:6]), np.log(np.maximum(lam[:6], 1e-300)), 1)[0]
        sr = np.polyfit(np.log(eps[:6]), np.log(np.maximum(red[:6], 1e-300)), 1)[0]
        out["%s_T%d" % (kind, T)] = {"lam_min_median": lam.tolist(), "var_red_median": red.tolist(), "slope_lam": sl, "slope_red": sr}
        print(kind, "T", T, "slope lam %.2f  slope var-red %.2f" % (sl, sr), " lam@eps=1e-2 %.2e  red@1e-2 %.2e" % (np.interp(1e-2, eps, lam), np.interp(1e-2, eps, red)))
json.dump(out, open(os.path.join("..", "results", "eps_sweep_v31.json"), "w"), indent=1)
