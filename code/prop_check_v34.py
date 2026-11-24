import json, os, numpy as np
dt, k, kp = 0.5, 0.35, 0.6
edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
N = np.zeros((6, 6))
for e, (a, b) in enumerate(edges): N[a, e] = 1; N[b, e] = -1
E = np.diag([1, 1, 0, 0, 0, 0.]); C = np.eye(6)[:2]; v = np.array([0, 0, 0, 1, 0, -1.]) / np.sqrt(2)
A0 = np.eye(6) - dt * (k * N @ N.T + kp * E)
A1 = lambda u: -dt * k * N @ np.diag(u) @ N.T
def Mof(A, T): return np.vstack([C @ np.linalg.matrix_power(A, j) for j in range(T)])
def pieces(u, T):
    M0 = Mof(A0, T); a1 = A1(u)
    c = np.concatenate([C @ sum(np.linalg.matrix_power(A0, j - 1 - i) @ a1 @ np.linalg.matrix_power(A0, i) @ v for i in range(j)) if j > 0 else np.zeros(2) for j in range(T)])
    Q, _ = np.linalg.qr(M0, mode="reduced"); U, sv, _ = np.linalg.svd(M0, full_matrices=False); Ur = U[:, sv > 1e-10]
    Pic = c - Ur @ (Ur.T @ c); return M0, c, Pic
def s_of(u, eps, T):
    A = A0 + eps * A1(u); M = Mof(A, T); V = np.linalg.svd(np.outer(v, v), )[0]
    Vp = np.linalg.svd(np.eye(6) - np.outer(v, v))[0][:, :5]
    y = np.linalg.lstsq(M @ Vp, -M @ v, rcond=None)[0]; r = M @ (v + Vp @ y); return float(r @ r), float(np.linalg.eigvalsh(M.T @ M)[0])
rng = np.random.default_rng(0); out = {}
for T in (6, 64):
    rows = []
    for _ in range(5):
        u = rng.uniform(-1, 1, 6); M0, c, Pic = pieces(u, T); th = float(Pic @ Pic)
        for eps in (1e-2, 1e-3):
            s, lam = s_of(u, eps, T); rows.append({"eps": eps, "s_over_eps2": s / eps ** 2, "lam_over_eps2": lam / eps ** 2, "theory": th, "lam_over_s": lam / s})
    out["T%d" % T] = rows
    r3 = [x for x in rows if x["eps"] == 1e-3]
    print("T", T, "max rel dev s/eps^2 vs theory %.2e" % max(abs(x["s_over_eps2"] / x["theory"] - 1) for x in r3), " max |lam/s-1| %.2e" % max(abs(x["lam_over_s"] - 1) for x in rows))
bad = 0
for _ in range(1000):
    u = rng.uniform(-1, 1, 6); w = A1(u) @ v; inspan = np.linalg.norm(w - (w @ v) * v) < 1e-12; cond = abs(u[2] - u[5]) < 1e-12 and abs(u[3] - u[4]) < 1e-12; bad += inspan != cond
u = rng.uniform(-1, 1, 6); u[5] = u[2]; u[4] = u[3]; w = A1(u) @ v; print("mirror-symmetric u: |A1 v - (v.A1v) v| =", np.linalg.norm(w - (w @ v) * v)); print("random u mismatches of the equivalence:", bad)
for T in (6, 64):
    G = np.zeros((2, 2)); f = lambda a, b: (lambda u: float((lambda P: P @ P)(pieces(u, T)[2])))(np.array([0, 0, a, b, -b, -a]))
    G[0, 0] = f(1, 0); G[1, 1] = f(0, 1); G[0, 1] = G[1, 0] = (f(1, 1) - f(1, 0) - f(0, 1)) / 2
    print("T", T, "G =", np.round(G, 6).tolist(), "eigenvalues", np.linalg.eigvalsh(G)); out["G_T%d" % T] = {"G": G.tolist(), "eig": np.linalg.eigvalsh(G).tolist()}
    u = rng.uniform(-1, 1, 6); ua = np.array([0, 0, (u[2] - u[5]) / 2, (u[3] - u[4]) / 2, -(u[3] - u[4]) / 2, -(u[2] - u[5]) / 2]); print("  Pi c(u)^2 vs antisym part only:", float(pieces(u, T)[2] @ pieces(u, T)[2]), float(pieces(ua, T)[2] @ pieces(ua, T)[2]))
json.dump(out, open(os.path.join("..", "results", "prop_check_v34.json"), "w"), indent=1)
