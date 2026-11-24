import itertools, json, os
import numpy as np
import benchmark as TD
DT, KP, K0 = TD.DT, TD.KP, TD.K0


def autos(n, tanks, edges, fixed_extra=()):
    J = [j for j in range(n) if j not in tanks]; es = {frozenset(e) for e in edges}; out = []
    for p in itertools.permutations(J):
        m = {**{t: t for t in tanks}, **dict(zip(J, p))}
        if all(m[x] == x for x in fixed_extra) and {frozenset((m[a], m[b])) for a, b in edges} == es: out.append(m)
    return out


def perm_mat(m, n):
    Pm = np.zeros((n, n))
    for a, b in m.items(): Pm[b, a] = 1
    return Pm


def Mof(A, C, T): return np.vstack([C @ np.linalg.matrix_power(A, j) for j in range(T)])
def Cof(n, nodes):
    C = np.zeros((len(nodes), n))
    for i, j in enumerate(nodes): C[i, j] = 1
    return C
def A_of(n, tanks, edges, kv): return TD.build(n, tanks, edges, kv)
def A1_of(n, tanks, edges, u):
    return (TD.build(n, tanks, edges, K0 * (1 + 1e-6 * u)) - TD.build(n, tanks, edges, K0 * (1 - 1e-6 * u))) / 2e-6
def unobs_dim(A, C, n, T): return n - np.linalg.matrix_rank(Mof(A, C, T), tol=1e-9)
def sym_part(u, edges, auts):
    idx = {frozenset(e): i for i, e in enumerate(edges)}; out = np.zeros_like(u)
    for m in auts:
        for i, (a, b) in enumerate(edges): out[i] += u[idx[frozenset((m[a], m[b]))]]
    return out / len(auts)


rng = np.random.default_rng(3); res = {}
for name, (n, tanks, edges) in TD.TOPOS.items():
    T = 64; C = Cof(n, tanks); A0 = A_of(n, tanks, edges, np.full(len(edges), K0)); G = autos(n, tanks, edges)
    P = sum(perm_mat(m, n) for m in G) / len(G); w, V = np.linalg.eigh(np.eye(n) - P); U = V[:, w > 0.5]; d = U.shape[1]
    orbits = n - d
    ud = unobs_dim(A0, C, n, T); inside = np.linalg.norm(Mof(A0, C, T) @ U) if d else 0.0
    u = rng.uniform(-1, 1, len(edges)); us = sym_part(u, edges, G); As = A_of(n, tanks, edges, K0 * (1 + 0.3 * us)); sym_keep = np.linalg.norm(Mof(As, C, T) @ U) if d else 0.0
    M0 = Mof(A0, C, T); Uo, sv, _ = np.linalg.svd(M0, full_matrices=False); Ur = Uo[:, sv > 1e-9]
    rows = []
    if d:
        for _ in range(4):
            u = rng.uniform(-1, 1, len(edges)); a1 = A1_of(n, tanks, edges, u)
            cj = [np.zeros((len(C), d))]
            for j in range(1, T):
                acc = np.zeros((n, d))
                for i in range(j): acc += np.linalg.matrix_power(A0, j - 1 - i) @ a1 @ np.linalg.matrix_power(A0, i) @ U
                cj.append(C @ acc)
            c1 = np.vstack(cj); pc = c1 - Ur @ (Ur.T @ c1); Gm = pc.T @ pc
            eps = 1e-3; A = A_of(n, tanks, edges, K0 * (1 + eps * u)); W = Mof(A, C, T); ev = np.linalg.eigvalsh(W.T @ W)[:d]
            th = np.linalg.eigvalsh(Gm) * eps ** 2
            us = sym_part(u, edges, G); a1a = A1_of(n, tanks, edges, u - us)
            cj2 = [np.zeros((len(C), d))]
            for j in range(1, T):
                acc = np.zeros((n, d))
                for i in range(j): acc += np.linalg.matrix_power(A0, j - 1 - i) @ a1a @ np.linalg.matrix_power(A0, i) @ U
                cj2.append(C @ acc)
            c2 = np.vstack(cj2); pc2 = c2 - Ur @ (Ur.T @ c2); Gm2 = pc2.T @ pc2
            rows.append({"eig_W": ev.tolist(), "eps2_eig_G": th.tolist(), "max_rel_dev": float(np.max(np.abs(ev / th - 1))), "G_min_eig": float(np.linalg.eigvalsh(Gm)[0]),
                         "G_sym_removed_dev": float(np.linalg.norm(Gm - Gm2) / max(1e-30, np.linalg.norm(Gm)))})
    chosen = []; cur = G; J = [j for j in range(n) if j not in tanks]
    while len(cur) > 1:
        cand = max(J, key=lambda j: len(cur) - len(autos(n, tanks, edges, tuple(chosen) + (j,))))
        chosen.append(cand); cur = autos(n, tanks, edges, tuple(chosen))
    ud_after = unobs_dim(A0, Cof(n, list(tanks) + chosen), n, T)
    res[name] = {"n": n, "aut_order": len(G), "orbits": int(orbits), "dim_V0": int(d), "unobs_dim_homog": int(ud), "V0_residual": float(inside), "sym_perturbation_residual": float(sym_keep),
                 "greedy_base": chosen, "log2_order": float(np.log2(len(G))), "unobs_dim_after_greedy": int(ud_after), "checks": rows}
    print("%-55s |G|=%2d d=%d unobs=%d  |M U|=%.1e  sym-pert resid=%.1e  base=%s (<=log2|G|=%.2f)  unobs after=%d" % (name[:55], len(G), d, ud, inside, sym_keep, chosen, np.log2(len(G)), ud_after))
    for r in rows: print("      eig(W)=%s  eps2*eig(G)=%s  dev=%.1e  Gmin=%.2e  sym-removed dev=%.1e" % (np.array2string(np.array(r["eig_W"]), precision=3), np.array2string(np.array(r["eps2_eig_G"]), precision=3), r["max_rel_dev"], r["G_min_eig"], r["G_sym_removed_dev"]))
json.dump(res, open(os.path.join("..", "results", "symmetry_propositions_check.json"), "w"), indent=1)
