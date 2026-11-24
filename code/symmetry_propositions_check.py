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

rng2 = np.random.default_rng(11)
rg = {"graphs": 0, "with_unobservable": 0, "unobservable_equals_symmetry": 0, "unobservable_exceeds_symmetry": 0, "max_rel_dev": 0.0, "zero_H_cases": 0, "zero_H_max_abs_eig": 0.0}
for _ in range(400):
    nn = int(rng2.integers(6, 9))
    ed = set()
    for j in range(2, nn):
        ed.add((int(rng2.integers(2, j)) if j > 2 else 2, j) if j > 2 else None)
    ed.discard(None)
    ed = [e for e in ed if e[0] != e[1]]
    extra = int(rng2.integers(0, 4))
    for _e in range(extra):
        a, b = sorted(int(x) for x in rng2.choice(np.arange(2, nn), 2, replace=False))
        if (a, b) not in ed:
            ed.append((a, b))
    t1, t2 = (int(x) for x in rng2.choice(np.arange(2, nn), 2, replace=False))
    ed = [(0, t1), (1, t2)] + ed
    nodes_seen = {x for e in ed for x in e}
    if len(nodes_seen) < nn:
        continue
    rg["graphs"] += 1
    tk = [0, 1]; C_ = Cof(nn, tk); A0_ = A_of(nn, tk, ed, np.full(len(ed), K0))
    d_ = unobs_dim(A0_, C_, nn, 64)
    if d_ == 0:
        continue
    rg["with_unobservable"] += 1
    G_ = autos(nn, tk, ed); P_ = sum(perm_mat(m, nn) for m in G_) / len(G_)
    w_, V_ = np.linalg.eigh(np.eye(nn) - P_); symd = int((w_ > 0.5).sum())
    rg["unobservable_equals_symmetry" if d_ == symd else "unobservable_exceeds_symmetry"] += 1
    M0_ = Mof(A0_, C_, 64); Uo_, sv_, Vt_ = np.linalg.svd(M0_, full_matrices=True)
    Ur_ = Uo_[:, :int((sv_ > 1e-9).sum())]; N0_ = Vt_[int((sv_ > 1e-9).sum()):].T
    u_ = rng2.uniform(-1, 1, len(ed)); a1_ = A1_of(nn, tk, ed, u_)
    cj_ = [np.zeros((len(C_), d_))]
    for j in range(1, 64):
        acc = np.zeros((nn, d_))
        for i in range(j):
            acc += np.linalg.matrix_power(A0_, j - 1 - i) @ a1_ @ np.linalg.matrix_power(A0_, i) @ N0_
        cj_.append(C_ @ acc)
    c1_ = np.vstack(cj_); pc_ = c1_ - Ur_ @ (Ur_.T @ c1_); H_ = pc_.T @ pc_
    eps_ = 1e-3; Ae_ = A_of(nn, tk, ed, K0 * (1 + eps_ * u_)); We_ = Mof(Ae_, C_, 64); ev_ = np.linalg.eigvalsh(We_.T @ We_)[:d_]; th_ = np.linalg.eigvalsh(H_) * eps_ ** 2
    for x, y in zip(ev_, th_):
        if y > 1e-16:
            rg["max_rel_dev"] = max(rg["max_rel_dev"], abs(x / y - 1))
        else:
            rg["zero_H_cases"] += 1; rg["zero_H_max_abs_eig"] = max(rg["zero_H_max_abs_eig"], abs(float(x)))
print("random graphs:", rg)
res["random_graphs"] = rg
json.dump(res, open(os.path.join("..", "results", "symmetry_propositions_check.json"), "w"), indent=1)
