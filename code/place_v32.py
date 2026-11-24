import itertools, json, os
import numpy as np
from scipy.linalg import solve_discrete_are
import topo_v10_defs as TD

SV2, T, P0 = TD.SV2, 64, 0.25
QD = (TD.DT * 0.05) ** 2


def gram(A, C):
    W = np.zeros((A.shape[0],) * 2); Ak = np.eye(A.shape[0])
    for _ in range(T): W += Ak.T @ C.T @ C @ Ak; Ak = A @ Ak
    return W


def Cmat(n, nodes):
    C = np.zeros((len(nodes), n))
    for i, j in enumerate(nodes): C[i, j] = 1.0
    return C


def kf_mse(A, n, nodes, tanks):
    Q = np.diag([QD if j not in tanks else 1e-10 for j in range(n)]); C = Cmat(n, list(tanks) + list(nodes)); R = SV2 * np.eye(len(C))
    P = solve_discrete_are(A.T, C.T, Q, R)
    Pf = P - P @ C.T @ np.linalg.solve(C @ P @ C.T + R, C @ P)
    J = [j for j in range(n) if j not in tanks]
    return float(np.mean(np.diag(Pf)[J]))


def automorphisms(n, tanks, edges):
    J = [j for j in range(n) if j not in tanks]; es = {frozenset(e) for e in edges}; out = []
    for p in itertools.permutations(J):
        m = {**{t: t for t in tanks}, **dict(zip(J, p))}
        if {frozenset((m[a], m[b])) for a, b in edges} == es: out.append(m)
    return out


def run(b, design_on):
    res = {}
    for name, (n, tanks, edges) in TD.TOPOS.items():
        J = [j for j in range(n) if j not in tanks]; sets = list(itertools.combinations(J, b))
        auts = automorphisms(n, tanks, edges)
        moved = [j for j in J if any(m[j] != j for m in auts)]
        sym_sets = [s for s in sets if all(j in moved for j in s)] or sets
        rng = np.random.default_rng(2024); acc = {k: [] for k in ("random", "gram_E", "gram_trace", "symmetry", "bayes", "oracle")}
        for _ in range(200):
            k = rng.uniform(0.25, 0.45, len(edges)); A = TD.build(n, tanks, edges, k)
            Ad = A if design_on == "true" else TD.build(n, tanks, edges, np.full(len(edges), TD.K0))
            ev = {s: kf_mse(A, n, s, tanks) for s in sets}
            cr = {}
            for s in sets:
                C = Cmat(n, list(tanks) + list(s)); W = gram(Ad, C)
                cr[s] = (np.linalg.eigvalsh(W)[0], np.trace(W), np.trace(np.linalg.inv(np.eye(n) / P0 + W / SV2)))
            pick = lambda f: min(sets, key=f)
            def tied(key, sign):
                vals = {s: round(sign * key(s), 9) for s in sets}; best = min(vals.values())
                return float(np.mean([ev[s] for s in sets if vals[s] == best]))
            acc["random"].append(float(np.mean(list(ev.values()))))
            acc["gram_E"].append(tied(lambda s: cr[s][0], -1)); acc["gram_trace"].append(tied(lambda s: cr[s][1], -1))
            acc["symmetry"].append(float(np.mean([ev[s] for s in sym_sets]))); acc["bayes"].append(tied(lambda s: cr[s][2], 1))
            acc["oracle"].append(min(ev.values()))
        o = np.array(acc["oracle"])
        res[name] = {"n_symmetries": len(auts) - 1, **{k: {"mean_mse": float(np.mean(v)), "mean_ratio_to_oracle": float(np.mean(np.array(v) / o))} for k, v in acc.items()}}
    return res


if __name__ == "__main__":
    out = {}
    for b in (1, 2):
        for d in ("true", "nominal"):
            out["b%d_%s" % (b, d)] = r = run(b, d)
            print("b=%d design on %s model" % (b, d))
            for nm, v in r.items():
                print("  %-55s" % nm[:55], "  ".join("%s %.3f" % (k[:6], v[k]["mean_ratio_to_oracle"]) for k in ("random", "gram_E", "gram_trace", "symmetry", "bayes")), "| oracle MSE %.4f" % v["oracle"]["mean_mse"])
    json.dump(out, open(os.path.join("..", "results", "place_v32.json"), "w"), indent=1)
