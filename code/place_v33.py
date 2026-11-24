import itertools, json, os
import numpy as np
from scipy.linalg import solve_discrete_are
import topo_v10_defs as TD
from place_v32 import gram, Cmat, automorphisms, SV2, T, P0, QD


def kf_cov(A, n, nodes, tanks):
    Q = np.diag([QD if j not in tanks else 1e-10 for j in range(n)]); C = Cmat(n, list(tanks) + list(nodes)); R = SV2 * np.eye(len(C))
    P = solve_discrete_are(A.T, C.T, Q, R)
    return P - P @ C.T @ np.linalg.solve(C @ P @ C.T + R, C @ P)


def run(b, design_on):
    res = {}
    for name, (n, tanks, edges) in TD.TOPOS.items():
        J = [j for j in range(n) if j not in tanks]; sets = list(itertools.combinations(J, b))
        A0 = TD.build(n, tanks, edges, np.full(len(edges), TD.K0)); w, V = np.linalg.eigh(gram(A0, Cmat(n, list(tanks))))
        m = max(1, int((w < 1e-9 * w[-1]).sum())); U = V[:, :m]
        auts = automorphisms(n, tanks, edges); moved = [j for j in J if any(a[j] != j for a in auts)]
        sym_sets = [s for s in sets if all(j in moved for j in s)] or sets
        rng = np.random.default_rng(2024); keys = ("random", "symmetry", "gram_E", "gram_trace", "bayes", "bayes_weak", "oracle", "none")
        acc = {k: [] for k in keys}
        for _ in range(200):
            k = rng.uniform(0.25, 0.45, len(edges)); A = TD.build(n, tanks, edges, k)
            Ad = A if design_on == "true" else A0
            met = lambda s: float(np.trace(U.T @ kf_cov(A, n, s, tanks) @ U) / m)
            ev = {s: met(s) for s in sets}; none = met(()); cr = {}
            for s in sets:
                W = gram(Ad, Cmat(n, list(tanks) + list(s))); Pp = np.linalg.inv(np.eye(n) / P0 + W / SV2)
                cr[s] = (np.linalg.eigvalsh(W)[0], np.trace(W), np.trace(Pp), np.trace(U.T @ Pp @ U))

            def tied(i, sign):
                vals = {s: round(sign * cr[s][i], 9) for s in sets}; best = min(vals.values())
                return float(np.mean([ev[s] for s in sets if vals[s] == best]))
            acc["random"].append(float(np.mean(list(ev.values())))); acc["symmetry"].append(float(np.mean([ev[s] for s in sym_sets])))
            acc["gram_E"].append(tied(0, -1)); acc["gram_trace"].append(tied(1, -1)); acc["bayes"].append(tied(2, 1)); acc["bayes_weak"].append(tied(3, 1))
            acc["oracle"].append(min(ev.values())); acc["none"].append(none)
        o = np.array(acc["oracle"]); nn = np.array(acc["none"])
        res[name] = {"weak_dim": m, **{k: {"mean_var": float(np.mean(v)), "ratio_to_oracle": float(np.mean(np.array(v) / o)), "fraction_removed": float(np.mean(1 - np.array(v) / nn))} for k, v in acc.items()}}
    return res


if __name__ == "__main__":
    out = {}
    for b in (1, 2):
        for d in ("true", "nominal"):
            out["b%d_%s" % (b, d)] = r = run(b, d); print("b=%d design on %s" % (b, d))
            for nm, v in r.items():
                print("  %-52s dim %d" % (nm[:52], v["weak_dim"]), " ".join("%s %.2f" % (k[:8], v[k]["ratio_to_oracle"]) for k in ("random", "symmetry", "gram_E", "gram_trace", "bayes", "bayes_weak")), "| removed(oracle) %.2f" % v["oracle"]["fraction_removed"])
    json.dump(out, open(os.path.join("..", "results", "place_v33.json"), "w"), indent=1)
