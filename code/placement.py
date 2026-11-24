import itertools, json, os
import numpy as np
from scipy.linalg import solve_discrete_are
import benchmark as TD
import itertools, json, os, sys
from multiprocessing import Pool
import epanet_simulator as H
import sys
SV2, T, P0 = (TD.SV2, 64, 0.25)
QD = (TD.DT * 0.05) ** 2

def gram(A, C):
    W = np.zeros((A.shape[0],) * 2)
    Ak = np.eye(A.shape[0])
    for _ in range(T):
        W += Ak.T @ C.T @ C @ Ak
        Ak = A @ Ak
    return W

def Cmat(n, nodes):
    C = np.zeros((len(nodes), n))
    for i, j in enumerate(nodes):
        C[i, j] = 1.0
    return C

def kf_mse(A, n, nodes, tanks):
    Q = np.diag([QD if j not in tanks else 1e-10 for j in range(n)])
    C = Cmat(n, list(tanks) + list(nodes))
    R = SV2 * np.eye(len(C))
    P = solve_discrete_are(A.T, C.T, Q, R)
    Pf = P - P @ C.T @ np.linalg.solve(C @ P @ C.T + R, C @ P)
    J = [j for j in range(n) if j not in tanks]
    return float(np.mean(np.diag(Pf)[J]))

def automorphisms(n, tanks, edges):
    J = [j for j in range(n) if j not in tanks]
    es = {frozenset(e) for e in edges}
    out = []
    for p in itertools.permutations(J):
        m = {**{t: t for t in tanks}, **dict(zip(J, p))}
        if {frozenset((m[a], m[b])) for a, b in edges} == es:
            out.append(m)
    return out

def run(b, design_on):
    res = {}
    for name, (n, tanks, edges) in TD.TOPOS.items():
        J = [j for j in range(n) if j not in tanks]
        sets = list(itertools.combinations(J, b))
        auts = automorphisms(n, tanks, edges)
        moved = [j for j in J if any((m[j] != j for m in auts))]
        sym_sets = [s for s in sets if all((j in moved for j in s))] or sets
        rng = np.random.default_rng(2024)
        acc = {k: [] for k in ('random', 'gram_E', 'gram_trace', 'symmetry', 'bayes', 'oracle')}
        for _ in range(200):
            k = rng.uniform(0.25, 0.45, len(edges))
            A = TD.build(n, tanks, edges, k)
            Ad = A if design_on == 'true' else TD.build(n, tanks, edges, np.full(len(edges), TD.K0))
            ev = {s: kf_mse(A, n, s, tanks) for s in sets}
            cr = {}
            for s in sets:
                C = Cmat(n, list(tanks) + list(s))
                W = gram(Ad, C)
                cr[s] = (np.linalg.eigvalsh(W)[0], np.trace(W), np.trace(np.linalg.inv(np.eye(n) / P0 + W / SV2)))
            pick = lambda f: min(sets, key=f)

            def tied(key, sign):
                vals = {s: round(sign * key(s), 9) for s in sets}
                best = min(vals.values())
                return float(np.mean([ev[s] for s in sets if vals[s] == best]))
            acc['random'].append(float(np.mean(list(ev.values()))))
            acc['gram_E'].append(tied(lambda s: cr[s][0], -1))
            acc['gram_trace'].append(tied(lambda s: cr[s][1], -1))
            acc['symmetry'].append(float(np.mean([ev[s] for s in sym_sets])))
            acc['bayes'].append(tied(lambda s: cr[s][2], 1))
            acc['oracle'].append(min(ev.values()))
        o = np.array(acc['oracle'])
        res[name] = {'n_symmetries': len(auts) - 1, **{k: {'mean_mse': float(np.mean(v)), 'mean_ratio_to_oracle': float(np.mean(np.array(v) / o))} for k, v in acc.items()}}
    return res

def kf_cov(A, n, nodes, tanks):
    Q = np.diag([QD if j not in tanks else 1e-10 for j in range(n)])
    C = Cmat(n, list(tanks) + list(nodes))
    R = SV2 * np.eye(len(C))
    P = solve_discrete_are(A.T, C.T, Q, R)
    return P - P @ C.T @ np.linalg.solve(C @ P @ C.T + R, C @ P)

def run_weak_direction(b, design_on):
    res = {}
    for name, (n, tanks, edges) in TD.TOPOS.items():
        J = [j for j in range(n) if j not in tanks]
        sets = list(itertools.combinations(J, b))
        A0 = TD.build(n, tanks, edges, np.full(len(edges), TD.K0))
        w, V = np.linalg.eigh(gram(A0, Cmat(n, list(tanks))))
        m = max(1, int((w < 1e-09 * w[-1]).sum()))
        U = V[:, :m]
        auts = automorphisms(n, tanks, edges)
        moved = [j for j in J if any((a[j] != j for a in auts))]
        sym_sets = [s for s in sets if all((j in moved for j in s))] or sets
        rng = np.random.default_rng(2024)
        keys = ('random', 'symmetry', 'gram_E', 'gram_trace', 'bayes', 'bayes_weak', 'oracle', 'none')
        acc = {k: [] for k in keys}
        for _ in range(200):
            k = rng.uniform(0.25, 0.45, len(edges))
            A = TD.build(n, tanks, edges, k)
            Ad = A if design_on == 'true' else A0
            met = lambda s: float(np.trace(U.T @ kf_cov(A, n, s, tanks) @ U) / m)
            ev = {s: met(s) for s in sets}
            none = met(())
            cr = {}
            for s in sets:
                W = gram(Ad, Cmat(n, list(tanks) + list(s)))
                Pp = np.linalg.inv(np.eye(n) / P0 + W / SV2)
                cr[s] = (np.linalg.eigvalsh(W)[0], np.trace(W), np.trace(Pp), np.trace(U.T @ Pp @ U))

            def tied(i, sign):
                vals = {s: round(sign * cr[s][i], 9) for s in sets}
                best = min(vals.values())
                return float(np.mean([ev[s] for s in sets if vals[s] == best]))
            acc['random'].append(float(np.mean(list(ev.values()))))
            acc['symmetry'].append(float(np.mean([ev[s] for s in sym_sets])))
            acc['gram_E'].append(tied(0, -1))
            acc['gram_trace'].append(tied(1, -1))
            acc['bayes'].append(tied(2, 1))
            acc['bayes_weak'].append(tied(3, 1))
            acc['oracle'].append(min(ev.values()))
            acc['none'].append(none)
        o = np.array(acc['oracle'])
        nn = np.array(acc['none'])
        res[name] = {'weak_dim': m, **{k: {'mean_var': float(np.mean(v)), 'ratio_to_oracle': float(np.mean(np.array(v) / o)), 'fraction_removed': float(np.mean(1 - np.array(v) / nn))} for k, v in acc.items()}}
    return res
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
W6 = 6
RULES = ('random', 'symmetry', 'gram_E', 'gram_trace', 'bayes', 'bayes_weak')

def make_config(i):
    r = np.random.default_rng(700 + i)
    het = None if i < 3 else r.uniform(0.8, 1.25, 6)
    return {'hetero': None if het is None else het.tolist(), 'build_kw': {'hw': float(r.uniform(80, 140)), 'pipe_len': float(r.uniform(400, 900))}}

def gen(cfg, seed, n):
    rng = np.random.default_rng(seed)
    O, M, X = ([], [], [])
    het = None if cfg['hetero'] is None else np.array(cfg['hetero'])
    for _ in range(n):
        o, m, x, _ = H.simulate(rng, hetero=het, build_kw=cfg['build_kw'])
        O.append(o)
        M.append(m)
        X.append(x)
    return (np.array(O), np.array(M), np.array(X))

def features(O, M, X, sensors, rng):
    n, T, _ = O.shape
    chans = [O[:, :, 0], O[:, :, 1]] + [X[:, :, 2 + j] + rng.normal(0, H.MEAS_STD, (n, T)) for j in sensors]
    masks = [M[:, :, 0].astype(float), M[:, :, 1].astype(float)]
    base = np.stack(chans + masks, axis=2)
    F = []
    for lag in range(W6):
        sh = np.zeros_like(base)
        sh[:, lag:] = base[:, :T - lag]
        F.append(sh)
    t = np.arange(T)[None, :, None] * np.ones((n, 1, 1))
    ph = 2 * np.pi * t / 48
    return np.concatenate(F + [np.sin(ph), np.cos(ph)], axis=2).reshape(n * T, -1)

def ridge_eval(tr, te, sensors, seed):
    rng = np.random.default_rng(seed)
    Ftr = features(*tr, sensors, rng)
    Fte = features(*te, sensors, rng)
    Ytr = tr[2][:, :, 2:].reshape(-1, 4)
    Yte = te[2][:, :, 2:].reshape(-1, 4)
    mu, sd = (Ftr.mean(0), Ftr.std(0) + 1e-09)
    Ftr = (Ftr - mu) / sd
    Fte = (Fte - mu) / sd
    ym = Ytr.mean(0)
    Wt = np.linalg.solve(Ftr.T @ Ftr + 1.0 * np.eye(Ftr.shape[1]), Ftr.T @ (Ytr - ym))
    E = Fte @ Wt + ym - Yte
    v4 = np.array([0, 1, 0, -1.0]) / np.sqrt(2)
    return (float((E ** 2).mean()), float(((E @ v4) ** 2).mean()))

def fit_linear(X):
    x0, x1 = (X[:, :-1].reshape(-1, 6), X[:, 1:].reshape(-1, 6))
    t = np.tile(np.arange(63), len(X))
    ph = 2 * np.pi * t / 48
    D = np.column_stack([x0, np.ones(len(x0)), np.sin(ph), np.cos(ph)])
    Bm = np.linalg.lstsq(D, x1, rcond=None)[0]
    return Bm[:6].T

def criteria(A, P0, sets, tanks, U):
    out = {}
    for s in sets:
        Wg = gram(A, Cmat(6, list(tanks) + list(s)))
        Pp = np.linalg.inv(np.linalg.inv(P0) + Wg / SV2)
        out[s] = (np.linalg.eigvalsh(Wg)[0], np.trace(Wg), np.trace(Pp), np.trace(U.T @ Pp @ U))
    return out

def job(i):
    cfg = make_config(i)
    tr = gen(cfg, 5000 + i, 1000)
    te = gen(cfg, 9000 + i, 300)
    tanks = [0, 1]
    Jn = [2, 3, 4, 5]
    n0, e0, w0 = (None, None, None)
    base = ridge_eval(tr, te, [], i)
    nom = TD.TOPOS['paper (T1-J1, T2-J3, loop J1-J2-J3-J4)']
    A_nom = TD.build(6, nom[1], nom[2], np.full(6, TD.K0))
    P_nom = 0.25 * np.eye(6)
    A_fit = fit_linear(tr[2])
    Xc = tr[2].reshape(-1, 6) - tr[2].reshape(-1, 6).mean(0)
    P_fit = np.cov(Xc.T) + 1e-06 * np.eye(6)
    res = {'config': i, 'hetero': cfg['hetero'], 'tank_only': base}
    for b in (1, 2):
        sets = list(itertools.combinations(Jn, b))
        real = {s: ridge_eval(tr, te, [j - 2 for j in s], 100 * i + b) for s in sets}
        res['b%d' % b] = {'real': {','.join(map(str, s)): v for s, v in real.items()}}
        for mname, A, P0 in (('nominal', A_nom, P_nom), ('fitted', A_fit, P_fit)):
            w, Vv = np.linalg.eigh(gram(A, Cmat(6, tanks)))
            U = Vv[:, :1]
            cr = criteria(A, P0, sets, tanks, U)
            out = {}
            moved = [3, 5]
            sym = [s for s in sets if all((j in moved for j in s))] or sets
            for rule in RULES:
                if rule == 'random':
                    chosen = sets
                elif rule == 'symmetry':
                    chosen = sym
                else:
                    idx, sign = {'gram_E': (0, -1), 'gram_trace': (1, -1), 'bayes': (2, 1), 'bayes_weak': (3, 1)}[rule]
                    vals = {s: round(sign * cr[s][idx], 9) for s in sets}
                    best = min(vals.values())
                    chosen = [s for s in sets if vals[s] == best]
                out[rule] = {'total': float(np.mean([real[s][0] for s in chosen])), 'weak': float(np.mean([real[s][1] for s in chosen])), 'sets': [','.join(map(str, s)) for s in chosen]}
            res['b%d' % b][mname] = out
    print('config', i, 'done', flush=True)
    return res
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'kalman':
        out = {}
        for b in (1, 2):
            for d in ('true', 'nominal'):
                out['b%d_%s' % (b, d)] = r = run(b, d)
                print('b=%d design on %s model' % (b, d))
                for nm, v in r.items():
                    print('  %-55s' % nm[:55], '  '.join(('%s %.3f' % (k[:6], v[k]['mean_ratio_to_oracle']) for k in ('random', 'gram_E', 'gram_trace', 'symmetry', 'bayes'))), '| oracle MSE %.4f' % v['oracle']['mean_mse'])
        json.dump(out, open(os.path.join('..', 'results', 'placement_kalman.json'), 'w'), indent=1)
    elif _cmd == 'weak_direction':
        out = {}
        for b in (1, 2):
            for d in ('true', 'nominal'):
                out['b%d_%s' % (b, d)] = r = run_weak_direction(b, d)
                print('b=%d design on %s' % (b, d))
                for nm, v in r.items():
                    print('  %-52s dim %d' % (nm[:52], v['weak_dim']), ' '.join(('%s %.2f' % (k[:8], v[k]['ratio_to_oracle']) for k in ('random', 'symmetry', 'gram_E', 'gram_trace', 'bayes', 'bayes_weak'))), '| removed(oracle) %.2f' % v['oracle']['fraction_removed'])
        json.dump(out, open(os.path.join('..', 'results', 'placement_weak_direction.json'), 'w'), indent=1)
    elif _cmd == 'hydraulic':
        n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 15
        with Pool(min(15, n_cfg)) as p:
            runs = p.map(job, range(n_cfg), chunksize=1)
        json.dump({'runs': runs}, open(os.path.join('..', 'results', 'placement_hydraulic.json'), 'w'), indent=1)
        print('done')
    else:
        raise SystemExit('usage: python placement.py <kalman|weak_direction|hydraulic> [args]')
