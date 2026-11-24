import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import sensor_only_identification as E
import sensor_only_wrong_structure as A


def job(args):
    name, seed, start, n_fit = args
    t0 = time.time(); st = A.Struct(name)
    rng = np.random.default_rng(900 + seed + 1000 * start)
    theta0 = E.TRUE * np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=5))
    train = E.train_set(100 + seed, n_fit)

    def nll(lt):
        M = A.Model(np.exp(lt), st); return -sum(A.run_imm(M, o, m)[1] for o, m in train)
    r = minimize(nll, np.log(theta0), method="Nelder-Mead", options={"xatol": 2e-3, "fatol": 1e-2, "maxiter": 500})
    th = np.exp(r.x); M = A.Model(th, st); se = an = 0.0; n = 0
    for o, m, x in E.test_set():
        e = A.run_imm(M, o, m)[0] - x; se += float((e ** 2).mean()); an += float(((e @ E.V) ** 2).mean()); n += 1
    res = {"structure": name, "seed": seed, "start": start, "theta0": theta0.tolist(), "theta": th.tolist(), "rel_err_to_true": (th / E.TRUE - 1).tolist(),
           "nll_fitted": float(r.fun), "nll_at_true_params": float(nll(np.log(E.TRUE))), "nfev": int(r.nfev), "mse": se / n, "antisym": an / n, "seconds": time.time() - t0}
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for k, v in res.items() if k != "theta0"}), flush=True)
    return res


if __name__ == "__main__":
    n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    specs = [(nm, s, j, n_fit) for j in range(n_starts) for nm in ("true", "no_spike", "spike_x2", "wrong_rates", "calm_only") for s in range(3)]
    with Pool(nproc) as p:
        res = p.map(job, specs, chunksize=1)
    json.dump({"n_fit": n_fit, "n_starts": n_starts, "runs": res}, open(os.path.join("..", "results", "sensor_only_multistart.json"), "w"), indent=1)
    print("done")
