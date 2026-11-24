import json, os
import numpy as np
from multiprocessing import Pool
import est_v10 as E10

if __name__ == "__main__":
    jobs = [(nm, s, i) for nm in ("true_base", "identified") for s in (777, 888, 999) for i in range(50)]
    with Pool(12) as p: res = p.map(E10.one, jobs, chunksize=3)
    out = {}
    for nm in ("true_base", "identified"):
        r = [x for x in res if x[0] == nm]
        out[nm] = {"mse": float(np.mean([a for _, a, _ in r])), "antisym": float(np.mean([b for _, _, b in r])), "n": len(r)}
    print(json.dumps(out)); json.dump(out, open(os.path.join("..", "results", "est_v13_ref.json"), "w"), indent=1)
