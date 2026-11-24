import json, numpy as np, collections
rows=[json.loads(l[len("RESULT "):]) for l in open("results_v3.txt") if l.startswith("RESULT ")]
def key(r): return (r["arch"], r["cand"], r["lam"])
g=collections.defaultdict(list)
for r in rows: g[key(r)].append(r)
def ms(v):
    v=[x for x in v if x is not None]
    if not v: return "n/a"
    return f"{np.mean(v):.3f} ± {np.std(v,ddof=1) if len(v)>1 else 0:.3f}"
print("N runs", len(rows))
print("\n== Candidates (GNN) ==")
print("cand | lam | seeds | MSE | p95 | p99 | rec med | rec p90 | 1-step viol | zero-viol traj | domain exit")
for cand in ["quadratic","neural","barrier","convex","combined"]:
    for lam in [0.0,0.5]:
        rs=g.get(("gnn",cand,lam),[])
        if not rs: continue
        print(f"{cand} | {lam} | {len(rs)} | {ms([r['mse'] for r in rs])} | {ms([r['tail_p95'] for r in rs])} | {ms([r['tail_p99'] for r in rs])} | {ms([r['recovery_median'] for r in rs])} | {ms([r['recovery_p90'] for r in rs])} | {ms([r['onestep_violation_rate'] for r in rs])} | {ms([r['traj_zero_violation_rate'] for r in rs])} | {ms([r['domain_exit_rate'] for r in rs])}")
print("\n== Architectures (lam=0, no candidate) ==")
print("arch | seeds | MSE | p95 | p99 | rec med | rec p90")
for a in ["gnn","gnndeep","randgraph","pointnet","pointnetpp"]:
    rs=g.get((a,"none",0.0),[])
    if not rs: continue
    print(f"{a} | {len(rs)} | {ms([r['mse'] for r in rs])} | {ms([r['tail_p95'] for r in rs])} | {ms([r['tail_p99'] for r in rs])} | {ms([r['recovery_median'] for r in rs])} | {ms([r['recovery_p90'] for r in rs])}")
print("\n== Paired per-seed MSE differences vs gnn ==")
base={r["seed"]:r["mse"] for r in g.get(("gnn","none",0.0),[])}
for a in ["gnndeep","randgraph","pointnet","pointnetpp"]:
    d=[]
    for r in g.get((a,"none",0.0),[]):
        if r["seed"] in base: d.append(r["mse"]-base[r["seed"]])
    if d: print(f"{a} - gnn: mean {np.mean(d):+.4f}, sd {np.std(d,ddof=1):.4f}, n={len(d)}, sign-consistent={'yes' if all(x>0 for x in d) or all(x<0 for x in d) else 'no'}")
