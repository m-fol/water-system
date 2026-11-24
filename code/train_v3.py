import sys, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(2)
from data import generate_batch, generate_trajectory
from models import ARCHITECTURES, GraphConv, StabilityObserverGNNLSTM, _normalized_adjacency
from network import STATE_DIM
import candidates_v3 as CV

ALPHA = 0.3
C_FLOOR = 4 * (0.5 * 0.05) ** 2
TAU = 0.5

class DeepGNNEncoder(nn.Module):
    def __init__(self, layers=4, hidden=16, out_dim=24):
        super().__init__()
        A = _normalized_adjacency()
        self.state_dim = STATE_DIM
        self.node_embed = nn.Linear(1, hidden)
        self.gcs = nn.ModuleList([GraphConv(hidden, hidden, A) for _ in range(layers)])
        self.readout = nn.Linear(hidden * STATE_DIM, out_dim)
        self.act = nn.ReLU()
    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        for gc in self.gcs:
            h = self.act(gc(h))
        return self.readout(h.reshape(h.size(0), -1))

def make_observer(arch):
    if arch == "gnn":
        return ARCHITECTURES["gnn"]()
    if arch == "pointnet":
        return ARCHITECTURES["pointnet"]()
    if arch == "pointnetpp":
        return ARCHITECTURES["pointnetpp"]()
    if arch == "gnndeep":
        obs = StabilityObserverGNNLSTM(); obs.encoder = DeepGNNEncoder(); return obs
    if arch == "randgraph":
        rng = np.random.default_rng(2024)
        A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)
        pairs = [(i, j) for i in range(STATE_DIM) for j in range(i + 1, STATE_DIM)]
        for k in rng.choice(len(pairs), 6, replace=False):
            i, j = pairs[k]; A[i, j] = A[j, i] = 1.0
        A += np.eye(STATE_DIM, dtype=np.float32)
        return StabilityObserverGNNLSTM(adj=torch.tensor(A))
    raise ValueError(arch)

def train(arch, cand, lam, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    obs = make_observer(arch)
    params = list(obs.parameters())
    lyap = CV.CANDIDATES[cand]() if cand != "none" else None
    if lyap is not None: params += list(lyap.parameters())
    opt = optim.Adam(params, lr=1e-3)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32); x = torch.tensor(x, dtype=torch.float32)
        xh, _ = obs(o)
        loss = nn.functional.mse_loss(xh, x)
        if lyap is not None and lam > 0:
            V = lyap((xh - x).reshape(-1, STATE_DIM)).reshape(B, T)
            loss = loss + lam * torch.relu(V[:, 1:] - (1 - ALPHA) * V[:, :-1] - C_FLOOR).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return obs, lyap

def evaluate(obs, lyap, n=1000):
    rng = np.random.default_rng(777)
    obs_l, x_l, ev_l = [], [], []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        obs_l.append(o); x_l.append(x); ev_l.append(ev)
    O = torch.tensor(np.stack(obs_l), dtype=torch.float32)
    X = np.stack(x_l)
    with torch.no_grad():
        xh, _ = obs(O)
    E = xh.numpy() - X
    per_traj = (E ** 2).mean(axis=(1, 2))
    out = {"mse": float(per_traj.mean()),
           "tail_p95": float(np.percentile(per_traj, 95)),
           "tail_p99": float(np.percentile(per_traj, 99))}
    rec, censored = [], 0
    for i in range(n):
        for (t0, t1, kind, node) in ev_l[i]:
            norms = np.linalg.norm(E[i, t1:], axis=1)
            idx = np.where(norms < TAU)[0]
            if len(idx): rec.append(int(idx[0]))
            else: censored += 1
    out["recovery_median"] = float(np.median(rec)) if rec else None
    out["recovery_p90"] = float(np.percentile(rec, 90)) if rec else None
    out["recovery_censored"] = censored
    out["n_events"] = int(sum(len(e) for e in ev_l))
    if lyap is not None:
        with torch.enable_grad():
            Ve = lyap(torch.tensor(E.reshape(-1, STATE_DIM), dtype=torch.float32)).reshape(n, -1).detach().numpy()
            outside = lyap.outside(torch.tensor(E.reshape(-1, STATE_DIM), dtype=torch.float32)).reshape(n, -1).numpy() if hasattr(lyap, "outside") else np.zeros((n, 64), bool)
        viol = Ve[:, 1:] > (1 - ALPHA) * Ve[:, :-1] + C_FLOOR
        out["onestep_violation_rate"] = float(viol.mean())
        out["traj_zero_violation_rate"] = float((~viol.any(axis=1)).mean())
        out["domain_exit_rate"] = float(outside.mean())
    return out

if __name__ == "__main__":
    arch, cand, lam, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    obs, lyap = train(arch, cand, lam, seed)
    res = evaluate(obs, lyap)
    res.update(arch=arch, cand=cand, lam=lam, seed=seed)
    print("RESULT " + json.dumps(res), flush=True)
