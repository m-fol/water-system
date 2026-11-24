import sys, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(2)
import data as D
from data import generate_batch, generate_trajectory
from models import StabilityObserverGNNLSTM
import diag_k as V2
from network import STATE_DIM

class ProbObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = StabilityObserverGNNLSTM()
        self.head_logsd = nn.Linear(32, STATE_DIM)
    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.base.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hs, _ = self.base.lstm(spatial)
        return self.base.readout(hs), self.head_logsd(hs)

def train(kind, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if kind in ("prob", "probmse"):
        net = ProbObserver()
    else:
        net = V2.make_observer(kind == "diag")
    opt = optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32); x = torch.tensor(x, dtype=torch.float32)
        if kind in ("prob", "probmse"):
            mu, logsd = net(o)
            if kind == "probmse":
                loss = nn.functional.mse_loss(mu, x)
                opt.zero_grad(); loss.backward(); opt.step(); continue
            nll = (0.5 * ((x - mu) / logsd.exp()) ** 2 + logsd).mean()
            loss = nll
        else:
            xh, _ = net(o)
            loss = nn.functional.mse_loss(xh, x)
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def evaluate(net, kind, n=1000):
    rng = np.random.default_rng(777)
    O, X = [], []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o); X.append(x)
    O = torch.tensor(np.stack(O), dtype=torch.float32); X = np.stack(X)
    with torch.no_grad():
        if kind == "prob":
            mu, logsd = net(O); mu = mu.numpy(); sd = logsd.exp().numpy()
        else:
            mu = net(O)[0].numpy(); sd = None
    E = mu - X
    out = {"mse": float((E ** 2).mean()),
           "mse_tank": float((E[..., :2] ** 2).mean()),
           "mse_junction": float((E[..., 2:] ** 2).mean()),
           "per_traj_p95": float(np.percentile((E ** 2).mean(axis=(1, 2)), 95))}
    if sd is not None:
        inside = np.abs(E) <= 1.96 * sd
        out["coverage95"] = float(inside.mean())
        out["coverage95_tank"] = float(inside[..., :2].mean())
        out["coverage95_junction"] = float(inside[..., 2:].mean())
        out["mean_sd"] = float(sd.mean())
    return out

if __name__ == "__main__":
    kind, het, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if het == "het":
        V2.set_pipe_conductance(V2.hetero_K())
    net = train(kind, seed)
    res = evaluate(net, kind)
    res.update(kind=kind, het=het, seed=seed)
    print("RESULT " + json.dumps(res), flush=True)
