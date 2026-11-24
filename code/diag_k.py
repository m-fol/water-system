import numpy as np, torch, torch.nn as nn
import data as D
from models import GNNEncoder, StabilityObserverGNNLSTM

class DiagGNNEncoder(GNNEncoder):
    def __init__(self, out_dim=24, adj=None, state_dim=None):
        super().__init__(out_dim=out_dim, adj=adj, state_dim=state_dim)
        n = self.state_dim
        self.w1 = nn.Parameter(torch.ones(n))
        self.w2 = nn.Parameter(torch.ones(n))

    def _layer(self, h, gc, w):
        ah = torch.einsum("ij,bjf->bif", gc.A, h)
        ah = w.view(1, -1, 1) * ah
        return gc.lin(ah)

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        h = self.act(self._layer(h, self.gc1, self.w1))
        h = self.act(self._layer(h, self.gc2, self.w2))
        return self.readout(h.reshape(h.size(0), -1))

def make_observer(diag):
    obs = StabilityObserverGNNLSTM()
    if diag:
        obs.encoder = DiagGNNEncoder(out_dim=24)
    return obs

def hetero_K(seed=2024, lo=0.25, hi=0.45):
    rng = np.random.default_rng(seed)
    K = np.zeros_like(D.ADJ)
    for i, j in zip(*np.nonzero(np.triu(D.ADJ, 1))):
        K[i, j] = K[j, i] = rng.uniform(lo, hi)
    return K

def set_pipe_conductance(K):
    def flow(x, adj, nonlinear, gain):
        dh = x[None, :] - x[:, None]
        return (K * dh).sum(axis=1)
    D._flow_term = flow
