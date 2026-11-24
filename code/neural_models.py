import torch
import torch.nn as nn
from benchmark import STATE_DIM, state_adjacency_matrix
import numpy as np, torch, torch.nn as nn
import benchmark as D
import torch, torch.nn as nn, torch.nn.functional as F
from benchmark import STATE_DIM
import sys

def _normalized_adjacency(adj=None):
    A = torch.tensor(state_adjacency_matrix(add_self_loops=True)) if adj is None else torch.as_tensor(adj)
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, adj_norm):
        super().__init__()
        self.register_buffer('A', adj_norm)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        h = torch.einsum('ij,bjf->bif', self.A, h)
        return self.lin(h)

class GNNEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24, adj=None, state_dim=None):
        super().__init__()
        A = _normalized_adjacency(adj)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM
        self.node_embed = nn.Linear(1, hidden)
        self.gc1 = GraphConv(hidden, hidden, A)
        self.gc2 = GraphConv(hidden, hidden, A)
        self.readout = nn.Linear(hidden * self.state_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        h = self.act(self.gc1(h))
        h = self.act(self.gc2(h))
        return self.readout(h.reshape(h.size(0), -1))

class PointNetEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24):
        super().__init__()
        self.node_embed = nn.Linear(1, hidden)
        self.point_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        h = self.point_mlp(h)
        pooled, _ = h.max(dim=1)
        return self.readout(pooled)

class PointNetPlusPlusEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24, adj=None, state_dim=None):
        super().__init__()
        if adj is None:
            adj = torch.tensor(state_adjacency_matrix(add_self_loops=True))
        else:
            adj = torch.as_tensor(adj)
        self.register_buffer('group_mask', adj > 0)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM
        self.node_embed = nn.Linear(1, hidden)
        self.local_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.global_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, obs_t):
        batch = obs_t.size(0)
        h = self.node_embed(obs_t.unsqueeze(-1))
        h_local = self.local_mlp(h)
        n = h_local.size(1)
        expanded = h_local.unsqueeze(1).expand(batch, n, n, -1)
        mask = self.group_mask.unsqueeze(0).unsqueeze(-1)
        masked = expanded.masked_fill(~mask, float('-inf'))
        local_pooled, _ = masked.max(dim=2)
        h_global = self.global_mlp(local_pooled)
        pooled, _ = h_global.max(dim=1)
        return self.readout(pooled)

class StabilityObserverGNNLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, adj=None, state_dim=None):
        super().__init__()
        self.encoder = GNNEncoder(out_dim=spatial_dim, adj=adj, state_dim=state_dim)
        self.state_dim = self.encoder.state_dim
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return (x_hat, hidden_seq)

class StabilityObserverPointNetLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, state_dim=None):
        super().__init__()
        self.encoder = PointNetEncoder(out_dim=spatial_dim)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return (x_hat, hidden_seq)

class StabilityObserverPointNetPPLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, adj=None, state_dim=None):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(out_dim=spatial_dim, adj=adj, state_dim=state_dim)
        self.state_dim = self.encoder.state_dim
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return (x_hat, hidden_seq)
ARCHITECTURES = {'gnn': StabilityObserverGNNLSTM, 'pointnet': StabilityObserverPointNetLSTM, 'pointnetpp': StabilityObserverPointNetPPLSTM}

class DiagGNNEncoder(GNNEncoder):

    def __init__(self, out_dim=24, adj=None, state_dim=None):
        super().__init__(out_dim=out_dim, adj=adj, state_dim=state_dim)
        n = self.state_dim
        self.w1 = nn.Parameter(torch.ones(n))
        self.w2 = nn.Parameter(torch.ones(n))

    def _layer(self, h, gc, w):
        ah = torch.einsum('ij,bjf->bif', gc.A, h)
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
C_BARRIER = 5.0

class Quadratic(nn.Module):

    def __init__(self, dim=STATE_DIM, eps=0.001):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, e):
        return ((F.softplus(self.a) + self.eps) * e.pow(2)).sum(-1)

class Neural(nn.Module):

    def __init__(self, dim=STATE_DIM, hidden=16, eps=0.001):
        super().__init__()
        self.eps = eps
        self.g = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, e):
        z = torch.zeros(1, e.shape[-1], dtype=e.dtype)
        return (self.g(e) - self.g(z)).squeeze(-1).pow(2) + self.eps * e.pow(2).sum(-1)

class ShiftedBarrier(nn.Module):

    def __init__(self, c=C_BARRIER, eps=1e-06):
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(self, e):
        r = torch.clamp(e.pow(2).sum(-1) / self.c ** 2, max=1.0 - self.eps)
        return -torch.log(1.0 - r)

    def outside(self, e):
        return e.pow(2).sum(-1) >= self.c ** 2

class ConvexAnchored(nn.Module):

    def __init__(self, dim=STATE_DIM, hidden=16, eps=0.001):
        super().__init__()
        self.eps = eps
        self.Wx1 = nn.Linear(dim, hidden)
        self.Wx2 = nn.Linear(dim, hidden)
        self.Wz = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.wout_z = nn.Parameter(torch.randn(1, hidden) * 0.05)
        self.wout_x = nn.Linear(dim, 1, bias=False)

    def f(self, e):
        z = F.softplus(self.Wx1(e))
        z = F.softplus(z @ F.softplus(self.Wz).T + self.Wx2(e))
        return (z @ F.softplus(self.wout_z).T + self.wout_x(e)).squeeze(-1)

    def forward(self, e):
        z0 = torch.zeros(1, e.shape[-1], dtype=e.dtype, requires_grad=True)
        g0 = torch.autograd.grad(self.f(z0).sum(), z0, create_graph=True)[0]
        lin = (g0 * e).sum(-1)
        return self.f(e) - self.f(z0) - lin + self.eps * e.pow(2).sum(-1)

class Combined(nn.Module):

    def __init__(self, beta=0.5):
        super().__init__()
        self.q = Quadratic()
        self.b = ShiftedBarrier()
        self.beta = beta

    def forward(self, e):
        return self.q(e) + self.beta * self.b(e)

    def outside(self, e):
        return self.b.outside(e)
CANDIDATES = {'quadratic': Quadratic, 'neural': Neural, 'barrier': ShiftedBarrier, 'convex': ConvexAnchored, 'combined': Combined}
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'models':
        for name, cls in ARCHITECTURES.items():
            model = cls()
            dummy = torch.randn(4, 10, STATE_DIM)
            x_hat, hidden_seq = model(dummy)
            print(f'{name}: x_hat={tuple(x_hat.shape)}  hidden_seq={tuple(hidden_seq.shape)}  params={sum((p.numel() for p in model.parameters()))}')
    else:
        raise SystemExit('usage: python neural_models.py <models> [args]')
