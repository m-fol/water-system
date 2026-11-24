import torch, torch.nn as nn, torch.nn.functional as F
from network import STATE_DIM
C_BARRIER = 5.0

class Quadratic(nn.Module):
    def __init__(self, dim=STATE_DIM, eps=1e-3):
        super().__init__(); self.a = nn.Parameter(torch.zeros(dim)); self.eps = eps
    def forward(self, e):
        return ((F.softplus(self.a) + self.eps) * e.pow(2)).sum(-1)

class Neural(nn.Module):
    def __init__(self, dim=STATE_DIM, hidden=16, eps=1e-3):
        super().__init__(); self.eps = eps
        self.g = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, e):
        z = torch.zeros(1, e.shape[-1], dtype=e.dtype)
        return (self.g(e) - self.g(z)).squeeze(-1).pow(2) + self.eps * e.pow(2).sum(-1)

class ShiftedBarrier(nn.Module):
    def __init__(self, c=C_BARRIER, eps=1e-6):
        super().__init__(); self.c = c; self.eps = eps
    def forward(self, e):
        r = torch.clamp(e.pow(2).sum(-1) / self.c ** 2, max=1.0 - self.eps)
        return -torch.log(1.0 - r)
    def outside(self, e):
        return e.pow(2).sum(-1) >= self.c ** 2

class ConvexAnchored(nn.Module):
    def __init__(self, dim=STATE_DIM, hidden=16, eps=1e-3):
        super().__init__(); self.eps = eps
        self.Wx1 = nn.Linear(dim, hidden); self.Wx2 = nn.Linear(dim, hidden)
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
        super().__init__(); self.q = Quadratic(); self.b = ShiftedBarrier(); self.beta = beta
    def forward(self, e):
        return self.q(e) + self.beta * self.b(e)
    def outside(self, e):
        return self.b.outside(e)

CANDIDATES = {"quadratic": Quadratic, "neural": Neural, "barrier": ShiftedBarrier,
              "convex": ConvexAnchored, "combined": Combined}
