import json, os
from fractions import Fraction as F
import sympy as sp
dt, k, kp = F(1, 2), F(7, 20), F(3, 5)
edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
N = sp.zeros(6, 6)
for e, (a, b) in enumerate(edges): N[a, e] = 1; N[b, e] = -1
R = lambda x: sp.Rational(x.numerator, x.denominator)
E = sp.diag(1, 1, 0, 0, 0, 0); A0 = sp.eye(6) - R(dt) * (R(k) * N * N.T + R(kp) * E)
C = sp.Matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]); v = sp.Matrix([0, 0, 0, 1, 0, -1])
assert A0 * v == sp.Rational(13, 20) * v and C * v == sp.zeros(2, 1)
B = sp.Matrix([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]); B[3, 4] = 1; B[5, 4] = 1
lam = sp.Rational(13, 20); res = {}
def Gform(u, T):
    A1 = -R(dt) * R(k) * N * sp.diag(*u) * N.T; Ap = [sp.eye(6)]
    for _ in range(T): Ap.append(Ap[-1] * A0)
    M0 = sp.Matrix.vstack(*[C * Ap[j] for j in range(T)])
    c = sp.Matrix.vstack(*[sp.zeros(2, 1)] + [C * sum((lam ** i * Ap[j - 1 - i] * A1 * v for i in range(j)), sp.zeros(6, 1)) for j in range(1, T)])
    MB = M0 * B; Pc = c - MB * (MB.T * MB).inv() * MB.T * c; return (Pc.T * Pc)[0, 0]
for T in (6, 64):
    a = sp.Symbol("a"); f = lambda x, y: Gform([0, 0, x, y, -y, -x], T)
    g11, g22, g12 = f(1, 0), f(0, 1), (f(1, 1) - f(1, 0) - f(0, 1)) / 2
    g11, g22, g12 = g11 / 2, g22 / 2, g12 / 2
    det = g11 * g22 - g12 ** 2
    res["T%d" % T] = {"G11": float(g11), "G22": float(g22), "G12": float(g12), "det": float(det), "G11_exact": str(g11)[:60], "det_positive": bool(det > 0 and g11 > 0)}
    print("T", T, "G11 = %.6e  G22 = %.6e  G12 = %.6e  det = %.6e" % (float(g11), float(g22), float(g12), float(det)), " G11>0:", g11 > 0, " det>0:", det > 0)
json.dump(res, open(os.path.join("..", "results", "proposition1_exact_certificate.json"), "w"), indent=1)
