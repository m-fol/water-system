import json, os
import numpy as np

DT, KP, K0, SV2 = 0.5, 0.6, 0.35, 0.03 ** 2
TOPOS = {
    "paper (T1-J1, T2-J3, loop J1-J2-J3-J4)": (6, [0, 1], [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]),
    "tanks on adjacent junctions (T1-J1, T2-J2)": (6, [0, 1], [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 2)]),
    "both tanks on the same junction (T1-J1, T2-J1)": (6, [0, 1], [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 2)]),
    "loop plus chord J1-J3": (6, [0, 1], [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2), (2, 4)]),
    "8 nodes: 6-junction ring, tanks at opposite junctions": (8, [0, 1], [(0, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]),
    "8 nodes: 6-junction ring, tanks at adjacent junctions": (8, [0, 1], [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]),
    "8 nodes: 6-junction ring + 2 chords, tanks opposite": (8, [0, 1], [(0, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2), (3, 6), (4, 7)]),
}


def build(n, tanks, edges, k):
    L = np.zeros((n, n))
    for e, (a, b) in enumerate(edges):
        L[a, a] += k[e]; L[b, b] += k[e]; L[a, b] -= k[e]; L[b, a] -= k[e]
    E = np.zeros((n, n))
    for t in tanks: E[t, t] = 1.0
    return np.eye(n) - DT * (L + KP * E)


def obs_matrix(A, tanks, n):
    C = np.zeros((len(tanks), n))
    for i, t in enumerate(tanks): C[i, t] = 1.0
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return O, C


def gram(A, C, T):
    W = np.zeros((A.shape[0],) * 2); Ak = np.eye(A.shape[0])
    for _ in range(T):
        W += Ak.T @ C.T @ C @ Ak; Ak = A @ Ak
    return W
