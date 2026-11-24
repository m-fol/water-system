import numpy as np

NODES = ["R1", "P1", "T1", "P3", "J1", "J2", "J3", "J4", "P4", "T2", "P2", "R2"]
NODE_INDEX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)

NODE_TYPE = {
    "R1": "reservoir", "R2": "reservoir",
    "P1": "primary_pump", "P2": "primary_pump",
    "T1": "tank", "T2": "tank",
    "P3": "booster_pump", "P4": "booster_pump",
    "J1": "junction", "J2": "junction", "J3": "junction", "J4": "junction",
}

INSTRUMENTED = {n: (NODE_TYPE[n] != "junction") for n in NODES}

EDGES = [
    ("R1", "P1"), ("P1", "T1"), ("T1", "P3"), ("P3", "J1"),
    ("R2", "P2"), ("P2", "T2"), ("T2", "P4"), ("P4", "J3"),
    ("J1", "J2"), ("J2", "J3"), ("J3", "J4"), ("J4", "J1"),
]

STATE_NODES = ["T1", "T2", "J1", "J2", "J3", "J4"]
STATE_INDEX = {n: i for i, n in enumerate(STATE_NODES)}
STATE_DIM = len(STATE_NODES)

STATE_MASK = np.array([INSTRUMENTED[n] for n in STATE_NODES], dtype=bool)


def adjacency_matrix(add_self_loops=True):
    A = np.zeros((N, N), dtype=np.float32)
    for a, b in EDGES:
        i, j = NODE_INDEX[a], NODE_INDEX[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    if add_self_loops:
        A += np.eye(N, dtype=np.float32)
    return A


def state_adjacency_matrix(add_self_loops=True):
    reduced_edges = [("T1", "J1"), ("T2", "J3"), ("J1", "J2"), ("J2", "J3"), ("J3", "J4"), ("J4", "J1")]
    A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)
    for a, b in reduced_edges:
        i, j = STATE_INDEX[a], STATE_INDEX[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    if add_self_loops:
        A += np.eye(STATE_DIM, dtype=np.float32)
    return A


def has_cycle(edges=None, nodes=None):
    edges = edges or EDGES
    nodes = nodes or NODES
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


if __name__ == "__main__":
    print(f"{N} nodes, {len(EDGES)} edges")
    print("instrumented:", sum(INSTRUMENTED.values()), "/ unmeasured:", N - sum(INSTRUMENTED.values()))
    print("contains a cycle:", has_cycle())
    print("state adjacency:\n", state_adjacency_matrix())
