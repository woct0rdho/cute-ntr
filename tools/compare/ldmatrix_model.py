"""Simulate the (N,K) K staging + ldmatrix round trip to decode the probe.

Model (from _fi_copy.copy_matrix_cpasync, operand "B" + swizzle, and
builtins.copy_tiled native-B branch):

  staging: source element (n, k) -> physical cell n*64 + (k ^ (8*(n & 7)))
  load: lane L address -> (row = n_base + (L % 16), col = k0 + 8*(L // 16))
        physical = row*64 + (col ^ (8*(row & 7)))
        register r_j of lane L = matrix j of that address, element
        (row = L//4, col_pair = 2*(L%4)) read from consecutive cells.
"""
from __future__ import annotations

N = K = 64


def staged(n, k):
    return n * 64 + (k ^ (8 * (n & 7)))


def source_at(physical):
    for n in range(N):
        for k in range(K):
            if staged(n, k) == physical:
                return n, k
    raise KeyError(physical)


def load(addressing, lane, k0=0, n_base=0):
    """Return the 4 registers (as source (n, k) pairs) for one lane."""
    if addressing == "current":
        row = n_base + (lane % 16)
        col = k0 + 8 * (lane // 16)
    elif addressing == "quad":
        # lanes 0-7 -> (n0-7, k0), 8-15 -> (n0-7, k8),
        # 16-23 -> (n8-15, k0), 24-31 -> (n8-15, k8)
        g = lane // 8
        row = n_base + (g // 2) * 8 + lane % 8
        col = k0 + (g % 2) * 8
    elif addressing == "vert":
        row = n_base + (lane % 16)
        col = k0
    else:
        raise ValueError(addressing)
    phys = row * 64 + (col ^ (8 * (row & 7)))
    # an 8x8 matrix read: 8 consecutive cells from `phys`
    regs = []
    for j in range(4):
        start = phys + 8 * j
        r = lane // 4
        c = 2 * (lane % 4)
        regs.append((source_at(start + c), source_at(start + c + 1)))
    return regs


OBSERVED = {
    "current": {
        0: [((0, 0), (0, 1)), ((0, 8), (0, 9)), ((8, 0), (8, 1)),
            ((8, 8), (8, 9))],
        1: [((16, 0), (16, 1)), ((16, 8), (16, 9)), ((24, 0), (24, 1)),
            ((24, 8), (24, 9))],
        4: [((48, 2), (48, 3)), ((48, 10), (48, 11)), ((56, 2), (56, 3)),
            ((56, 10), (56, 11))],
    }
}


def b_fragment(lane, k0=0):
    """What the m16n8k16 B operand wants for the n-atom starting at n=0."""
    n = lane // 4
    k_lo = 2 * (lane % 4)
    k_hi = k_lo + 8
    return [((n, k0 + k_lo), (n, k0 + k_lo + 1)),
            ((n, k0 + k_hi), (n, k0 + k_hi + 1))]


def main() -> int:
    print("model check against the probe (addressing='current'):")
    for lane, expected in OBSERVED["current"].items():
        got = load("current", lane)
        ok = got == expected
        print(f"  lane {lane}: {'MATCH' if ok else 'DIFF'}  model={got}")
        if not ok:
            print(f"            probe={expected}")

    print()
    print("what the B fragment needs for n-atom 0 (k0=0):")
    for lane in (0, 1, 4):
        want = b_fragment(lane)
        for addr in ("current", "quad", "vert"):
            got = load(addr, lane)
            hit = (got[0], got[1]) == (want[0], want[1])
            print(f"  lane {lane} {addr:8s} regs0,1={got[0]},{got[1]}"
                  f"  want={want[0]},{want[1]}  {'OK' if hit else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
