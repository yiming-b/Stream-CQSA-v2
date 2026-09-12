"""
Animated schematic of Stream-CQSA: where Q/K/V, the subproblem inputs, the
partial statistics and the gradients live, and how they move, step by step,
during the forward and the backward.  ->  docs/stream_cqsa_demo.gif

    python next/docs/make_gif.py [out.gif]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = sys.argv[1] if len(sys.argv) > 1 else "stream_cqsa_demo.gif"
C = 7
QUORUM = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]
COL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]
RED, GRN, YEL, GRY = "#c0392b", "#27ae60", "#fff3b0", "#7f8c8d"

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=80)

def box(x, y, w, h, fc, text="", lw=1.2, alpha=1.0, fs=9, tc="black", z=2, ec="black", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1, weight=weight)

def arrow(x0, y0, x1, y1, color, lw=2.4, z=6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=18, color=color, lw=lw, zorder=z,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

def host_icon(x, y):
    """CPU chip with pins + two RAM sticks, at (x, y) lower-left, ~1.1 x 0.9."""
    ax.add_patch(Rectangle((x, y), 0.55, 0.55, fc="#2c3e50", ec="black", lw=1, zorder=3))
    ax.add_patch(Rectangle((x + 0.12, y + 0.12), 0.31, 0.31, fc="#95a5a6", ec="black", lw=0.6, zorder=4))
    ax.text(x + 0.275, y + 0.275, "CPU", ha="center", va="center", fontsize=6, color="black", zorder=5, weight="bold")
    for i in range(6):                                            # pins
        t = x + 0.06 + i * 0.09
        ax.add_patch(Rectangle((t, y - 0.07), 0.04, 0.07, fc="#7f8c8d", ec="none", zorder=3))
        ax.add_patch(Rectangle((t, y + 0.55), 0.04, 0.07, fc="#7f8c8d", ec="none", zorder=3))
    for j in range(2):                                            # RAM sticks
        rx = x + 0.7 + j * 0.2
        ax.add_patch(Rectangle((rx, y - 0.05), 0.13, 0.7, fc="#27ae60", ec="black", lw=0.8, zorder=3))
        for k in range(5):
            ax.add_patch(Rectangle((rx + 0.03, y + 0.02 + k * 0.13), 0.07, 0.09, fc="#1e8449", ec="none", zorder=4))
    ax.text(x + 1.25, y + 0.28, "HOST: CPU + large RAM\nQ/K/V, accumulators, gradients live here", ha="left", va="center", fontsize=8.5, color="#2c3e50", weight="bold")

def gpu_icon(x, y, active=False):
    """Graphics card with two fans and a PCIe edge, at (x, y) lower-left, ~1.3 x 0.8."""
    body = "#d35400" if active else "#34495e"
    ax.add_patch(Rectangle((x, y), 1.3, 0.62, fc=body, ec="black", lw=1, zorder=3))
    ax.add_patch(Rectangle((x + 0.05, y - 0.1), 0.8, 0.1, fc="#b8860b", ec="black", lw=0.6, zorder=3))   # PCIe edge
    for cx in (x + 0.4, x + 0.9):
        ax.add_patch(Circle((cx, y + 0.31), 0.22, fc="#ecf0f1", ec="black", lw=0.8, zorder=4))
        for a in range(4):                                        # fan blades
            import math
            ang = math.radians(a * 90 + (45 if active else 0))
            ax.add_patch(Polygon([(cx, y + 0.31), (cx + 0.19 * math.cos(ang), y + 0.31 + 0.19 * math.sin(ang)),
                                  (cx + 0.19 * math.cos(ang + 0.9), y + 0.31 + 0.19 * math.sin(ang + 0.9))], fc="#95a5a6", ec="none", zorder=5))
        ax.add_patch(Circle((cx, y + 0.31), 0.05, fc="black", zorder=6))
    ax.text(x + 1.45, y + 0.28, "DEVICE: GPU, small memory" + ("  (computing)" if active else "") + "\nonly one subproblem at a time",
            ha="left", va="center", fontsize=8.5, color="#d35400" if active else "#2c3e50", weight="bold")

# ---------------------------------------------------------------- frame plan
STAGES = ["gather", "h2d", "compute", "d2h", "merge"]
frames = [("title", None, None)]
for i in range(C):
    for st in STAGES:
        frames.append(("fwd", i, st))
frames.append(("fwd_done", None, None))
for i in range(C):
    for st in STAGES:
        frames.append(("bwd", i, st))
frames.append(("bwd_done", None, None))

STEP_TEXT = {
    "gather": "step 1/5  gather: pick the 3 chunks of this subproblem from host memory",
    "h2d": "step 2/5  H2D: copy only those chunks to the device (3N/7 tokens)",
    "compute": "step 3/5  compute: FlashAttention-2 tiles on the kept chunk pairs (masked tiles skipped)",
    "d2h": "step 4/5  D2H: copy the small partial result back to the host",
    "merge": "step 5/5  merge: fold it into the host accumulator (max-shifted, fp32)",
}
STEP_TEXT_BWD = {
    "gather": "step 1/5  gather: q, k, v, dO and the GLOBAL lse of the 3 chunks",
    "h2d": "step 2/5  H2D: copy them to the device",
    "compute": "step 3/5  compute: p = exp(s - lse); dV, dK, dQ contributions of these pairs",
    "d2h": "step 4/5  D2H: copy the partial gradients back",
    "merge": "step 5/5  scatter-add: accumulate them into the host dQ, dK, dV",
}

def draw(fi):
    ax.cla(); ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis("off")
    kind, i, st = frames[fi]
    ax.text(0.2, 7.05, "Stream-CQSA: exact attention when the whole call does not fit the device", fontsize=13, weight="bold", va="top")
    # panels
    box(0.2, 0.3, 5.2, 6.15, "#f7f7f2", lw=1.0, z=1); box(6.6, 0.3, 5.2, 6.15, "#eef4ff", lw=1.0, z=1)
    host_icon(0.45, 5.55); gpu_icon(6.85, 5.5, active=(st == "compute"))
    # Q/K/V chunks on the host
    ax.text(0.35, 4.12, "Q, K, V: 7 contiguous chunks (cyclic quorum set c=7, interest set {0,1,3})", fontsize=7.8)
    hl = set(QUORUM[i]) if (kind in ("fwd", "bwd") and st in ("gather", "h2d")) else set()
    for c in range(C):
        box(0.4 + c * 0.7, 4.35, 0.62, 0.48, COL[c], f"{c}", fs=9, tc="white", lw=2.6 if c in hl else 1.0, ec=RED if c in hl else "black")
    if kind == "title":
        ax.text(6.0, 2.4, "7 subproblems, each on 3 chunks (owner + 2 partners), run one at a time.\n"
                          "Only the in-flight subproblem's inputs and its partial result are ever on the device;\n"
                          "Q/K/V, the accumulator and the gradients stay in host memory.\n\n"
                          "Forward: partial (out_i, lse_i) merge exactly.   Backward: partial gradients add exactly.",
                ha="center", va="center", fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        ax.text(6.0, 0.8, "every kept query-key pair is computed exactly once", ha="center", fontsize=10, style="italic", color=GRY)
        return
    fwd = kind in ("fwd", "fwd_done")
    q = QUORUM[i] if i is not None else None
    done = C if kind in ("fwd_done", "bwd_done") else i + (1 if st == "merge" else 0)
    # ---- host accumulators
    if fwd:
        box(0.35, 2.85, 5.0, 1.15, "white", lw=1.2); ax.text(0.45, 3.8, "host accumulator (fp32): running (m, l, acc) per token", fontsize=8.5)
    else:
        box(0.35, 2.85, 5.0, 1.15, "white", lw=1.2); ax.text(0.45, 3.8, "host gradient accumulators (fp32): dQ, dK, dV", fontsize=8.5)
    for c in range(C):
        cover = sum(1 for j in range(done) if c in QUORUM[j])
        flash = (kind in ("fwd", "bwd") and st == "merge" and c in q)
        box(0.45 + c * 0.7, 2.95, 0.62, 0.6, COL[c], f"{cover}/3", alpha=0.25 + 0.25 * cover, fs=8.5,
            lw=2.6 if flash else 0.8, ec=GRN if flash else "black")
    ax.text(0.45, 2.68, "j/3 = how many of the chunk's 3 quorum subproblems have been folded in", fontsize=7.5, color=GRY)
    if fwd:
        ax.text(0.35, 2.2, "global lse (log-sum-exp per token, fp32) -- kept for the backward", fontsize=8.5)
        box(0.35, 1.55, 5.0, 0.5, YEL, "lse", fs=9)
    else:
        ax.text(0.35, 2.2, "also on the host: dO in 7 chunks, and the global lse from the forward", fontsize=8.5)
        for c in range(C):
            box(0.4 + c * 0.7, 1.6, 0.62, 0.45, COL[c], "dO", alpha=0.6, fs=8, lw=2.2 if c in hl else 0.8, ec=RED if c in hl else "black")
        box(0.35, 0.85, 5.0, 0.5, YEL, "lse (global)", fs=9)
    if kind in ("fwd_done", "bwd_done"):
        msg = ("Forward done.\n\noutput = acc / l -- exact: every kept pair (i, j)\nwas counted once, and the max-shifted merge\nis the arithmetic FlashAttention uses inside one kernel."
               if fwd else
               "Backward done.\n\nEach subproblem used the GLOBAL lse, so its\np = exp(s - lse) are the true attention weights;\nthe per-subproblem gradients simply add.\nNo exp(lse) is ever formed: no overflow.")
        ax.text(9.2, 2.9, msg, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        return
    # ---- device side
    ax.text(6.75, 4.12, f"{'backward, ' if not fwd else ''}subproblem {i + 1}/7: chunks {q}   (owner = {q[0]})", fontsize=9.5, weight="bold")
    on_dev = st != "gather"
    for k_, c in enumerate(q):
        box(6.85 + k_ * 0.8, 4.35, 0.7, 0.48, COL[c], f"{c}", alpha=1.0 if on_dev else 0.18, fs=9, tc="white", ec="black" if on_dev else GRY)
    ax.text(9.35, 4.59, "q_i, k_i, v_i", fontsize=8.5, va="center")
    if not fwd:
        box(9.3, 3.62, 2.2, 0.28, YEL, "dO_i, lse_i (global)", fs=7.5, alpha=1.0 if on_dev else 0.25)
    if st == "h2d":
        for k_, c in enumerate(q):
            arrow(0.71 + c * 0.7, 4.85, 7.2 + k_ * 0.8, 4.85, COL[c], lw=2.6, rad=-0.18)
        if not fwd:
            arrow(5.4, 1.85, 9.3, 3.76, RED, lw=2.4, style="-|>", ls="--", rad=-0.3)
            ax.text(5.35, 1.55, "dO chunks + lse", ha="left", fontsize=8, color=RED)
    # kernel box
    busy = st == "compute"
    box(6.85, 2.4, 4.7, 1.05, "#d5f5e3" if busy else "#eeeeee", lw=2.4 if busy else 1.2, ec=GRN if busy else "black")
    ax.text(9.2, 3.17, "CQS kernel: FlashAttention-2 + pair mask" if fwd else "CQS backward kernel (global-lse form)", ha="center", fontsize=9, weight="bold")
    ax.text(9.2, 2.72, ("tiles of chunk pairs: owner x kept partners;\nnon-owner diagonal tiles are skipped" if fwd
                        else "p = exp(s - lse);  dV += p^T dO;  dS = p (dP - delta);\ndK += dS^T q;  dQ += dS k"), ha="center", fontsize=7.8)
    if busy:
        for k_ in range(3):
            arrow(7.2 + k_ * 0.8, 4.33, 7.8 + k_ * 0.6, 3.47, GRY, lw=1.2)
    # partial result
    if st in ("compute", "d2h", "merge"):
        box(6.85, 1.3, 4.7, 0.62, YEL, "partial: out_i (fp32), lse_i" if fwd else "partial: dq_i, dk_i, dv_i", fs=9,
            lw=2.4 if st == "d2h" else 1.0, ec=RED if st == "d2h" else "black")
        if busy:
            arrow(9.2, 2.38, 9.2, 1.95, GRY, lw=1.2)
    if st == "d2h":
        arrow(6.8, 1.6, 5.45, 3.2, RED, lw=3.0, rad=0.2)
    if st == "merge":
        arrow(5.6, 3.25, 5.1, 3.3, GRN, lw=3.0)
        ax.text(5.95, 2.75, "merged" if fwd else "added", ha="center", fontsize=9, color=GRN, weight="bold")
    # step banner
    txt = (STEP_TEXT if fwd else STEP_TEXT_BWD)[st]
    box(0.2, 0.0, 11.6, 0.26, "#2c3e50", txt, fs=9.5, tc="white", lw=0.5)

anim = FuncAnimation(fig, draw, frames=len(frames))
anim.save(OUT, writer=PillowWriter(fps=0.8))
print("wrote", OUT, len(frames), "frames")
