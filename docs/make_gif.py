"""
Animated schematic of Stream-CQSA: where Q/K/V, the subproblem inputs, the
partial statistics and the gradients live, and how they move, during the
forward and the backward.  ->  docs/stream_cqsa_demo.gif

    python next/docs/make_gif.py [out.gif]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = sys.argv[1] if len(sys.argv) > 1 else "stream_cqsa_demo.gif"
C = 7                                   # chunks
QUORUM = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]   # cyclic (0,1,3)+i mod 7
CHUNK_COL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=80)
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, color, text="", lw=1.2, alpha=1.0, fs=9, tc="black", zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="black", lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=zorder + 1)
    return p

def arrow(x0, y0, x1, y1, color="black", lw=2.0, zorder=5, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16, color=color, lw=lw, zorder=zorder)
    ax.add_patch(a); return a

# ---------------------------------------------------------------- frame plan
# Forward: for each subproblem i: gather 3 chunks -> H2D -> kernel -> (out_i, lse_i) -> D2H -> merge into host accumulator
# Backward: for each subproblem i: gather q,k,v,dO,lse(global) -> kernel -> dq_i,dk_i,dv_i -> D2H -> scatter-add into host dQ,dK,dV
frames = []
frames.append(("title", None, 0))
for i in range(C):
    for stage in ("gather", "compute", "merge"):
        frames.append(("fwd", i, stage))
frames.append(("fwd_done", None, 0))
for i in range(C):
    for stage in ("gather", "compute", "scatter"):
        frames.append(("bwd", i, stage))
frames.append(("bwd_done", None, 0))

def draw(frame_idx):
    ax.cla(); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    kind, i, stage = frames[frame_idx]
    # --- static layout
    ax.text(0.2, 5.95, "Stream-CQSA: exact attention when the whole call does not fit the device", fontsize=13, weight="bold", va="top")
    box(0.2, 0.3, 4.6, 5.2, "#f3f3f3", lw=1.0, zorder=1); ax.text(0.35, 5.35, "HOST memory (large)", fontsize=10, weight="bold")
    box(6.2, 0.3, 4.6, 5.2, "#eef4ff", lw=1.0, zorder=1); ax.text(6.35, 5.35, "GPU memory (small)", fontsize=10, weight="bold")
    # Q/K/V as 7 chunks on the host
    ax.text(0.35, 4.95, "Q, K, V in 7 chunks (cyclic quorum set c=7, interest set {0,1,3})", fontsize=8.5)
    for c in range(C):
        box(0.35 + c * 0.62, 4.35, 0.56, 0.45, CHUNK_COL[c], f"{c}", fs=9, tc="white")
    if kind == "title":
        ax.text(5.5, 3.0, "7 subproblems, each on 3 chunks (owner + 2), run one at a time.\n"
                          "Only the in-flight subproblem's inputs and partial result\nare ever on the device; everything else stays on the host.\n\n"
                          "Forward: partial (out_i, lse_i) merge exactly.\nBackward: partial gradients add exactly.",
                ha="center", va="center", fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        return
    if kind in ("fwd", "fwd_done"):
        # host accumulator
        box(0.35, 2.55, 4.3, 0.9, "#ffffff", lw=1.2); ax.text(0.45, 3.28, "host accumulator (fp32): running (m, l, acc) per token", fontsize=8.5)
        done = C if kind == "fwd_done" else (i + (1 if stage == "merge" else 0))
        for c in range(C):
            # a token chunk is 'covered' by the subproblems that own it; show fill growing with merges
            cover = sum(1 for j in range(done) if c in QUORUM[j])
            box(0.45 + c * 0.6, 2.65, 0.54, 0.55, CHUNK_COL[c], f"{cover}/3", alpha=0.25 + 0.25 * cover, fs=8, tc="black")
        ax.text(0.45, 2.3, "'j/3': merged quorum subproblems per chunk", fontsize=7.5, color="gray")
        ax.text(0.35, 1.5, "lse (global log-sum-exp, fp32, [B,H,N]) -- kept for the backward", fontsize=8.5)
        box(0.35, 0.75, 4.3, 0.55, "#fff5cc", "lse", fs=9)
        if kind == "fwd_done":
            ax.text(8.5, 3.0, "Forward done.\n\nOutput = acc / l, exact: every kept pair (i, j)\nwas counted exactly once, and the max-shifted\nmerge of partial statistics is the same arithmetic\nFlashAttention uses inside one kernel.",
                    ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
            return
        q = QUORUM[i]
        ax.text(6.35, 4.95, f"subproblem {i + 1}/7: chunks {q}  (owner = {q[0]})", fontsize=9.5, weight="bold")
        # gathered inputs on device
        for k_, c in enumerate(q):
            a = 1.0 if stage != "gather" else 0.5
            box(6.5 + k_ * 0.75, 4.3, 0.65, 0.45, CHUNK_COL[c], f"{c}", alpha=a, fs=9, tc="white")
        ax.text(8.9, 4.5, "q_i, k_i, v_i (3N/7 tokens)", fontsize=8.5, va="center")
        if stage == "gather":
            for k_, c in enumerate(q):
                arrow(0.63 + c * 0.62, 4.35, 6.82 + k_ * 0.75, 4.75, color=CHUNK_COL[c], lw=1.6)
            ax.text(5.5, 5.55, "gather + H2D", ha="center", fontsize=9, color="#333")
        # kernel
        box(6.5, 2.55, 4.0, 1.1, "#d9ead3" if stage == "compute" else "#eeeeee", lw=1.4)
        ax.text(8.5, 3.28, "CQS kernel (FlashAttention-2 + pair mask)", ha="center", fontsize=9, weight="bold")
        ax.text(8.5, 2.85, "tiles of chunk pairs: owner x all kept,\nnon-owner diagonal tiles skipped", ha="center", fontsize=8)
        if stage == "compute":
            for k_ in range(3): arrow(6.82 + k_ * 0.75, 4.28, 7.4 + k_ * 0.5, 3.68, color="gray", lw=1.2)
        # partial result
        if stage in ("compute", "merge"):
            box(6.5, 1.3, 4.0, 0.7, "#fff5cc", "partial: out_i (fp32), lse_i", fs=9)
            if stage == "compute": arrow(8.5, 2.53, 8.5, 2.02, color="gray", lw=1.2)
        if stage == "merge":
            arrow(6.45, 1.65, 4.7, 2.9, color="#c44e52", lw=2.2)
            ax.text(5.55, 1.05, "D2H + merge\n(overlaps the next kernel)", ha="center", fontsize=9, color="#c44e52")
        return
    # ---------------- backward
    box(0.35, 2.55, 4.3, 1.2, "#ffffff", lw=1.2); ax.text(0.45, 3.58, "host gradient accumulators (fp32): dQ, dK, dV", fontsize=8.5)
    done = C if kind == "bwd_done" else (i + (1 if stage == "scatter" else 0))
    for c in range(C):
        cover = sum(1 for j in range(done) if c in QUORUM[j])
        box(0.45 + c * 0.6, 2.65, 0.54, 0.7, CHUNK_COL[c], f"{cover}/3", alpha=0.25 + 0.25 * cover, fs=8)
    ax.text(0.35, 1.9, "also on the host: dO (7 chunks) and the global lse from the forward", fontsize=8.5)
    for c in range(C): box(0.35 + c * 0.62, 1.3, 0.56, 0.45, CHUNK_COL[c], "dO", alpha=0.6, fs=8)
    box(0.35, 0.6, 4.3, 0.5, "#fff5cc", "lse (global)", fs=9)
    if kind == "bwd_done":
        ax.text(8.5, 3.0, "Backward done.\n\nEach subproblem used the GLOBAL lse, so its\np = exp(s - lse) are the true attention weights;\nthe per-subproblem gradients simply add.\nNo exp(lse) is ever formed -> no overflow.",
                ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        return
    q = QUORUM[i]
    ax.text(6.35, 4.95, f"backward, subproblem {i + 1}/7: chunks {q}", fontsize=9.5, weight="bold")
    for k_, c in enumerate(q):
        a = 1.0 if stage != "gather" else 0.5
        box(6.5 + k_ * 0.75, 4.3, 0.65, 0.45, CHUNK_COL[c], f"{c}", alpha=a, fs=9, tc="white")
    ax.text(8.9, 4.5, "q_i, k_i, v_i, dO_i, lse_i", fontsize=8.5, va="center")
    if stage == "gather":
        for k_, c in enumerate(q):
            arrow(0.63 + c * 0.62, 4.35, 6.82 + k_ * 0.75, 4.75, color=CHUNK_COL[c], lw=1.6)
            arrow(0.63 + c * 0.62, 1.75, 6.82 + k_ * 0.75, 4.28, color=CHUNK_COL[c], lw=1.0, style="->")
        ax.text(5.5, 5.55, "gather + H2D", ha="center", fontsize=9, color="#333")
    box(6.5, 2.55, 4.0, 1.1, "#d9ead3" if stage == "compute" else "#eeeeee", lw=1.4)
    ax.text(8.5, 3.28, "CQS backward kernel (global-lse form)", ha="center", fontsize=9, weight="bold")
    ax.text(8.5, 2.85, "p = exp(s - lse); dV += p^T dO; dS = p (dP - delta);\ndK += dS^T q; dQ += dS k", ha="center", fontsize=7.8)
    if stage in ("compute", "scatter"):
        box(6.5, 1.3, 4.0, 0.7, "#fff5cc", "partial: dq_i, dk_i, dv_i", fs=9)
        if stage == "compute": arrow(8.5, 2.53, 8.5, 2.02, color="gray", lw=1.2)
    if stage == "scatter":
        arrow(6.45, 1.65, 4.7, 3.1, color="#c44e52", lw=2.2)
        ax.text(5.55, 1.05, "D2H + scatter-add\ninto the host accumulators", ha="center", fontsize=9, color="#c44e52")

anim = FuncAnimation(fig, draw, frames=len(frames), interval=900)
anim.save(OUT, writer=PillowWriter(fps=1.25))
print("wrote", OUT, len(frames), "frames")
