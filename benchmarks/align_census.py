"""
Alignment census for the classic engine's task layout: seg_align=0 vs seg_align=128 at itr=1..3.

CPU only, no CUDA, no timing. Answers, for the "ultimate A100 kernel" program, whether building
EVERY task on 128-aligned chunk boundaries (what results/ultimate/layout_ab_*.out showed is worth
1.365x -> 1.004x vs FA-2 at itr=1) is safe at itr=2 and itr=3, i.e. whether refined children can
grow past their parent (stable_stream.py:1286-1301) and how much the packed lengths change.

For every (N, c, itr, seg_align) it records:
  * task count (non-empty), empty children, min/mean/max packed length L, sum(L) and
    sum(ceil128(L)) relative to the unaligned layout, uniform-S packing efficiency;
  * a 128x128 tile census over the lower triangle (diagonal included, as layout_ab.tile_census):
      masked   : (AND_i & AND_j) != 0      every pair in the tile is CQS-masked -> the kernel skips it
      clear    : (OR_i  & OR_j)  == 0      no pair is masked -> plain FlashAttention path
      mixed    : neither                   per-element mask path (layout_ab's definition, at 128)
      straddle : the 128-row block or the 128-column block contains a chunk boundary of ANY level
                 (labels non-uniform); mixed_or_straddle is the union, the broad definition;
    plus the same census at 64x64 so the itr=1 rows can be checked against layout_ab.json;
  * theoretical pair work: live tiles x 128^2 / (N^2/2), live = not masked;
  * 128-block contiguity: how many full 128-blocks of token_ids are NOT one contiguous global run
    (segment_block_base / the in-place kernel need zero);
  * escalation emulation for itr >= 2: child L vs parent L (same layout), aligned vs unaligned at the
    same path, ceil128(child) vs parent, children equal to their parent (a refine that cannot shrink),
    and the degenerate regime L_parent < c*128 where chunk_layout hands out empty chunks.

Bits are rebuilt from build_path_state's label_history with the exact loop of
reference.group_bits_for_path and asserted equal to it for every path; at N=131072 every task is
additionally asserted equal (L, bits, seg_base) to stable_stream.build_tasks with the same seg_align.

Run (login node, single thread, ~30 s per N):
    export OMP_NUM_THREADS=1 CQSA_CPU_THREADS=1
    python benchmarks/align_census.py --N 131072      # merges into results/ultimate/align_census.json
    python benchmarks/align_census.py --N 524288
    python benchmarks/align_census.py --N 1048576
    python benchmarks/align_census.py --table-only    # reprint the table from the json
"""
import os, sys, json, time, argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CQSA_CPU_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stream_cqsa.reference import build_path_state, all_paths, group_bits_for_path, chunk_layout  # noqa: E402
from stream_cqsa.interface import cqs_block_summaries  # noqa: E402
from stream_cqsa.autoconfig import QUORUM_SETS  # noqa: E402
from stream_cqsa.stable_stream import build_tasks  # noqa: E402

OUT_DIR = os.environ.get("ALIGN_CENSUS_OUT", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ultimate")
OUT_JSON = os.path.join(OUT_DIR, "align_census.json")

TILE = 128
TILE_AB = 64                    # layout_ab.tile_census used 64x64 summaries
ALIGNS = (0, 128)               # seg_align as build_tasks takes it: 0 -> chunk_layout(align=1)
NS = (131072, 524288, 1048576)
CS = (7, 31, 73, 133)
ITRS = (1, 2, 3)
MAX_TASKS = 4096
GROW_THRESHOLD = 1.05


def key_of(N, c, itr, align):
    return f"N={N}|c={c}|itr={itr}|align={align}"


def configs_for(N):
    out = []
    for c in CS:
        for itr in ITRS:
            if itr == 3 and c not in (7, 31):
                continue
            if c ** itr > MAX_TASKS:
                continue
            out.append((N, c, itr))
    return out


# ---------------------------------------------------------------------------- per-path state
def bits_from_state(label_history, trace, path):
    """Exact replica of reference.group_bits_for_path's bit assignment (one bit per non-empty non-owner chunk per level)."""
    L = int(label_history[0].shape[0]) if label_history else 0
    bits = np.zeros((L,), dtype=np.int64)
    bit_id = 0
    for level, owner in enumerate(path):
        labels = label_history[level]
        for chunk_id in trace[level]["chunks"]:
            if int(chunk_id) == int(owner):
                continue
            idx = labels == int(chunk_id)
            if not idx.any():
                continue
            bits[idx] |= np.int64(1) << np.int64(bit_id)
            bit_id += 1
    return bits


def block_nonuniform_from_labels(label_history, L, tile):
    """nonuni[b] = True iff some level's chunk label changes inside block b (rows b*tile .. b*tile+tile-1)."""
    nb = (L + tile - 1) // tile
    nonuni = np.zeros((nb,), dtype=bool)
    if L == 0:
        return nonuni
    pad = nb * tile - L
    for labels in label_history:
        lab = labels if pad == 0 else np.concatenate([labels, np.full(pad, labels[-1], dtype=labels.dtype)])
        lab = lab.reshape(nb, tile)
        nonuni |= lab.min(axis=1) != lab.max(axis=1)
    return nonuni


def block_contiguity(ids, tile):
    """(#full blocks that are not one contiguous run of global ids, tail block contiguous?)."""
    L = int(ids.shape[0])
    nfull = L // tile
    noncontig = 0
    if nfull:
        first = ids[: nfull * tile: tile]
        last = ids[tile - 1: nfull * tile: tile]
        noncontig = int(np.count_nonzero(last - first != tile - 1))
    tail_ok = True
    if L > nfull * tile:
        tail_ok = bool(ids[L - 1] - ids[nfull * tile] == (L - nfull * tile - 1))
    return noncontig, tail_ok


def tile_census(blk_or, blk_and, nonuni, strip=1024):
    """Lower-triangular (diagonal included) tile census from the block summaries, in row strips to bound memory."""
    o = np.asarray(blk_or, dtype=np.int64)
    a = np.asarray(blk_and, dtype=np.int64)
    nb = int(o.shape[0])
    n = nb * (nb + 1) // 2
    masked = clear = mixed = straddle = mixed_or_straddle = 0
    cols = np.arange(nb)
    for i0 in range(0, nb, strip):
        i1 = min(nb, i0 + strip)
        rows = np.arange(i0, i1)
        tri = cols[None, :i1] <= rows[:, None]
        m = ((a[i0:i1, None] & a[None, :i1]) != 0) & tri
        cl = ((o[i0:i1, None] & o[None, :i1]) == 0) & tri
        mx = tri & ~m & ~cl
        st = tri & (nonuni[i0:i1, None] | nonuni[None, :i1])
        masked += int(m.sum()); clear += int(cl.sum()); mixed += int(mx.sum())
        straddle += int(st.sum()); mixed_or_straddle += int((mx | st).sum())
    return dict(n_tiles=n, masked=masked, clear=clear, mixed=mixed, live=n - masked,
                straddle=straddle, mixed_or_straddle=mixed_or_straddle, diag=nb)


def task_record(N, path, c, iset, align, verify_bits=True):
    a = int(align) if align else 1
    ids, lh, tr = build_path_state(N, path, c, iset, sorted_gather=True, align=a)
    L = int(ids.shape[0])
    rec = dict(path=list(int(x) for x in path), L=L)
    if L == 0:
        return rec, None, None
    bits = bits_from_state(lh, tr, path)
    if verify_bits:
        ids2, bits2 = group_bits_for_path(N, path, c, iset, sorted_gather=True, align=a)
        assert np.array_equal(ids, ids2) and np.array_equal(bits, bits2), f"bits mismatch at {path}"
    bo, ba = (t.numpy() for t in cqs_block_summaries(bits, TILE))
    nonuni = block_nonuniform_from_labels(lh, L, TILE)
    # bits are uniform inside a block iff every level's label is: one bit per non-owner chunk per level,
    # so two tokens in different chunks at any level differ in that level's bits.
    assert np.array_equal(nonuni, bo != ba), f"label/bit non-uniformity disagree at {path}"
    rec["census128"] = tile_census(bo, ba, nonuni)
    bo64, ba64 = (t.numpy() for t in cqs_block_summaries(bits, TILE_AB))
    rec["census64"] = tile_census(bo64, ba64, block_nonuniform_from_labels(lh, L, TILE_AB))
    rec["noncontig_blocks128"], rec["tail_block_contiguous"] = block_contiguity(ids, TILE)
    rec["L_pad128"] = ((L + TILE - 1) // TILE) * TILE
    rec["n_boundaries"] = int(sum(int(np.count_nonzero(lab[1:] != lab[:-1])) for lab in lh))
    return rec, ids, bits


# ---------------------------------------------------------------------------- one (N, c, itr, align)
def run_config(N, c, itr, align, verify_build_tasks):
    iset = QUORUM_SETS[c]
    t0 = time.time()
    recs = []
    Ls = {}
    ids_bits = {}
    for path in all_paths(itr, c):
        rec, ids, bits = task_record(N, path, c, iset, align)
        recs.append(rec)
        Ls[tuple(rec["path"])] = rec["L"]
        if verify_build_tasks and ids is not None:
            ids_bits[tuple(rec["path"])] = (ids, bits)
    nonempty = [r for r in recs if r["L"] > 0]
    n_empty = len(recs) - len(nonempty)
    L_arr = np.array([r["L"] for r in nonempty], dtype=np.int64)
    sum_L = int(L_arr.sum()); sum_pad = int(sum(r["L_pad128"] for r in nonempty))
    agg = dict(N=N, c=c, itr=itr, align=align, interest_set=list(iset), l=len(iset),
               n_paths=len(recs), n_tasks=len(nonempty), n_empty=n_empty,
               L_min=int(L_arr.min()), L_mean=float(L_arr.mean()), L_max=int(L_arr.max()),
               sum_L=sum_L, sum_L_pad128=sum_pad,
               uniform_S_efficiency=sum_L / (len(nonempty) * int(L_arr.max())),
               n_L_mod128_zero=int(np.count_nonzero(L_arr % TILE == 0)),
               noncontig_blocks128_total=int(sum(r["noncontig_blocks128"] for r in nonempty)),
               tail_blocks_noncontig=int(sum(not r["tail_block_contiguous"] for r in nonempty)),
               boundaries_per_task_mean=float(np.mean([r["n_boundaries"] for r in nonempty])),
               build_s=time.time() - t0)
    for name in ("census128", "census64"):
        tot = {}
        for r in nonempty:
            for k_, v_ in r[name].items():
                tot[k_] = tot.get(k_, 0) + v_
        n = tot["n_tiles"]
        agg[name] = dict(tot, masked_frac=tot["masked"] / n, clear_frac=tot["clear"] / n, mixed_frac=tot["mixed"] / n,
                         live_frac=tot["live"] / n, straddle_frac=tot["straddle"] / n,
                         mixed_or_straddle_frac=tot["mixed_or_straddle"] / n)
    tile_area = TILE * TILE
    mono = N * N / 2.0
    agg["work_ratio_live"] = agg["census128"]["live"] * tile_area / mono
    agg["work_ratio_clear"] = agg["census128"]["clear"] * tile_area / mono
    agg["work_ratio_mixed"] = agg["census128"]["mixed"] * tile_area / mono
    agg["work_ratio_monolithic_tiles"] = (((N + TILE - 1) // TILE) * ((N + TILE - 1) // TILE + 1) // 2) * tile_area / mono
    # chunk layout at the TOP level for reference
    sizes, _, _ = chunk_layout(N, c, int(align) if align else 1)
    agg["top_chunk_sizes_min_max"] = [min(sizes), max(sizes)]
    if verify_build_tasks:
        tasks = build_tasks(N, itr, B=1, H=8, D=64, itemsize=2, sorted_gather=True, pin=False,
                            c=c, interest_set=iset, seg_align=int(align))
        assert len(tasks) == len(nonempty), (len(tasks), len(nonempty))
        for t in tasks:
            ids, bits = ids_bits[tuple(t.path)]
            assert int(t.local_size) == ids.shape[0]
            assert np.array_equal(t.token_ids.numpy(), ids) and np.array_equal(t.group_bits.numpy(), bits)
            if align:
                sb = t.extra["seg_base"].numpy()
                assert np.array_equal(sb, ids[::TILE][: sb.shape[0]].astype(np.int32)), "seg_base mismatch"
        agg["verified_against_build_tasks"] = True
    else:
        agg["verified_against_build_tasks"] = False
    agg["bits_verified_against_group_bits_for_path"] = True
    agg["tasks"] = [dict(path=r["path"], L=r["L"], L_pad128=r.get("L_pad128", 0),
                         c128=r.get("census128"), c64=r.get("census64"),
                         noncontig=r.get("noncontig_blocks128", 0), boundaries=r.get("n_boundaries", 0))
                    for r in recs]
    return agg, Ls


# ---------------------------------------------------------------------------- escalation emulation
def escalation_stats(N, c, itr, L_al, L_un, L_al_parent, L_un_parent):
    """Children = all paths at level itr; parents = their prefixes at level itr-1. L dicts: path -> L (0 = empty)."""
    iset = QUORUM_SETS[c]
    paths = [tuple(p) for p in all_paths(itr, c)]
    grow = []          # (ratio child_al / parent_al, path)
    equal_parent = 0
    ratios_al_un = []  # child_al / child_un at the same path
    ratios_pad = []    # ceil128(child_al) / parent_al
    cross = []         # child_al / parent_UNaligned (an unaligned parent refined with alignment)
    empties_al = empties_un = 0
    for p in paths:
        par = p[:-1]
        ca, cu = L_al[p], L_un[p]
        pa, pu = L_al_parent[par], L_un_parent[par]
        if ca == 0:
            empties_al += 1
        if cu == 0:
            empties_un += 1
        if pa > 0:
            r = ca / pa
            if ca > pa:
                grow.append((r, p))
            if ca == pa:
                equal_parent += 1
            ratios_pad.append((((ca + TILE - 1) // TILE) * TILE / pa, p))
        if cu > 0 and ca > 0:
            ratios_al_un.append((ca / cu, p))
        if pu > 0:
            cross.append((ca / pu, p))
    parents = sorted(set(p[:-1] for p in paths))
    par_al = np.array([L_al_parent[q] for q in parents], dtype=np.int64)
    qb = (par_al // TILE) // c
    worst_ratio = max(ratios_al_un) if ratios_al_un else (0.0, None)
    worst_grow = max(grow) if grow else (max((L_al[p] / L_al_parent[p[:-1]] for p in paths if L_al_parent[p[:-1]] > 0), default=0.0), None)
    return dict(
        N=N, c=c, itr=itr, n_children=len(paths), n_parents=len(parents),
        child_L_min_aligned=int(min(L_al[p] for p in paths)), child_L_min_unaligned=int(min(L_un[p] for p in paths)),
        child_L_min_nonempty_aligned=int(min((L_al[p] for p in paths if L_al[p] > 0), default=0)),
        child_L_max_aligned=int(max(L_al[p] for p in paths)), child_L_max_unaligned=int(max(L_un[p] for p in paths)),
        empty_children_aligned=empties_al, empty_children_unaligned=empties_un,
        n_children_grow_past_parent_aligned=len(grow),
        worst_child_over_parent_aligned=float(worst_grow[0]), worst_child_over_parent_path=(list(worst_grow[1]) if worst_grow[1] else None),
        n_children_equal_parent_aligned=equal_parent,
        worst_ceil128_child_over_parent_aligned=float(max(ratios_pad)[0]) if ratios_pad else 0.0,
        n_ceil128_child_over_parent_gt1=int(sum(1 for r, _ in ratios_pad if r > 1.0)),
        worst_aligned_over_unaligned_same_path=float(worst_ratio[0]), worst_aligned_over_unaligned_path=(list(worst_ratio[1]) if worst_ratio[1] else None),
        n_aligned_over_unaligned_gt_threshold=int(sum(1 for r, _ in ratios_al_un if r > GROW_THRESHOLD)),
        threshold=GROW_THRESHOLD,
        worst_aligned_child_over_unaligned_parent=float(max(cross)[0]) if cross else 0.0,
        parent_L_min_aligned=int(par_al.min()), parent_blocks_per_chunk_min=int(qb.min()), parent_blocks_per_chunk_max=int(qb.max()),
        n_degenerate_parents=int(np.count_nonzero(par_al < c * TILE)),
        degenerate_rule=f"parent L < c*{TILE} = {c * TILE}: chunk_layout gives {c}-n_blocks-1 empty chunks",
        note="l=%d chunks per child; child_L <= parent_L always because a child is a union of whole parent chunks" % len(iset),
    )


def projected_depth_limits(N, c):
    """Analytic: level k children are ~ (l/c)^k N tokens; the aligned layout is degenerate at level k when the
    level-(k-1) parent has fewer than c*128 tokens, and its children exceed 1.05x the unaligned ones once
    the parent has fewer than ~20*c*128 tokens (chunk sizes then differ by a whole 128-block, 1/qb relative)."""
    l = len(QUORUM_SETS[c])
    out = {}
    Lk = float(N)
    k = 0
    first_deg = first_spread = None
    while Lk >= 1 and k < 12:
        if first_spread is None and Lk < 20 * c * TILE:
            first_spread = k + 1
        if first_deg is None and Lk < c * TILE:
            first_deg = k + 1
        Lk *= l / c
        k += 1
    out["first_itr_where_aligned_layout_is_degenerate"] = first_deg
    out["first_itr_where_aligned_child_may_exceed_1.05x"] = first_spread
    return out


# ---------------------------------------------------------------------------- table
def fmt_row(a):
    c128 = a["census128"]
    return (f"{a['N']:>8} {a['c']:>4} {a['itr']:>3} {a['align']:>5} {a['n_tasks']:>5} {a['n_empty']:>4} "
            f"{a['L_min']:>7} {a['L_mean']:>9.1f} {a['L_max']:>7} {a.get('sum_L_ratio_vs_unaligned', 1.0):>7.4f} "
            f"{a.get('sum_pad128_ratio_vs_unaligned', 1.0):>7.4f} {a['uniform_S_efficiency']:>6.3f} | "
            f"{c128['n_tiles']:>10} {c128['live_frac'] * 100:>6.2f} {c128['masked_frac'] * 100:>7.2f} {c128['mixed_frac'] * 100:>7.3f} "
            f"{c128['straddle_frac'] * 100:>8.3f} | {a['work_ratio_live']:>8.4f} {a['work_ratio_clear']:>8.4f} | "
            f"{a['noncontig_blocks128_total']:>9} {a['census64']['mixed_frac'] * 100:>7.3f}")


def print_table(res):
    aggs = sorted((v for k, v in res["configs"].items()), key=lambda a: (a["N"], a["c"], a["itr"], a["align"]))
    print("\n==================== ALIGNMENT CENSUS (classic engine tasks, sorted_gather=True, causal, 128x128 tiles, lower triangle incl. diagonal) ====================")
    print(f"{'N':>8} {'c':>4} {'itr':>3} {'align':>5} {'tasks':>5} {'empt':>4} {'L_min':>7} {'L_mean':>9} {'L_max':>7} {'sumL/un':>7} "
          f"{'pad/un':>7} {'uniS':>6} | {'tiles128':>10} {'live%':>6} {'masked%':>7} {'mixed%':>7} {'strad%':>8} | {'work_liv':>8} {'work_clr':>8} | "
          f"{'noncontig':>9} {'mix64%':>7}")
    for a in aggs:
        print(fmt_row(a))
    print("  sumL/un, pad/un: sum(L) and sum(ceil128(L)) over tasks relative to the seg_align=0 layout at the same (N,c,itr).")
    print("  uniS: sum(L)/(n_tasks*max L), the wave kernel's uniform-S packing efficiency.  work_liv/clr: live (clear+mixed) / clear tiles x 128^2 / (N^2/2).")
    print("  strad%: tiles whose row or column block contains a chunk boundary of any level (broad 'mixed'); mixed%: layout_ab's OR/AND test at 128.")
    print("  noncontig: full 128-blocks whose token ids are not one contiguous global run (the in-place/segmented kernel needs 0).")
    esc = sorted(res["escalation"].values(), key=lambda e: (e["N"], e["c"], e["itr"]))
    if esc:
        print("\n==================== ESCALATION EMULATION (children at itr vs their parents at itr-1, same layout) ====================")
        print(f"{'N':>8} {'c':>4} {'itr':>3} {'children':>8} | {'minL_al':>7} {'minL_un':>7} {'maxL_al':>7} {'maxL_un':>7} {'empty_al':>8} | "
              f"{'grow>par':>8} {'worst c/p':>9} {'eq_par':>6} {'ceil/p':>7} | {'al/un max':>9} {'#>1.05':>6} {'path':>12} | {'par_qb':>6} {'degen':>5}")
        for e in esc:
            print(f"{e['N']:>8} {e['c']:>4} {e['itr']:>3} {e['n_children']:>8} | {e['child_L_min_aligned']:>7} {e['child_L_min_unaligned']:>7} "
                  f"{e['child_L_max_aligned']:>7} {e['child_L_max_unaligned']:>7} {e['empty_children_aligned']:>8} | "
                  f"{e['n_children_grow_past_parent_aligned']:>8} {e['worst_child_over_parent_aligned']:>9.4f} {e['n_children_equal_parent_aligned']:>6} "
                  f"{e['worst_ceil128_child_over_parent_aligned']:>7.4f} | {e['worst_aligned_over_unaligned_same_path']:>9.4f} "
                  f"{e['n_aligned_over_unaligned_gt_threshold']:>6} {str(e['worst_aligned_over_unaligned_path']):>12} | "
                  f"{e['parent_blocks_per_chunk_min']:>6} {e['n_degenerate_parents']:>5}")
        print("  grow>par: aligned children with L > their aligned parent's L (worst c/p = max child/parent).  eq_par: children equal to their parent (refine is a no-op).")
        print("  ceil/p: max ceil128(child L)/parent L.  al/un: aligned child L / unaligned child L at the same path (#>1.05 counts children over the threshold).")
        print("  par_qb: min over parents of (L_parent//128)//c, the whole 128-blocks per chunk; degen: parents with L < c*128 (some chunks empty).")
    if res.get("projected"):
        print("\n-- analytic depth limits per (N, c): level-k task ~ (l/c)^k N tokens --")
        for k_, v_ in sorted(res["projected"].items()):
            print(f"  {k_}: aligned layout degenerate from itr={v_['first_itr_where_aligned_layout_is_degenerate']}, "
                  f"aligned child may exceed 1.05x unaligned from itr={v_['first_itr_where_aligned_child_may_exceed_1.05x']}")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="*", default=None, help="sequence lengths to (re)compute; default all of %s" % (NS,))
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--no-per-task", action="store_true", help="drop per-task records from the json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res.setdefault("configs", {}); res.setdefault("escalation", {}); res.setdefault("projected", {})
    res["meta"] = dict(tile=TILE, tile_ab=TILE_AB, aligns=list(ALIGNS), max_tasks=MAX_TASKS, sorted_gather=True, causal=True,
                       quorum_sets={str(c): list(QUORUM_SETS[c]) for c in CS}, torch=torch.__version__, cuda_used=False,
                       definition="lower-triangular tiles incl. diagonal; masked=(AND&AND)!=0, clear=(OR&OR)==0, mixed=rest (layout_ab.tile_census at 128); "
                                  "straddle=row or col block holds a chunk boundary of any level; live=n_tiles-masked; "
                                  "work_ratio=live*128^2/(N^2/2)")
    if args.table_only:
        print_table(res)
        return
    Ns = args.N if args.N else list(NS)
    t_all = time.time()
    for N in Ns:
        for (N_, c, itr) in configs_for(N):
            L_by_align = {}
            for align in ALIGNS:
                t0 = time.time()
                agg, Ls = run_config(N, c, itr, align, verify_build_tasks=(N == 131072 or itr == 1))
                L_by_align[align] = Ls
                res["configs"][key_of(N, c, itr, align)] = agg
                print(f"[{time.time() - t_all:6.1f}s] N={N} c={c} itr={itr} align={align}: tasks={agg['n_tasks']} empty={agg['n_empty']} "
                      f"L {agg['L_min']}..{agg['L_max']} mixed128 {agg['census128']['mixed_frac'] * 100:.3f}% straddle {agg['census128']['straddle_frac'] * 100:.3f}% "
                      f"noncontig={agg['noncontig_blocks128_total']} ({time.time() - t0:.1f}s)", flush=True)
            un = res["configs"][key_of(N, c, itr, 0)]
            for align in ALIGNS:
                a = res["configs"][key_of(N, c, itr, align)]
                a["sum_L_ratio_vs_unaligned"] = a["sum_L"] / un["sum_L"]
                a["sum_pad128_ratio_vs_unaligned"] = a["sum_L_pad128"] / un["sum_L_pad128"]
            if itr >= 2:
                par_al = {tuple(t["path"]): t["L"] for t in res["configs"][key_of(N, c, itr - 1, 128)]["tasks"]}
                par_un = {tuple(t["path"]): t["L"] for t in res["configs"][key_of(N, c, itr - 1, 0)]["tasks"]}
                e = escalation_stats(N, c, itr, L_by_align[128], L_by_align[0], par_al, par_un)
                res["escalation"][f"N={N}|c={c}|itr={itr}"] = e
                print(f"    escalation itr={itr}: grow>parent {e['n_children_grow_past_parent_aligned']} (worst {e['worst_child_over_parent_aligned']:.4f}), "
                      f"equal-parent {e['n_children_equal_parent_aligned']}, aligned/unaligned max {e['worst_aligned_over_unaligned_same_path']:.4f} "
                      f"(#>{GROW_THRESHOLD}: {e['n_aligned_over_unaligned_gt_threshold']}), min child al/un {e['child_L_min_aligned']}/{e['child_L_min_unaligned']}, "
                      f"parent qb min {e['parent_blocks_per_chunk_min']}", flush=True)
        for c in CS:
            res["projected"][f"N={N}|c={c}"] = projected_depth_limits(N, c)
        if args.no_per_task:
            for a in res["configs"].values():
                a.pop("tasks", None)
        json.dump(res, open(OUT_JSON, "w"), indent=1)
        print(f"wrote {OUT_JSON} after N={N} ({time.time() - t_all:.0f} s)", flush=True)
    print_table(res)


if __name__ == "__main__":
    main()
