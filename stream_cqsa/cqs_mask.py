from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class CQS_mask:
    """
    Compact CQS mask utility with 3 public APIs:
    1) generate_cqs_mask
    2) generate_masked_token_list
    3) visualize_cqs_mask
    """

    def __init__(self, interest_set: Sequence[int] = (0, 1, 3), c: int = 7) -> None:
        self.interest_set = tuple(int(x) for x in interest_set)
        self.c = int(c)
        self._validate_interest_and_chunk(self.interest_set, self.c)

    @staticmethod
    def _validate_interest_and_chunk(interest_set: Sequence[int], c: int) -> None:
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        if len(interest_set) == 0:
            raise ValueError("interest_set must be non-empty")
        if len(set(int(x) for x in interest_set)) != len(interest_set):
            raise ValueError(f"interest_set must contain unique offsets, got {interest_set}")
        if 0 not in set(int(x) for x in interest_set):
            raise ValueError("interest_set must include 0 so each sub-grid contains owner chunk")

    @staticmethod
    def _chunk_layout(n_tokens: int, c: int) -> Tuple[List[int], List[int], List[int]]:
        """
        First r chunks have one extra token, where r = n_tokens % c.
        Example: n=9, c=7 -> sizes [2,2,1,1,1,1,1].
        """
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
        q, r = divmod(n_tokens, c)
        sizes = [q for _ in range(c)]
        for i in range(r):
            sizes[i] += 1

        starts = [0 for _ in range(c)]
        for i in range(1, c):
            starts[i] = starts[i - 1] + sizes[i - 1]
        ends = [starts[i] + sizes[i] for i in range(c)]
        return sizes, starts, ends

    @staticmethod
    def _indices_to_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
        if indices.size == 0:
            return []
        runs: List[Tuple[int, int]] = []
        start = int(indices[0])
        prev = start
        for x in indices[1:]:
            xi = int(x)
            if xi == prev + 1:
                prev = xi
                continue
            runs.append((start, prev + 1))
            start = xi
            prev = xi
        runs.append((start, prev + 1))
        return runs

    @staticmethod
    def _runs_to_indices(runs: Sequence[Tuple[int, int]]) -> np.ndarray:
        if len(runs) == 0:
            return np.zeros((0,), dtype=np.int64)
        parts = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                parts.append(np.arange(si, ei, dtype=np.int64))
        if len(parts) == 0:
            return np.zeros((0,), dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _quorum_chunks(subseq_i: int, c: int, interest_set: Sequence[int]) -> List[int]:
        return [int((subseq_i + off) % c) for off in interest_set]

    def _resolve_params(
        self,
        interest_set: Sequence[int] | None,
        c: int | None,
    ) -> Tuple[Tuple[int, ...], int]:
        i_set = self.interest_set if interest_set is None else tuple(int(x) for x in interest_set)
        n_chunk = self.c if c is None else int(c)
        self._validate_interest_and_chunk(i_set, n_chunk)
        return i_set, n_chunk

    def _build_path_state(
        self,
        N: int,
        quorum_idx: List[int],
        interest_set: Sequence[int],
        c: int,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str, Any]]]:
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        for i, q in enumerate(quorum_idx):
            if q < 0 or q >= c:
                raise ValueError(f"quorum_idx[{i}]={q} out of range [0, {c - 1}]")

        token_ids = np.arange(N, dtype=np.int64)
        label_history: List[np.ndarray] = []
        trace: List[Dict[str, Any]] = []

        for itr, q in enumerate(quorum_idx):
            cur_len = int(token_ids.shape[0])
            _, starts, ends = self._chunk_layout(cur_len, c)
            chunks = self._quorum_chunks(q, c, interest_set)

            labels_cur = np.empty((cur_len,), dtype=np.int16)
            for chunk_id in range(c):
                s, e = int(starts[chunk_id]), int(ends[chunk_id])
                if e > s:
                    labels_cur[s:e] = chunk_id

            gather_segments: List[np.ndarray] = []
            local_ranges: Dict[int, Tuple[int, int]] = {}
            offset = 0
            for chunk_id in chunks:
                s, e = int(starts[chunk_id]), int(ends[chunk_id])
                if e <= s:
                    local_ranges[chunk_id] = (offset, offset)
                    continue
                gather_segments.append(np.arange(s, e, dtype=np.int64))
                seg_len = int(e - s)
                local_ranges[chunk_id] = (offset, offset + seg_len)
                offset += seg_len

            gather_idx = (
                np.concatenate(gather_segments, axis=0) if len(gather_segments) > 0 else np.zeros((0,), dtype=np.int64)
            )

            for t in range(len(label_history)):
                label_history[t] = label_history[t][gather_idx]
            label_history.append(labels_cur[gather_idx])
            token_ids = token_ids[gather_idx]

            trace.append(
                {
                    "iteration": int(itr),
                    "subseq_i": int(q),
                    "chunks": [int(x) for x in chunks],
                    "local_ranges": local_ranges,
                    "selected_len": int(token_ids.shape[0]),
                }
            )

        return token_ids, label_history, trace

    @staticmethod
    def _local_mask_from_group_runs(local_size: int, group_runs: Sequence[Tuple[Tuple[int, int], ...]]) -> np.ndarray:
        mask = np.zeros((local_size, local_size), dtype=bool)
        for runs in sorted(set(group_runs)):
            idx = CQS_mask._runs_to_indices(runs)
            if idx.size > 0:
                mask[np.ix_(idx, idx)] = True
        return mask

    @staticmethod
    def _is_seq_like(x: Any) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def _normalize_num_itr_and_quorum(
        self,
        num_itr: int | Sequence[int],
        quorum_idx: Sequence[int] | None,
    ) -> Tuple[int, List[int] | None]:
        # Backward-compatible old style: num_itr is actually quorum_idx list.
        if self._is_seq_like(num_itr):
            if quorum_idx is not None:
                raise ValueError("If num_itr is a path/list, do not also pass quorum_idx.")
            qidx = [int(x) for x in num_itr]
            return int(len(qidx)), qidx

        n_itr = int(num_itr)
        if n_itr < 0:
            raise ValueError(f"num_itr must be >= 0, got {n_itr}")
        if quorum_idx is None:
            return n_itr, None
        qidx = [int(x) for x in quorum_idx]
        if len(qidx) != n_itr:
            raise ValueError(
                f"len(quorum_idx) must equal num_itr. "
                f"Got len(quorum_idx)={len(qidx)}, num_itr={n_itr}."
            )
        return n_itr, qidx

    def _generate_single_mask(
        self,
        N: int,
        quorum_idx: List[int],
        i_set: Sequence[int],
        n_chunk: int,
        include_trace: bool = False,
    ) -> Dict[str, Any]:
        token_ids, label_history, trace = self._build_path_state(N, quorum_idx, i_set, n_chunk)
        local_size = int(token_ids.shape[0])

        group_runs: List[Tuple[Tuple[int, int], ...]] = []
        for itr, q in enumerate(quorum_idx):
            if itr >= len(label_history):
                break
            owner = int(q)
            labels = label_history[itr]
            chunks = trace[itr]["chunks"]
            for chunk_id in chunks:
                if chunk_id == owner:
                    continue
                idx = np.nonzero(labels == chunk_id)[0]
                if idx.size == 0:
                    continue
                runs = tuple((int(s), int(e)) for s, e in self._indices_to_runs(idx))
                if len(runs) > 0:
                    group_runs.append(runs)

        unique_group_runs = sorted(set(group_runs))
        out = {
            "mode": "single",
            "N": int(N),
            "num_itr": int(len(quorum_idx)),
            "quorum_idx": [int(x) for x in quorum_idx],
            "interest_set": [int(x) for x in i_set],
            "c": int(n_chunk),
            "local_size": local_size,
            "token_ids": [int(x) for x in token_ids.tolist()],
            "group_runs": [[(int(s), int(e)) for s, e in runs] for runs in unique_group_runs],
        }
        if include_trace:
            out["trace"] = trace
        return out

    def gen_mask(
        self,
        N: int,
        num_itr: int | Sequence[int],
        quorum_idx: List[int] | None = None,
        interest_set: Sequence[int] | None = None,
        c: int | None = None,
        include_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate CQS mask(s).

        New style:
        - pass num_itr as int
        - if quorum_idx is None: generate all c^num_itr masks
        - if quorum_idx is provided: generate one mask for that path

        Backward-compatible old style:
        - pass quorum_idx list/tuple as the 2nd positional arg.
        """
        i_set, n_chunk = self._resolve_params(interest_set, c)
        n_itr, qidx = self._normalize_num_itr_and_quorum(num_itr, quorum_idx)

        if qidx is not None:
            return self._generate_single_mask(
                N=N,
                quorum_idx=qidx,
                i_set=i_set,
                n_chunk=n_chunk,
                include_trace=include_trace,
            )

        masks: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for path in product(range(n_chunk), repeat=n_itr):
            path_list = [int(x) for x in path]
            mask_one = self._generate_single_mask(
                N=N,
                quorum_idx=path_list,
                i_set=i_set,
                n_chunk=n_chunk,
                include_trace=include_trace,
            )
            # In mode='all', keep shared fields only once at top-level.
            mask_entry: Dict[str, Any] = {
                "local_size": mask_one["local_size"],
                "token_ids": mask_one["token_ids"],
                "group_runs": mask_one["group_runs"],
            }
            if include_trace and "trace" in mask_one:
                mask_entry["trace"] = mask_one["trace"]
            masks[tuple(path_list)] = mask_entry
        return {
            "mode": "all",
            "N": int(N),
            "num_itr": int(n_itr),
            "interest_set": [int(x) for x in i_set],
            "c": int(n_chunk),
            "num_masks": int(len(masks)),
            "include_trace": bool(include_trace),
            "masks": masks,
        }

    def generate_masked_token_list(
        self,
        N: int,
        num_itr: int | Sequence[int],
        quorum_idx: List[int] | None = None,
        interest_set: Sequence[int] | None = None,
        c: int | None = None,
        cqs_mask: Dict[str, Any] | None = None,
    ) -> Any:
        """
        Apply cqs_mask dict and return masked token pair list(s).

        Returns:
        - List[Tuple[int,int]] when cqs_mask is single
        - Dict[path_tuple, List[Tuple[int,int]]] when cqs_mask is all
        """
        if cqs_mask is None:
            cqs_mask = self.gen_mask(
                N=N,
                num_itr=num_itr,
                quorum_idx=quorum_idx,
                interest_set=interest_set,
                c=c,
                include_trace=False,
            )

        if cqs_mask.get("mode") == "all":
            all_pairs: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
            for path, mask_one in cqs_mask["masks"].items():
                token_ids = np.asarray(mask_one["token_ids"], dtype=np.int64)
                local_size = int(mask_one["local_size"])
                group_runs = [tuple((int(s), int(e)) for s, e in runs) for runs in mask_one["group_runs"]]
                mask_local = self._local_mask_from_group_runs(local_size, group_runs)
                rows, cols = np.where(mask_local)
                pairs = [(int(token_ids[r]), int(token_ids[c])) for r, c in zip(rows.tolist(), cols.tolist())]
                all_pairs[tuple(path)] = pairs
            return all_pairs

        token_ids = np.asarray(cqs_mask["token_ids"], dtype=np.int64)
        local_size = int(cqs_mask["local_size"])
        group_runs = [tuple((int(s), int(e)) for s, e in runs) for runs in cqs_mask["group_runs"]]
        mask_local = self._local_mask_from_group_runs(local_size, group_runs)

        rows, cols = np.where(mask_local)
        pairs: List[Tuple[int, int]] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            pairs.append((int(token_ids[r]), int(token_ids[c])))
        return pairs

    def validate_pair_coverage(
        self,
        N: int,
        quorum_idx: List[int],
        interest_set: Sequence[int] | None = None,
        c: int | None = None,
        *,
        max_dense_pairs: int = 120_000_000,
        max_paths: int = 2_000_000,
        sample_missing: int = 10,
    ) -> Dict[str, Any]:
        """
        Validate full pair coverage in the original N x N grid.

        Rule used (as requested):
        - let len(quorum_idx) = n
        - last iteration index = n - 1
        - total subgrids at that iteration = c^(n - 1)

        Implementation detail:
        - enumerate all quorum paths of length (n - 1)
        - generate each final subgrid's CQS mask
        - union only unmasked token pairs from that subgrid
        - compare union size with N^2
        """
        i_set, n_chunk = self._resolve_params(interest_set, c)
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        if len(quorum_idx) == 0:
            raise ValueError("quorum_idx must be non-empty for coverage validation")

        n = len(quorum_idx)
        last_itr = n - 1
        path_len = max(0, last_itr)
        num_subgrids = int(n_chunk**path_len)

        if num_subgrids > int(max_paths):
            raise ValueError(
                f"num_subgrids={num_subgrids} exceeds max_paths={max_paths}. "
                "Increase max_paths or reduce len(quorum_idx)."
            )
        total_pairs = int(N) * int(N)
        if total_pairs > int(max_dense_pairs):
            raise ValueError(
                f"N^2={total_pairs} exceeds max_dense_pairs={max_dense_pairs}. "
                "Increase max_dense_pairs or use a smaller N."
            )

        coverage = np.zeros((N, N), dtype=bool)
        for path in product(range(n_chunk), repeat=path_len):
            cqs_mask = self.gen_mask(
                N=N,
                num_itr=len(path),
                quorum_idx=list(path),
                interest_set=i_set,
                c=n_chunk,
                include_trace=False,
            )
            token_ids = np.asarray(cqs_mask["token_ids"], dtype=np.int64)
            local_size = int(cqs_mask["local_size"])
            if token_ids.size == 0 or local_size == 0:
                continue

            group_runs = [
                tuple((int(s), int(e)) for s, e in runs)
                for runs in cqs_mask["group_runs"]
            ]
            masked_local = self._local_mask_from_group_runs(local_size, group_runs)
            unmasked_rows, unmasked_cols = np.where(~masked_local)
            if unmasked_rows.size == 0:
                continue
            coverage[token_ids[unmasked_rows], token_ids[unmasked_cols]] = True

        union_size = int(coverage.sum())
        covered = bool(union_size == total_pairs)
        result: Dict[str, Any] = {
            "N": int(N),
            "n": int(n),
            "last_itr": int(last_itr),
            "c": int(n_chunk),
            "interest_set": [int(x) for x in i_set],
            "num_subgrids": int(num_subgrids),
            "union_size": int(union_size),
            "total_pairs": int(total_pairs),
            "covered": covered,
        }

        if not covered and sample_missing > 0:
            miss = np.argwhere(~coverage)
            take = int(min(sample_missing, miss.shape[0]))
            result["missing_examples"] = [
                (int(miss[i, 0]), int(miss[i, 1])) for i in range(take)
            ]
        else:
            result["missing_examples"] = []
        return result

    def visualize_cqs_mask(
        self,
        N: int,
        quorum_idx: List[int],
        interest_set: Sequence[int] | None = None,
        c: int | None = None,
        cqs_mask: Dict[str, Any] | None = None,
        *,
        axes: Any = None,
        annotate_chunks: bool = True,
        show_token_grid: bool = True,
        max_labeled_tokens: int = 64,
        show: bool = True,
    ) -> Tuple[Any, Tuple[Any, Any]]:
        """
        Plot both:
        - global N x N mask grid
        - final local subgrid mask
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
        except Exception as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for visualize_cqs_mask") from exc

        if cqs_mask is None:
            cqs_mask = self.gen_mask(
                N=N,
                num_itr=len(quorum_idx),
                quorum_idx=quorum_idx,
                interest_set=interest_set,
                c=c,
                include_trace=annotate_chunks,
            )
        if cqs_mask.get("mode") == "all":
            raise ValueError("visualize_cqs_mask expects a single-path mask, not mode='all'.")

        token_ids = np.asarray(cqs_mask["token_ids"], dtype=np.int64)
        local_size = int(cqs_mask["local_size"])
        group_runs = [tuple((int(s), int(e)) for s, e in runs) for runs in cqs_mask["group_runs"]]
        mask_local = self._local_mask_from_group_runs(local_size, group_runs)  # True=masked

        if axes is None:
            fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(13, 6))
        else:
            try:
                ax_global, ax_local = axes
            except Exception as exc:  # pragma: no cover
                raise ValueError("axes must be a 2-tuple: (ax_global, ax_local)") from exc
            fig = ax_global.figure

        def _apply_token_grid(ax: Any, size: int, tick_labels: Sequence[int] | None = None) -> None:
            # Draw per-token cell boundaries.
            ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.45, alpha=1.0)

            # Keep labels readable when size is large.
            if size <= max_labeled_tokens:
                ticks = np.arange(size)
            else:
                step = max(1, int(np.ceil(size / max_labeled_tokens)))
                ticks = np.arange(0, size, step)
            if tick_labels is None:
                labels = [str(int(i)) for i in ticks]
            else:
                labels = [str(int(tick_labels[int(i)])) for i in ticks]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)

        def _outline_global_subgrid(ax: Any, ids: np.ndarray, max_cells: int = 200_000) -> None:
            # Outline only current sub-grid cells in global view; leave all other cells outline-free.
            ids_list = [int(x) for x in ids.tolist()]
            lsz = len(ids_list)
            if lsz == 0:
                return
            total_cells = lsz * lsz
            if total_cells <= max_cells:
                patches = []
                for r in ids_list:
                    y = float(r) - 0.5
                    for c in ids_list:
                        x = float(c) - 0.5
                        patches.append(Rectangle((x, y), 1.0, 1.0))
                coll = PatchCollection(
                    patches,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=1.0,
                    zorder=3,
                )
                ax.add_collection(coll)
                return

            # Fallback for very large local grids: outline contiguous run blocks.
            ids_sorted = sorted(ids_list)
            runs = []
            rs = ids_sorted[0]
            re = rs + 1
            for t in ids_sorted[1:]:
                if t == re:
                    re += 1
                else:
                    runs.append((rs, re))
                    rs, re = t, t + 1
            runs.append((rs, re))
            for r0, r1 in runs:
                for c0, c1 in runs:
                    ax.add_patch(
                        Rectangle(
                            (float(c0) - 0.5, float(r0) - 0.5),
                            float(c1 - c0),
                            float(r1 - r0),
                            fill=False,
                            edgecolor="black",
                            linewidth=0.45,
                            alpha=1.0,
                            zorder=3,
                        )
                    )

        # Global masked view.
        global_grid = np.zeros((N, N), dtype=np.int32)
        rows, cols = np.where(mask_local)
        for r, c in zip(rows.tolist(), cols.tolist()):
            global_grid[int(token_ids[r]), int(token_ids[c])] = 1

        bw_cmap = ListedColormap(["white", "black"])
        im0 = ax_global.imshow(global_grid, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
        ax_global.set_title(f"Global Mask Grid: N={N}, quorum_idx={list(quorum_idx)}", fontsize=10)
        ax_global.set_xlabel("Key token index (global)")
        ax_global.set_ylabel("Query token index (global)")
        ax_global.set_xlim(-0.5, N - 0.5)
        ax_global.set_ylim(N - 0.5, -0.5)
        _outline_global_subgrid(ax_global, token_ids)

        if annotate_chunks:
            _, starts, ends = self._chunk_layout(N, int(cqs_mask["c"]))
            boundaries = [0] + [int(e) for e in ends]
            for b in boundaries:
                ax_global.axhline(y=b - 0.5, color="tab:blue", linewidth=0.7, alpha=0.7)
                ax_global.axvline(x=b - 0.5, color="tab:blue", linewidth=0.7, alpha=0.7)
            for i in range(int(cqs_mask["c"])):
                s, e = int(starts[i]), int(ends[i])
                if e <= s:
                    continue
                mid = (s + e - 1) / 2.0
                ax_global.text(mid, -0.9, f"c{i}", ha="center", va="center", fontsize=8, color="tab:red")
                ax_global.text(-0.9, mid, f"c{i}", ha="center", va="center", fontsize=8, color="tab:red")

        # Local final subgrid masked view.
        local_grid = mask_local.astype(np.int32)
        im1 = ax_local.imshow(local_grid, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
        ax_local.set_title(f"Final Sub-grid Mask: L={local_size}", fontsize=10)
        ax_local.set_xlabel("Key token index (local)")
        ax_local.set_ylabel("Query token index (local)")
        ax_local.set_xlim(-0.5, local_size - 0.5)
        ax_local.set_ylim(local_size - 0.5, -0.5)
        if show_token_grid:
            # Local cell index i maps to global token_ids[i].
            _apply_token_grid(ax_local, local_size, tick_labels=token_ids.tolist())

        trace = cqs_mask.get("trace", [])
        if annotate_chunks and len(trace) == 0:
            i_set = tuple(int(x) for x in cqs_mask.get("interest_set", (self.interest_set if interest_set is None else interest_set)))
            c_use = int(cqs_mask.get("c", self.c if c is None else c))
            _, _, trace = self._build_path_state(N, [int(x) for x in quorum_idx], i_set, c_use)
        if annotate_chunks and len(trace) > 0:
            last = trace[-1]
            ranges = {int(k): tuple(v) for k, v in last["local_ranges"].items()}
            boundaries = sorted(set([0] + [int(v[1]) for v in ranges.values()]))
            for b in boundaries:
                ax_local.axhline(y=b - 0.5, color="tab:blue", linewidth=0.7, alpha=0.7)
                ax_local.axvline(x=b - 0.5, color="tab:blue", linewidth=0.7, alpha=0.7)

        token_str = ",".join(str(int(x)) for x in token_ids.tolist())
        if len(token_str) > 64:
            token_str = token_str[:61] + "..."
        ax_local.text(0.0, -0.18, f"global tokens: [{token_str}]", transform=ax_local.transAxes, fontsize=8, ha="left")

        cbar0 = fig.colorbar(im0, ax=ax_global, fraction=0.046, pad=0.04)
        cbar0.set_ticks([0, 1])
        cbar0.set_ticklabels(["unmasked", "masked"])

        cbar1 = fig.colorbar(im1, ax=ax_local, fraction=0.046, pad=0.04)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(["unmasked", "masked"])

        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax_global, ax_local)
