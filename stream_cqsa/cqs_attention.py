from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .cqs_mask import CQS_mask


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    denom = np.sum(e, axis=1, keepdims=True)
    return e / denom


def _group_runs_to_mask(local_size: int, group_runs: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
    """
    Build local boolean mask from group-runs.
    True means masked.
    """
    mask = np.zeros((local_size, local_size), dtype=bool)
    for runs in group_runs:
        parts = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                parts.append(np.arange(si, ei, dtype=np.int64))
        if len(parts) == 0:
            continue
        idx = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if idx.size > 0:
            mask[np.ix_(idx, idx)] = True
    return mask


def cqsa_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_itr: int,
    interest_set: Sequence[int] = (0, 1, 3),
    c: int = 7,
    *,
    cqs_mask_engine: CQS_mask | None = None,
    return_aux: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cyclic Quorum Set Attention (CQSA) forward pass.

    Inputs:
    - Q, K, V: shape [N, d]
    - num_itr: number of CQS divide iterations
    - interest_set, c: CQS mask parameters

    Algorithm:
    1) num_itr=0 means no CQS divide (single full grid), num_itr=1 means one divide, etc.
    2) generate all CQS subgrids at that iteration (c^num_itr paths)
    3) for each path:
       R_i = Qi @ Ki.T
       P_i = exp(R_i)
       P_i[M_i] = 0
       Den_i = row_sum(P_i)
       Num_i = P_i @ Vi
    4) accumulate Num_i and Den_i by global token ids
    5) normalize final O row-wise by Den
    """
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be rank-2 matrices [N, d].")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q, K, V must have identical shape. Got {Q.shape}, {K.shape}, {V.shape}.")
    if num_itr < 0:
        raise ValueError(f"num_itr must be >= 0, got {num_itr}")

    N, d = Q.shape
    work_dtype = np.float64
    Qw = np.asarray(Q, dtype=work_dtype)
    Kw = np.asarray(K, dtype=work_dtype)
    Vw = np.asarray(V, dtype=work_dtype)

    depth = int(num_itr)
    cqs = cqs_mask_engine or CQS_mask(interest_set=interest_set, c=c)
    masks = cqs.gen_mask(
        N=N,
        num_itr=depth,
        quorum_idx=None,
        interest_set=interest_set,
        c=c,
        include_trace=False,
    )
    if masks.get("mode") != "all":
        raise RuntimeError("Expected mode='all' from gen_mask when quorum_idx is None.")

    Num_global = np.zeros((N, d), dtype=work_dtype)
    Den_global = np.zeros((N,), dtype=work_dtype)

    for path, mask_spec in masks["masks"].items():
        token_ids = np.asarray(mask_spec["token_ids"], dtype=np.int64)
        local_size = int(mask_spec["local_size"])
        if local_size == 0 or token_ids.size == 0:
            continue

        Qi = Qw[token_ids]
        Ki = Kw[token_ids]
        Vi = Vw[token_ids]

        R_i = Qi @ Ki.T
        P_i = np.exp(R_i)
        M_i = _group_runs_to_mask(local_size, mask_spec["group_runs"])
        P_i[M_i] = 0.0

        Den_i = np.sum(P_i, axis=1)
        Num_i = P_i @ Vi

        np.add.at(Num_global, token_ids, Num_i)
        np.add.at(Den_global, token_ids, Den_i)

    eps = 1e-12
    O = np.zeros_like(Num_global)
    nz = Den_global > eps
    O[nz] = Num_global[nz] / Den_global[nz, None]

    if return_aux:
        return O, {
            "Den": Den_global,
            "S": Den_global,  # backward-compatible alias
            "depth": int(depth),
            "num_itr": int(depth),
            "num_subgrids": int(masks["num_masks"]),
            "zero_rows": np.where(~nz)[0].tolist(),
        }
    return O


def full_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be rank-2 matrices [N, d].")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q, K, V must have identical shape. Got {Q.shape}, {K.shape}, {V.shape}.")
    Qw = np.asarray(Q, dtype=np.float64)
    Kw = np.asarray(K, dtype=np.float64)
    Vw = np.asarray(V, dtype=np.float64)
    P = _softmax_rows(Qw @ Kw.T)
    return P @ Vw


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d = 49, 16
    Q = rng.standard_normal((N, d))
    K = rng.standard_normal((N, d))
    V = rng.standard_normal((N, d))

    O_cqsa, aux = cqsa_attention(Q, K, V, num_itr=2, return_aux=True)
    print(
        f"N={N}, d={d}, O_shape={tuple(O_cqsa.shape)}, "
        f"num_subgrids={aux['num_subgrids']}, zero_rows={len(aux['zero_rows'])}"
    )
