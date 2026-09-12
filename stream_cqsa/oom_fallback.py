"""
Drop-in OOM recovery: run the normal attention path, fall back to Stream-CQSA.

    from stream_cqsa.oom_fallback import attention_oom_safe
    out = attention_oom_safe(q, k, v, causal=True)      # same shape/dtype as SDPA

Why this is not just `try: sdpa(...) except OutOfMemoryError: stream_cqsa(...)`
-----------------------------------------------------------------------------
Four things have to happen between the exception and the retry, and skipping any
of them makes the fallback fail too:

1. **Drain the allocator.** The failed call leaves its partial allocations in the
   caching allocator. Without `empty_cache()` the retry meets the same wall.

2. **Get the inputs off the device.** This is the one that actually matters, and
   it is not optional. Stream-CQSA *device-resident* needs the whole input set
   resident exactly like the baseline does, so it OOMs at the same N -- measured:
   at N=8M both FlashAttention and device-resident Stream-CQSA fail, and only the
   host-streamed configuration returns a result. Recovery therefore requires
   Q/K/V in host memory.

   But the caller still holds references to the device tensors, so a plain
   `q.cpu()` copies without freeing anything. `release_inputs=True` rebinds
   `t.data` to host storage, which frees the device allocation even though the
   caller's variable is still alive -- at the cost of mutating the caller's
   tensors. It is opt-in for that reason, and it is what makes the difference
   between recovering and failing again.

3. **Match the output contract.** `stream_cqsa_forward` returns `(out, info)` in
   fp32; SDPA returns a bare tensor in the input dtype. The wrapper reconciles
   both so it substitutes cleanly.

4. **Let the planner pick the depth.** `itr="auto"` reads free memory and chooses;
   a hardcoded depth either fails or over-decomposes.

Limitations, stated plainly
---------------------------
* **Forward only.** There is no `torch.autograd.Function` wrapper yet, so the
  returned tensor does not carry a backward. Use `stream_cqsa_backward`
  explicitly (it needs `info["lse"]`, so pass `return_info=True`).
* `dropout_p` must be 0 and GQA/MQA is unsupported, matching the kernel.
* With `release_inputs=False` (the default) recovery only succeeds if the
  transient workspace was what overflowed, not the inputs themselves.
"""
from __future__ import annotations

import gc
import warnings
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["attention_oom_safe", "stream_cqsa_auto", "ESCALATION"]

_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, _OOM) or "out of memory" in str(exc).lower()


def _drain():
    gc.collect()
    torch.cuda.empty_cache()


def attention_oom_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    primary=None,
    release_inputs: bool = False,
    max_itr: int | None = None,
    return_info: bool = False,
    verbose: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Attention that survives an out-of-memory failure of the normal path.

    q/k/v are ``[B, H, N, D]``. Returns the output in the inputs' dtype and
    device, so it substitutes for ``scaled_dot_product_attention``.

    `primary` is the fast path to try first (default: SDPA). `release_inputs`
    permits moving Q/K/V to host memory on the fallback -- see the module
    docstring; without it, inputs that do not fit cannot be recovered.
    """
    from .stable_stream import max_depth_for, stream_cqsa_forward

    if q.dim() != 4:
        raise ValueError(f"expected [B, H, N, D], got {tuple(q.shape)}")
    out_dtype, out_device = q.dtype, q.device
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5
    info: dict[str, Any] = {"path": "primary", "recovered": False}

    if primary is None:
        def primary(qq, kk, vv):
            return F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal,
                                                  scale=scale)

    if q.device.type == "cuda":
        try:
            out = primary(q, k, v)
            return (out, info) if return_info else out
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            if verbose:
                warnings.warn(
                    f"attention OOMed ({str(exc).split(chr(10))[0][:80]}); "
                    "falling back to Stream-CQSA", RuntimeWarning, stacklevel=2)
            info["primary_error"] = str(exc).split("\n")[0][:200]
            _drain()

    # ---- fallback ---------------------------------------------------------
    if release_inputs and q.device.type == "cuda":
        # Rebind the caller's tensors onto host storage. `.cpu()` alone would
        # copy and leave the device allocation alive behind the caller's own
        # reference, which is the usual reason a retry OOMs as well.
        for t in (q, k, v):
            t.data = t.data.to("cpu")
        _drain()

    host = q.device.type == "cpu"

    # Escalate the decomposition depth until it fits. Two reasons this cannot
    # just be itr="auto":
    #
    #   * The planner sizes itself from torch.cuda.mem_get_info, which reports
    #     DEVICE-WIDE free memory. It does not see a per-process cap
    #     (set_per_process_memory_fraction), another tenant's allocation, or a
    #     container limit, so it can report far more headroom than this process
    #     can actually use and pick too shallow a depth.
    #   * itr=0 means "no decomposition" -- a single monolithic call, which is
    #     exactly what just failed. Accepting it here guarantees a second OOM.
    #
    # So: start at least at 1, and on OOM go deeper. Each level shrinks the
    # subproblems by ~c/l^2, which is the guardrail behaviour the method claims.
    # The ceiling is structural, not a constant: c**itr <= N is the deepest
    # decomposition the sequence admits, past which a level would produce
    # subproblems smaller than a single chunk. Capping at a fixed number instead
    # would abandon calls that a legal deeper level would still have fitted.
    if max_itr is None:
        max_itr = max(1, max_depth_for(int(q.shape[-2])))

    attempts, out, cinfo = [], None, None
    for depth in range(1, max_itr + 1):
        try:
            out, cinfo = stream_cqsa_forward(q, k, v, itr=depth, causal=causal,
                                             scale=scale, stream_from_host=host)
            break
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            attempts.append(depth)
            if verbose:
                warnings.warn(f"itr={depth} also OOMed; trying itr={depth + 1}",
                              RuntimeWarning, stacklevel=2)
            _drain()
    if out is None:
        raise torch.cuda.OutOfMemoryError(
            f"Stream-CQSA could not fit even at itr={max_itr}. "
            + ("Pass release_inputs=True so Q/K/V can move to host memory."
               if not host else
               "Q/K/V are already host-resident; the device cannot hold even one "
               "subproblem at this depth.")) from None
    info.update({"path": "stream_cqsa", "recovered": True,
                 "itr": cinfo.get("itr"), "streamed": host,
                 "n_subproblems": cinfo.get("n_subproblems"),
                 "itr_attempts_oomed": attempts,
                 "plan_reason": cinfo.get("plan_reason")})
    if return_info:
        info["lse"] = cinfo.get("lse")       # the backward needs this
    out = out.to(device=out_device, dtype=out_dtype)
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------
# The no-brainer entry point
# ---------------------------------------------------------------------------

# Why the default ladder is ordered the way it is.
#
# Device memory is three terms, and deeper decomposition only shrinks one:
#
#     inputs Q/K/V        O(N)  -- invariant in itr; needs stream_from_host
#     fp32 accumulator    O(N)  -- invariant in itr; needs accumulate_on_gpu=False
#     in-flight pieces    O(N/l^itr) -- this is the only one itr touches
#
# So a ladder that only deepens `itr` stalls at a floor of (inputs + accumulator)
# and OOMs forever after, no matter how deep it goes. Any ladder that means to
# "keep trying until it fits" must therefore reach the fully-relocated
# configuration -- and since it must reach it anyway, the default starts there.
# The DEFAULT ladder starts from the safest configuration rather than the
# cheapest, because "auto" should mean "this returns an answer", not "this is
# fast if you are lucky". Rung 1 already has both O(N) device terms removed --
# inputs streamed from the host, fp32 accumulator host-resident -- so the only
# thing left to escalate is depth.
#
# Starting safe is not a speed sacrifice, which is the non-obvious part.
# `itr="auto"` asks the planner for a depth, and when a monolithic call fits the
# planner returns itr=0 and does not decompose at all -- "not decomposing is both
# faster and more accurate". The old cheapest-first ladder opened with a fixed
# `itr=1`, forcing a decomposition nobody needed. Measured on an A100-40GB,
# forward, fp16, B=1 H=8 D=64:
#
#     N=262144   old rung 1: 3.52 s / 2.37 GiB    default: 0.74 s / 1.51 GiB
#     N=1048576  old rung 1: 13.31 s / 9.48 GiB   default: 9.95 s / 6.03 GiB
#
# i.e. the safe default is 1.3-4.8x FASTER and uses 0.64x the peak memory.
ESCALATION = [
    dict(itr="auto", stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=2, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=3, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=4, stream_from_host=True, accumulate_on_gpu=False),
]

# Cheapest-first: keep everything device-resident as long as possible, and only
# relocate after an OOM. Lower per-subproblem overhead once a decomposition is
# genuinely needed, because nothing crosses the bus, but it OOMs far earlier --
# the (inputs + accumulator) floor is O(N) and no depth of `itr` shrinks it.
# Pass explicitly: `stream_cqsa_auto(..., ladder=ESCALATION_FAST)`.
ESCALATION_FAST = [
    dict(itr=1),
    dict(itr=2),
    dict(itr=1, stream_from_host=True),
    dict(itr=2, stream_from_host=True),
    dict(itr=2, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=3, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=4, stream_from_host=True, accumulate_on_gpu=False),
]


def stream_cqsa_auto(q, k, v, *, causal=False, scale=None, return_info=False,
                     ladder=None, verbose=False):
    """
    Exact attention that returns an answer, whatever N you give it.

    Walks `ESCALATION`, which starts from the SAFEST configuration -- inputs
    streamed from host memory, fp32 accumulator host-resident, depth chosen
    automatically -- and deepens the decomposition if even that OOMs. The first
    rung that completes wins.

    Depth is automatic, so this does not decompose when it does not have to:
    if a monolithic call fits, the planner says so and the call is monolithic.

    .. warning::
       **This relocates q/k/v to host memory in place.** Streaming from the host
       is what removes the largest O(N) device term, and the relocation rebinds
       ``.data`` so the device allocation is genuinely released rather than
       merely copied. The tensors you passed in will therefore be CPU-resident
       when the call returns. Pass ``.clone()`` if you need to keep them on the
       device, or use `stream_cqsa_forward` directly for explicit control.

    Parameters
    ----------
    ladder : list[dict], optional
        Override the escalation sequence. `ESCALATION_FAST` keeps everything
        device-resident as long as possible; it is lower-overhead once a
        decomposition is genuinely needed, but OOMs far earlier.

    Returns the output in the inputs' original dtype (or `(out, info)` with
    `return_info=True`; `info["lse"]` is required by `stream_cqsa_backward`,
    and `info["config"]` reports which rung ran).
    """
    from .stable_stream import stream_cqsa_forward

    out_dtype = q.dtype
    out_device = q.device
    tried = []
    for cfg in (ladder or ESCALATION):
        need_host = cfg.get("stream_from_host", False)
        try:
            if need_host and q.device.type == "cuda":
                for t in (q, k, v):
                    t.data = t.data.to("cpu")     # release, not copy
                _drain()
            if not need_host and q.device.type == "cpu":
                continue                          # cannot run device-resident now
            out, info = stream_cqsa_forward(q, k, v, causal=causal, scale=scale,
                                            **cfg)
            info = dict(info)
            info.update(config=cfg, rungs_tried=tried)
            out = out.to(device=out_device if not need_host else out.device,
                         dtype=out_dtype)
            return (out, info) if return_info else out
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            tried.append(cfg)
            if verbose:
                warnings.warn(f"{cfg} OOMed; escalating", RuntimeWarning, stacklevel=2)
            _drain()
    raise torch.cuda.OutOfMemoryError(
        f"exhausted the escalation ladder ({len(tried)} rungs). The device cannot "
        f"hold even one subproblem at the deepest setting; reduce N, H or D, or "
        f"use a larger GPU.")
