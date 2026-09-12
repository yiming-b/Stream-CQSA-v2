"""
torch.autograd support for the native Stream-CQSA kernel.

`stream_cqsa_attn` is a drop-in for `scaled_dot_product_attention` that
participates in autograd, so `.backward()` works and the operator composes with
the rest of a model:

    out = stream_cqsa_attn(q, k, v, causal=True)      # [B, H, N, D]
    out.sum().backward()                              # dq, dk, dv populated

The existing `stream_cqsa_autograd` in `autograd_op` wraps the *reference*
decomposition. This module wraps the native path of `stable_stream`, so the
forward and backward both run the CUDA kernel and both inherit its out-of-memory
recovery, including the depth escalation.

Three details are load-bearing, and each corresponds to a way a naive wrapper
gets this wrong.

*The global log-sum-exp is a saved tensor.* The backward cannot reconstruct it
from a subsequence, and rebuilding it from `exp` would overflow (see the paper's
overflow-free recomposition). The forward already produces it, so it is stashed
with the other saved tensors rather than recomputed.

*The forward's depth is carried over as the backward's starting depth.* This is
a performance default rather than a correctness requirement: every depth
decomposes the same pair set into a different partition of subproblems, and each
subproblem's contribution is computed against the global log-sum-exp, so the
backward is exact at any depth regardless of what the forward used. Reusing the
depth simply starts the backward where the forward found it could fit, which
saves rediscovering the same memory pressure from scratch. The backward escalates
on its own from there if it still does not fit.

*Residency is preserved across the pass.* With `stream_from_host=True` every
operand must be host-resident, and the forward returns `out` and `lse` wherever
its accumulator lived. The wrapper moves them itself instead of asking the
caller to remember, which is the rough edge the README documents for the
explicit API. The returned output follows the inputs' device for the same
reason: the accumulator's location is a scheduling decision, and letting it
decide where the result lands would make `grad_output` arrive on the wrong
device for host-resident operands.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .stable_stream import stream_cqsa_backward, stream_cqsa_forward

__all__ = ["stream_cqsa_attn", "StreamCQSAAttention"]


class _StreamCQSANative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, itr, c, interest_set,
                stream_from_host, accumulate_on_gpu, max_parallel, sorted_gather, bwd_itr):
        out32, info = stream_cqsa_forward(
            q, k, v,
            itr=itr, causal=bool(causal), scale=scale,
            sorted_gather=bool(sorted_gather),
            max_parallel=max_parallel,
            accumulate_on_gpu=bool(accumulate_on_gpu),
            c=int(c), interest_set=tuple(interest_set),
            stream_from_host=bool(stream_from_host),
        )
        lse = info["lse"]

        # Save the fp32 output: the backward's D = rowsum(dO . O) wants the
        # unrounded one, even though the caller receives the input dtype.
        ctx.save_for_backward(q, k, v, out32, lse)
        ctx.cqsa = dict(
            causal=bool(causal), scale=scale,
            # next/: the backward's depth is its own decision. "auto" (default) lets
            # the backward plan from free memory with its own estimate; "fwd"
            # reuses the forward's depth (old behaviour); an int pins it.
            itr=(int(info["itr"]) if bwd_itr == "fwd" else bwd_itr),
            c=int(c), interest_set=tuple(interest_set),
            stream_from_host=bool(stream_from_host),
            accumulate_on_gpu=bool(accumulate_on_gpu),
            max_parallel=max_parallel,
            sorted_gather=bool(sorted_gather),
            out_dtype=q.dtype,
            monolithic=bool(info.get("monolithic", False)),
        )
        # Match the inputs' device, not the accumulator's. The forward returns
        # its output wherever the accumulator lived, and with itr="auto" that
        # varies -- the monolithic path has no accumulator at all and leaves the
        # result on the device even when the operands are host-resident. An
        # autograd op whose output device depends on a scheduling decision is
        # not composable, and the caller's grad_output would arrive on the wrong
        # device.
        return out32.to(device=q.device, dtype=q.dtype)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out32, lse = ctx.saved_tensors
        cfg = ctx.cqsa
        host = cfg["stream_from_host"]

        # Every operand must sit where the chosen path expects it. The forward
        # may have returned out/lse on the device even when streaming.
        def place(t):
            if t is None:
                return None
            return t.cpu() if host else t

        dout_c = place(dout.contiguous())
        q_c, k_c, v_c = place(q), place(k), place(v)
        out_c, lse_c = place(out32), place(lse)

        bwd_itr = cfg["itr"]
        if cfg["monolithic"] and isinstance(bwd_itr, int) and bwd_itr == 0:
            bwd_itr = "auto"                     # the forward ran monolithically; the backward still decomposes
        dq, dk, dv = stream_cqsa_backward(
            q_c, k_c, v_c, dout_c, out_c, lse_c,
            itr=bwd_itr, causal=cfg["causal"], scale=cfg["scale"],
            sorted_gather=cfg["sorted_gather"],
            c=cfg["c"], interest_set=cfg["interest_set"],
            stream_from_host=host,
            # One knob for the whole call: the backward places its dQ/dK/dV the
            # same way the forward placed its accumulator, so a caller who asked
            # to keep O(N) terms off the device gets that in both directions.
            accumulate_on_gpu=cfg["accumulate_on_gpu"],
            max_parallel=cfg["max_parallel"],
        )

        dt = cfg["out_dtype"]
        to = lambda g, ref: g.to(device=ref.device, dtype=dt)
        return (to(dq, q), to(dk, k), to(dv, v),
                None, None, None, None, None, None, None, None, None, None)


def stream_cqsa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    itr: int | str = "auto",
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    stream_from_host: bool = False,
    accumulate_on_gpu: bool = True,
    max_parallel: int | None = None,
    sorted_gather: bool = True,
    bwd_itr: int | str = "auto",
) -> torch.Tensor:
    """
    Exact attention with autograd support.

    `bwd_itr`: depth of the backward decomposition, chosen independently of the
    forward's -- "auto" (default) plans it from free memory with the backward's
    own estimate, "fwd" reuses the forward's depth, an int pins it.

    Tensors are ``[B, H, N, D]``. The result carries the input dtype and takes
    part in the autograd graph, so ``.backward()`` produces ``dq``, ``dk`` and
    ``dv`` through the native decomposed backward.

    Depth is automatic by default and escalates under memory pressure. The
    backward starts from the depth the forward settled at and escalates further
    on its own if needed, which is a starting point rather than a constraint --
    both passes are exact at any depth.
    """
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"stream_cqsa_attn needs fp16 or bf16 operands, got {q.dtype}. The "
            "FlashAttention kernel accepts no other input dtype -- cast with "
            ".half() or .bfloat16(). Accumulation and the returned gradients are "
            "fp32 internally regardless, so this costs no accuracy in the merge."
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(
            f"q, k and v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}."
        )
    return _StreamCQSANative.apply(
        q, k, v, causal, scale, itr, c, tuple(interest_set),
        stream_from_host, accumulate_on_gpu, max_parallel, sorted_gather, bwd_itr)


class StreamCQSAAttention(torch.nn.Module):
    """`stream_cqsa_attn` as a module, for dropping into an existing block."""

    def __init__(self, *, causal: bool = False, scale: float | None = None,
                 itr: int | str = "auto", c: int = 7,
                 interest_set: Sequence[int] = (0, 1, 3),
                 stream_from_host: bool = False,
                 accumulate_on_gpu: bool = True,
                 max_parallel: int | None = None,
                 bwd_itr: int | str = "auto") -> None:
        super().__init__()
        self.cfg: dict[str, Any] = dict(
            causal=causal, scale=scale, itr=itr, c=c,
            interest_set=tuple(interest_set),
            stream_from_host=stream_from_host,
            accumulate_on_gpu=accumulate_on_gpu,
            max_parallel=max_parallel, bwd_itr=bwd_itr)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        return stream_cqsa_attn(q, k, v, **self.cfg)

    def extra_repr(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in self.cfg.items())
