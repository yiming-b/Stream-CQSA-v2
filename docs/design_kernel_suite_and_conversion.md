# Should Stream-CQSA ship a suite of attention kernels, and can a monolithic kernel be converted automatically?

Answers to the two design questions, with the reasoning and what was prototyped.

## 1. A suite of popular kernels (Kimi Delta Attention, linear attentions)?

**Recommendation: no suite of linear/delta-rule kernels; yes to a small set of
*softmax-family* variants, delivered through one adapter rather than one
kernel each.**

Why linear attention does not belong here. Stream-CQSA solves one problem:
softmax attention's working set is the O(N²) set of query–key pairs, and the
cyclic-quorum decomposition partitions *pairs* into independent subproblems
whose partial (m, l, acc) statistics recompose exactly. That decomposition
needs three properties of the kernel: (i) the output of a row is a
normalised sum over pairs, (ii) each pair's contribution depends only on that
pair (plus a per-row normaliser), (iii) partial sums over disjoint pair sets
merge associatively. Linear attention (Performer/RetNet/GLA), the delta-rule
family (DeltaNet, Gated DeltaNet, Kimi Delta Attention / KDA) and other
state-space-like kernels replace the pair set by a *recurrent state*: the
contribution of key j to query i is not a function of the pair (i, j) but of
the state after processing keys 1…j, i.e. an ordered product of per-token
transitions (the delta rule is a rank-1 update of a d×d state applied
sequentially, and KDA adds a diagonal gate). Their memory is O(N·d²) by
construction, so they have no OOM boundary of the kind this work recovers
from, and their chunked formulations already stream over the sequence with a
carried state. Wrapping them in CQS would be both unnecessary (nothing to
recover) and wrong (the quorum partition does not commute with the ordered
state update). The honest sentence for the paper is: Stream-CQSA applies to
kernels whose per-row output is a normalised sum of pairwise terms — softmax
attention and its biased/masked/gated relatives — and is orthogonal to
linear-time attention, which needs no recovery.

What *does* belong, and costs little: any attention of the form
`out_i = Σ_j softmax_j(s_ij + b_ij) v_j` over a pair set given by a mask.
That covers ALiBi and other relative biases, soft-capping (Gemma), sliding
windows, document/packing masks, GQA/MQA (a head-mapping, not a new kernel),
and gated variants where the gate multiplies the score or the value per pair.
Every one of these is expressible as a FlexAttention `score_mod`/`mask_mod`,
and `stream_cqsa.adapters.flex_inner` turns any such pair into a Stream-CQSA
inner kernel automatically (section 2). So the "suite" is one adapter plus a
catalogue of `score_mod`s, each checked once with `devkit.compare_kernels`.
The native CUDA kernel remains the fast path for plain (optionally causal)
softmax attention; the adapter is the coverage path.

Two variants deserve a native path eventually because they are common and
the FlexAttention route is 5–10× slower at subproblem sizes: sliding-window
causal attention (a `window_size` the FA-2 base already supports; wiring it
through the CQS verdict is a small change) and GQA/MQA (K/V head sharing;
also present in FA-2). Neither changes the decomposition.

## 2. Automatic conversion of a monolithic attention into a CQS inner kernel

An inner kernel has to (a) compute attention restricted to an arbitrary
pair set — Stream-CQSA hands it a gathered subsequence of L tokens and an
int64 group-bits word per token, pair (r, c) kept iff `bits[r] & bits[c] == 0`
— and (b) return the per-row log-sum-exp alongside the normalised output, so
the engine can merge subproblems exactly. A monolithic kernel can be
converted automatically iff it exposes those two things.

**What the prototype does** (`next/pkg/stream_cqsa/adapters.py`, verified with
`devkit.compare_kernels` on the interactive A100):

* `flex_inner(score_mod, extra_mask_mod)` — for anything expressed with
  torch FlexAttention. The CQS pair set becomes a `mask_mod` and a per-
  subproblem `BlockMask` (so masked tiles are skipped by the compiled kernel
  too), `return_lse=True` provides the row statistics. The one subtlety of the
  conversion: the user's `score_mod`/`mask_mod` are written against *global*
  positions, while a subproblem sees local indices of a gathered subsequence.
  The engine now passes the gather index (`token_ids`) to the inner kernel and
  the adapter remaps `q_idx`/`kv_idx` through it before calling the user's
  functions. Results: plain causal attention exact at itr 1 and 2 (rel 2.6e-4,
  equal to FlashAttention's own fp16 error vs float64); ALiBi score_mod exact
  (1.8e-4); sliding window (2048) exact (3.8e-4); and with the position remap
  deliberately disabled the devkit reports "NOT exact" (6.5e-2), which is the
  bug an automatic conversion must catch.
* `dense_inner(mono_masked, returns_lse=False)` — for any kernel that accepts
  a dense boolean mask (e.g. `scaled_dot_product_attention(attn_mask=…)`), for
  small L. If the kernel returns no lse, a second dense pass computes it;
  exact (2.6e-4) but the extra QK^T is the price of a kernel that hides its
  statistics.

**What cannot be converted automatically, and why.** Kernels that return only
the normalised output and accept no pairwise mask (most fused inference
kernels, e.g. plain `flash_attn_func` with a window but no lse in some
wrappers, or FlashInfer's decode kernels): the lse is not recoverable from
the output, and without a mask the subproblem's pair set cannot be restricted.
For those the conversion is *semi*-automatic: (1) if the kernel has a mask but
no lse, add the second pass (2× the score computation, or 1.3× if the kernel
can be asked for the row max and sum separately); (2) if it has an lse but no
mask, restrict the pair set by *gathering* per chunk pair instead — i.e. run
the kernel on (owner chunk queries) × (each kept chunk's keys) and merge the
partial (out, lse) with the engine's own `merge_lse` — this is what the
reference decomposition in `autograd_op.py` does and it needs only a plain
causal/non-causal kernel with lse, at the cost of ~3× more launches; (3) if it
has neither, it cannot be used.

**How I would structure the conversion tool.** A `KernelSpec` describing
capabilities — `{mask: none|dense|block|mask_mod, lse: yes|no, positional:
none|relative|absolute, dtype}` — and a resolver that picks the adapter:
mask_mod+lse → `flex_inner`; dense mask → `dense_inner` (+lse pass if
needed); lse only → chunk-pair gather (adapter to be written; the reference
path shows the mechanics); none → refuse with the reason. Every produced
inner kernel is validated by `compare_kernels` on a small N at itr 1 and 2
before it is accepted, so an unsound conversion (a score_mod that depends on
local position, a kernel that renormalises internally) is caught rather than
silently shipped. This is a few hundred lines on top of what exists; the
remaining real engineering is native CUDA support for windows and GQA, which
no adapter can make fast.
