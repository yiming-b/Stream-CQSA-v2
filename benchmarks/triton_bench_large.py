"""Kernel-level Triton vs native vs FA-2 at the 1M subproblem shape (L=449K) and 2M (L=899K), causal."""
import torch, gc
from stream_cqsa.triton_kernel import cqs_attention_forward
from stream_cqsa.stable_stream import local_stats_flash
from stream_cqsa.interface import cqs_block_summaries
from stream_cqsa.reference import group_bits_for_path
from flash_attn import flash_attn_func
def ms(fn, it=3, wu=1):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [fn() for _ in range(it)]; e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
H, D = 8, 64; scale = D ** -0.5
for N in (262144, 1048576, 2097152):
    ids, bits_np = group_bits_for_path(N, (0,), sorted_gather=True); L = len(bits_np)
    q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=torch.float16) for _ in range(3))
    b = torch.as_tensor(bits_np, device="cuda"); bo, ba = (t.cuda() for t in cqs_block_summaries(b))
    for causal in (True, False):
        r = dict(fa2=ms(lambda: flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)),
                 tri_plain=ms(lambda: cqs_attention_forward(q, k, v, None, causal=causal, scale=scale)),
                 tri=ms(lambda: cqs_attention_forward(q, k, v, b, causal=causal, scale=scale, blk_or=bo, blk_and=ba)),
                 nat=ms(lambda: local_stats_flash(q, k, v, b, causal=causal, scale=scale, blk_or=bo, blk_and=ba)))
        print(f"N={N:>8} L={L:>7} causal={int(causal)}: fa2 {r['fa2']:8.1f}  triton plain {r['tri_plain']:8.1f}  triton CQS {r['tri']:8.1f}  native CQS {r['nat']:8.1f}  triton/native {r['tri']/r['nat']:.2f}", flush=True)
    del q, k, v; gc.collect(); torch.cuda.empty_cache()
