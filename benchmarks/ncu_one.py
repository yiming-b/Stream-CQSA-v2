"""One kernel launch for ncu: python ncu_one.py <module> <pattern: real|zero|allone> <causal 0/1> [L-from-N]"""
import sys, torch, numpy as np, importlib, os
os.environ["CQSA_CUDA_MODULE"] = sys.argv[1]
from stream_cqsa.interface import flash_attn_func_cqs_group_bits, cqs_block_summaries
from stream_cqsa.reference import group_bits_for_path
N = int(sys.argv[4]) if len(sys.argv) > 4 else 131072
ids, bits_np = group_bits_for_path(N, (0,), sorted_gather=True); L = len(bits_np)
if sys.argv[2] == "fa2":      # upstream flash-attn on the same shape, for instruction-count comparison
    from flash_attn import flash_attn_func
    torch.manual_seed(0); q, k, v = (torch.randn(1, L, 8, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    causal = sys.argv[3] == "1"
    flash_attn_func(q, k, v, causal=causal); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure"); flash_attn_func(q, k, v, causal=causal); torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
    sys.exit(0)
if sys.argv[2] == "plain":    # the CQS build with CQS disabled
    from stream_cqsa.interface import flash_attn_func as cq_plain
    torch.manual_seed(0); q, k, v = (torch.randn(1, L, 8, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    causal = sys.argv[3] == "1"
    cq_plain(q, k, v, causal=causal); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure"); cq_plain(q, k, v, causal=causal); torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
    sys.exit(0)
pat = {"real": bits_np, "zero": np.zeros(L, np.int64), "allone": np.ones(L, np.int64)}[sys.argv[2]]
b = torch.as_tensor(pat, device="cuda"); bo, ba = (x.cuda() for x in cqs_block_summaries(b))
torch.manual_seed(0); q, k, v = (torch.randn(1, L, 8, 64, device="cuda", dtype=torch.float16) for _ in range(3))
causal = sys.argv[3] == "1"
flash_attn_func_cqs_group_bits(q, k, v, b, causal=causal, cqs_blk_or=bo, cqs_blk_and=ba)   # warm
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("measure")
flash_attn_func_cqs_group_bits(q, k, v, b, causal=causal, cqs_blk_or=bo, cqs_blk_and=ba)
torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
