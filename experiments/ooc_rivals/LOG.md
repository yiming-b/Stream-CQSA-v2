# OOC rivals study: log

## 2026-09-15

- Rival implemented: `stream_cqsa/baselines/rect_ooc.py` (R2 fixed / R3 adaptive; forward on
  `_flash_attn_forward` with fp32 log-sum-exp merge, backward on `_flash_attn_backward` with the global
  lse and O; independent q/kv tiles with exact causal sub-pairs; pinned event-guarded staging, double
  buffering and kv-outer backward for R3; byte / kernel / phase / memory accounting). Tests
  `tests/test_rect_ooc.py`: 7 passed (fp64 exactness within FA-2's error, both schedules, uneven N,
  non-causal, budget compliance, probe cache). A pinned-buffer reuse race (the next window's host write
  overtook the previous asynchronous H2D) was the one bug found on the way: buffers are now event-guarded.
- `stream_cqsa.attention(..., return_info=True)` added so the harness logs the automatic mode's engine,
  kernel, depth, subproblem count and retries.
- Harness `experiments/ooc_rivals/bench.py` (worker per measurement, allocator cap = budget, planner given
  the same budget, cold + 3 warm timings, float64 sampled-row forward check, gradient check against the
  first finished method's gradients per N (FA-2 where it fits)). Smoke on della-vis1 at 64K / 2 GiB: all
  four methods, both directions, ok (Stream-CQSA auto chose the monolithic call there).
- Grid submitted (gpu-short, `gpu80&pcie`, 96G host): jobs 13952418-13952425 = budgets {10, 20, 40, 80}
  GiB x {fwd, bwd}; N in {256K, 512K, 1M, 2M, 4M}, methods fa2 / rect_ooc / rect_ooc_adaptive / cqsa_auto.
