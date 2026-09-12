"""Smoke tests for next/: devkit, autoconfig planner, adapters. GPU + native kernel required for the first two."""
import pytest, torch
from stream_cqsa.autoconfig import hardware_from_dict, plan, CostModel
from stream_cqsa.devkit import Config, select_by_rule


def test_planner_monolithic_when_it_fits():
    hw = hardware_from_dict({"cuda:0": "40GiB", "host": "200GiB"})
    p = plan(N=262144, hardware=hw)
    assert p.mode == "mono" and p.itr == 0


def test_planner_decomposes_under_budget():
    hw = hardware_from_dict({"cuda:0": "3GiB", "host": "200GiB"})
    p = plan(N=1048576, hardware=hw)
    assert p.mode == "cqsa" and p.itr >= 1 and p.stream_from_host and p.est_peak_gib <= 3 * 0.85


def test_planner_uses_devices_when_faster():
    # 2 devices at c=7: the 4/7 shard bound plus merge does not beat one monolithic call (measured 34.6 vs 24.2 s at 2M);
    # with the quorum-set axis a c=13/21 split (7/6, 11/10 tasks) is predicted to win -- either answer is a valid plan here
    hw2 = hardware_from_dict({"cuda:0": "80GiB", "cuda:1": "80GiB", "host": "500GiB", "link_gbs": 200})
    p2 = plan(N=4194304, hardware=hw2); assert p2.mode in ("mono", "cqsa_dist")
    assert plan(N=4194304, hardware=hw2, quorum_sets={7: (0, 1, 3)}).mode == "mono"
    # 4 devices: measured 3.2x at 4M against 1.6x cqsa overhead -> distributed wins
    hw4 = hardware_from_dict({f"cuda:{i}": "80GiB" for i in range(4)} | {"host": "500GiB", "link_gbs": 200})
    p = plan(N=4194304, hardware=hw4)
    assert p.world == 4 and p.mode == "cqsa_dist"


def test_rule_pareto():
    rows = [dict(ok=True, s=1.0, peak_gib=10.0, config="a"), dict(ok=True, s=1.2, peak_gib=5.0, config="b"),
            dict(ok=True, s=0.9, peak_gib=12.0, config="c"), dict(ok=True, s=0.9, peak_gib=11.0, config="d")]
    assert select_by_rule(rows, budget_gib=20)["config"] == "d"      # fastest, tie broken to less memory
    assert select_by_rule(rows, budget_gib=10.5)["config"] == "a"    # c/d exceed the budget
    assert select_by_rule(rows, budget_gib=1.0) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_compare_kernels_native_exact():
    from stream_cqsa.devkit import compare_kernels
    from stream_cqsa.stable_stream import local_stats_flash
    rep = compare_kernels(local_stats_flash, N=16384, itr=2, reps=1, acc_rows=64, verbose=False)
    assert rep.verdict.startswith(("bit-identical", "exact"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_flex_adapter_positions():
    from stream_cqsa.devkit import compare_kernels
    from stream_cqsa.adapters import flex_inner
    from torch.nn.attention.flex_attention import flex_attention
    fa = torch.compile(flex_attention, dynamic=False)
    def alibi(score, b, h, q_idx, kv_idx): return score - 0.05 * (q_idx - kv_idx).abs()
    mono = lambda q, k, v: fa(q, k, v, score_mod=alibi, scale=64 ** -0.5)
    ok = compare_kernels(flex_inner(score_mod=alibi), mono_fn=mono, N=8192, itr=1, causal=False, reference=None, reps=1, verbose=False)
    bad = compare_kernels(flex_inner(score_mod=alibi, global_positions=False), mono_fn=mono, N=8192, itr=1, causal=False, reference=None, reps=1, verbose=False)
    assert ok.verdict.startswith("exact") and bad.verdict.startswith("NOT exact")


def test_planner_quorum_axis_and_direction():
    from stream_cqsa.autoconfig import QUORUM_SETS, is_perfect_difference_set
    assert all(is_perfect_difference_set(c, s) for c, s in QUORUM_SETS.items())
    hw = hardware_from_dict({"cuda:0": "3GiB", "host": "500GiB"})
    f = plan(N=1048576, hardware=hw, direction="fwd"); b = plan(N=1048576, hardware=hw, direction="bwd")
    assert f.mode == "cqsa" and b.mode == "cqsa" and (f.c, f.itr) != (7, 3)      # a larger c at a lower depth is available
    assert b.direction == "bwd" and f.est_peak_gib <= 3 * 0.85 and b.est_peak_gib <= 3 * 0.85
    kw = f.engine_kwargs(); assert kw["c"] == f.c and kw["interest_set"] == f.interest_set


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_backward_depth_independent():
    from stream_cqsa.native_autograd import stream_cqsa_attn
    N = 16384
    q, k, v = (torch.randn(1, 4, N, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
    stream_cqsa_attn(q, k, v, causal=True, itr=1, bwd_itr=2).float().sum().backward()
    g2 = q.grad.clone(); q.grad = None
    stream_cqsa_attn(q, k, v, causal=True, itr=1, bwd_itr="auto").float().sum().backward()
    assert ((q.grad - g2).float().norm() / g2.float().norm()).item() < 2e-3
