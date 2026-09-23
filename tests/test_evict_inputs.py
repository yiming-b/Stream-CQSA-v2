"""End-to-end model case: Q/K/V are on the device and there is little room for attention.
attention() releases them to host memory for the computation, delivers O on the device
first, and brings Q/K/V back only if room remains."""
import warnings

import pytest, torch
import torch.nn.functional as F

GIB = 2 ** 30
N, H, D = 262144, 8, 64                       # Q/K/V = 0.75 GiB fp16, O = 0.25 GiB
TIGHT = {"cuda:0": "1.2GiB", "host": "16GiB"}  # whole allowance: nothing fits next to the resident Q/K/V


def test_planner_models_resident_evictable_inputs():
    from stream_cqsa.autoconfig import plan, hardware_from_dict
    hw = hardware_from_dict(TIGHT)
    R = 3 * N * H * D * 2
    kw = dict(N=N, H=H, D=D, hardware=hw, allow_distributed=False, out_bytes_per_el=2,
              resident_input_bytes=R, budget_excludes_inputs=False)
    p = plan(evict_inputs=True, **kw)
    assert p.evict_inputs and p.stream_from_host and p.est_peak_gib * GIB <= 1.2 * GIB
    assert "evicted" in p.name()
    p0 = plan(evict_inputs=False, **kw)        # the device copy stays: streaming is charged for it, nothing fits
    assert not p0.evict_inputs and "nothing fits" in p0.reason
    # a budget measured with the inputs in place (free memory) credits them to device-resident candidates
    roomy = hardware_from_dict({"cuda:0": "0.6GiB", "host": "16GiB"})
    p1 = plan(N=N, H=H, D=D, hardware=roomy, allow_distributed=False, out_bytes_per_el=2,
              resident_input_bytes=R, budget_excludes_inputs=True, evict_inputs=True)
    assert any(c["ok"] and not c["stream_from_host"] for c in p1.candidates)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _qkv(n=N, grad=False):
    g = torch.Generator(device="cuda").manual_seed(0)
    return tuple(torch.randn(1, H, n, D, dtype=torch.float16, device="cuda", generator=g) for _ in range(3))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
def test_evict_compute_output_on_device_then_restore():
    import stream_cqsa
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    q0 = q.clone()
    out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=TIGHT, return_info=True)
    assert info["inputs_evicted"] and info["evict_released_gib"] > 0.7        # the device copies were really freed
    assert out.is_cuda and out.dtype == torch.float16 and _rel(out, ref) < 2e-3
    assert info["inputs_restored"] and q.is_cuda and k.is_cuda and v.is_cuda    # room on the real device: they come back
    assert torch.equal(q, q0)


@cuda
def test_restore_never_and_evict_never():
    import stream_cqsa
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=TIGHT, return_info=True, restore_inputs=False)
    assert out.is_cuda and info["inputs_evicted"] and not info["inputs_restored"]
    assert q.device.type == "cpu" and q.is_pinned()                             # still valid tensors, in host memory
    assert _rel(out, ref) < 2e-3
    q, k, v = _qkv()
    out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=TIGHT, return_info=True, evict_inputs=False)
    assert not info["inputs_evicted"] and q.is_cuda and _rel(out, ref) < 2e-3


@cuda
def test_views_of_a_live_fused_tensor_are_reported():
    import stream_cqsa
    g = torch.Generator(device="cuda").manual_seed(1)
    fused = torch.randn(3, 1, H, 65536, D, dtype=torch.float16, device="cuda", generator=g)
    q, k, v = fused.unbind(0)                                                   # views: `fused` keeps the storage alive
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, info = stream_cqsa.attention(q, k, v, is_causal=True, evict_inputs=True, itr=1, return_info=True)
    assert any("still references their storage" in str(x.message) for x in w)
    assert info["evict_released_gib"] < 0.05 and _rel(out, ref) < 2e-3


@cuda
def test_training_through_eviction():
    import stream_cqsa
    n = 32768
    leaves = [t.requires_grad_(True) for t in _qkv(n)]
    q, k, v = (t * 1.0 for t in leaves)                                         # non-leaf, as inside a model
    out = stream_cqsa.attention(q, k, v, is_causal=True, evict_inputs=True, itr=1)
    assert out.is_cuda
    do = torch.randn_like(out)
    out.backward(do)
    got = [t.grad.clone() for t in leaves]
    assert all(g_.is_cuda for g_ in got)
    for t in leaves:
        t.grad = None
    F.scaled_dot_product_attention(*(t * 1.0 for t in leaves), is_causal=True).backward(do)
    for g_, t in zip(got, leaves):
        assert _rel(g_, t.grad) < 5e-3


@cuda
def test_forced_eviction_on_the_monolithic_path_returns_to_the_device():
    import stream_cqsa
    q, k, v = _qkv(8192)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out, info = stream_cqsa.attention(q, k, v, is_causal=True, evict_inputs=True, return_info=True)
    assert info["engine"] == "monolithic" and info["inputs_evicted"]
    assert out.is_cuda and info["inputs_restored"] and q.is_cuda                # was: output returned in host memory
    assert _rel(out, ref) < 2e-3


@cuda
def test_restore_judges_room_under_a_memory_fraction_cap():
    import stream_cqsa
    q, k, v = _qkv()
    total = torch.cuda.get_device_properties(0).total_memory
    used = torch.cuda.memory_allocated(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, (used + 3 * GIB) / total), 0)
    try:
        out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=TIGHT, return_info=True)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # a reserve of 10% of the PHYSICAL device (8 GiB on an 80 GB card) never left room under a 3 GiB cap
    assert info["inputs_evicted"] and info["inputs_restored"] and q.is_cuda and out.is_cuda
