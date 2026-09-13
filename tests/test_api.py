"""The one-call API, the dry run, the SDPA patch, the doctor, the calibration cache."""
import os, tempfile
import pytest, torch, torch.nn.functional as F
import stream_cqsa
from stream_cqsa import attention, estimate, patch_sdpa, unpatch_sdpa, patched_sdpa, doctor, kernels_available

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
dev = torch.device("cuda")


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def ref64(q, k, v, causal):
    return F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=causal)


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_attention_monolithic_below_boundary(causal):
    q, k, v = (torch.randn(1, 4, 4096, 64, device=dev, dtype=torch.float16) for _ in range(3))
    out, p = attention(q, k, v, is_causal=causal, return_plan=True)
    assert p.mode == "mono" and out.dtype == q.dtype and out.shape == q.shape
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal))


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_attention_forced_decomposition_is_exact(causal):
    q, k, v = (torch.randn(1, 4, 4096, 64, device=dev, dtype=torch.float16) for _ in range(3))
    ref = ref64(q, k, v, causal)
    o = attention(q, k, v, is_causal=causal, itr=1)
    o_s = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert rel(o, ref) <= 1.2 * rel(o_s, ref)


@cuda
def test_attention_kernel_choices_agree():
    q, k, v = (torch.randn(1, 4, 4096, 64, device=dev, dtype=torch.float16) for _ in range(3))
    outs = {kn: attention(q, k, v, is_causal=True, itr=1, kernel=kn) for kn in ("cuda", "triton") if
            (kn != "cuda" or kernels_available()["cuda_extension"]) and (kn != "triton" or kernels_available()["triton"])}
    if kernels_available()["native_wave"]:
        outs["wave"] = attention(q, k, v, is_causal=True, itr=1, kernel="wave")
    os.environ.pop("CQSA_FORWARD", None); os.environ.pop("CQSA_BACKWARD", None)
    ref = ref64(q, k, v, True)
    for kn, o in outs.items():
        assert rel(o, ref) < 5e-4, kn


@cuda
def test_attention_host_inputs_and_memory_cap():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    q, k, v = (torch.randn(1, 4, 262144, 64, dtype=torch.float16) for _ in range(3))    # monolithic needs ~0.7 GiB
    try:
        torch.cuda.set_per_process_memory_fraction(1.5 / total)      # 1.5 GiB cap, 0.5 GiB budget: must decompose
        out, p = attention(q, k, v, is_causal=True, hardware={"cuda:0": "0.5GiB", "host": "64GiB"}, return_plan=True)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert p.mode != "mono" and out.device.type == "cpu" and out.dtype == torch.float16
    ref = F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev), is_causal=True).cpu()   # fp16 kernel as the yardstick
    assert rel(out, ref) < 1e-3


@cuda
def test_attention_autograd():
    q, k, v = (torch.randn(1, 4, 4096, 64, device=dev, dtype=torch.float16, requires_grad=True) for _ in range(3))
    dout = torch.randn_like(q)
    attention(q, k, v, is_causal=True, itr=1).backward(dout)
    g = [t.grad.clone() for t in (q, k, v)]
    q2, k2, v2 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=True).backward(dout.double())
    for a, b in zip(g, (q2.grad, k2.grad, v2.grad)):
        assert rel(a, b) < 2e-3


@cuda
def test_patch_sdpa_roundtrip():
    q, k, v = (torch.randn(1, 4, 4096, 64, device=dev, dtype=torch.float16) for _ in range(3))
    orig = F.scaled_dot_product_attention
    with patched_sdpa(min_tokens=1024):
        assert F.scaled_dot_product_attention is not orig
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # routed; below the boundary -> monolithic
        m = torch.zeros(4096, 4096, dtype=torch.bool, device=dev)
        o_m = F.scaled_dot_product_attention(q, k, v, attn_mask=m)            # masked: passed through to the original
    assert F.scaled_dot_product_attention is orig
    assert torch.equal(o, orig(q, k, v, is_causal=True)) and o_m.shape == q.shape


def test_attention_rejects_unsupported():
    q = torch.randn(1, 2, 64, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        attention(q, q, q, attn_mask=torch.ones(64, 64, dtype=torch.bool))
    with pytest.raises(TypeError):
        attention(q.float(), q.float(), q.float())


@cuda
def test_estimate_and_doctor():
    r = estimate(1 << 20, hardware={"cuda:0": "40GiB", "host": "256GiB"}, print_table=False)
    assert r["fwd"]["feasible"] and r["bwd"]["feasible"] and "plan" in r["fwd"]
    r2 = estimate(1 << 24, hardware={"cuda:0": "8GiB", "host": "64GiB"}, print_table=False)
    assert not r2["fwd"]["mono_fits"]
    d = doctor(check=True, print_report=False)
    assert d["check"]["ok"] and d["limits"]["fwd"]["stream_cqsa"] >= d["limits"]["fwd"]["monolithic"]


def test_cost_model_cache_roundtrip(tmp_path, monkeypatch):
    from stream_cqsa.autoconfig import CostModel, save_cost_model, load_cost_model
    monkeypatch.setenv("CQSA_CACHE_DIR", str(tmp_path))
    cm = CostModel(); cm.pair_rate = 1.23e12
    save_cost_model(cm, "cuda" if torch.cuda.is_available() else "cpu")
    back = load_cost_model("cuda" if torch.cuda.is_available() else "cpu")
    assert back is not None and back.pair_rate == 1.23e12
