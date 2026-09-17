"""place_inputs guardrail and the verbose configuration lines."""
import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_place_inputs_moves_when_it_fits_and_keeps_on_host_otherwise():
    import stream_cqsa
    t = torch.randn(1, 8, 4096, 64, dtype=torch.float16)
    (a,) = (stream_cqsa.place_inputs(t),)
    assert a.is_cuda
    big = torch.randn(1, 8, 4096, 64, dtype=torch.float16)
    b = stream_cqsa.place_inputs(big, reserve_fraction=1.0)          # nothing may be moved
    assert not b.is_cuda and b.is_pinned()
    q, k, v = stream_cqsa.place_inputs(t.cpu(), t.cpu(), t.cpu())
    out = stream_cqsa.attention(q, k, v, is_causal=True)
    assert out.shape == t.shape


def test_verbose_configuration_lines(capsys):
    import stream_cqsa
    q, k, v = (torch.randn(1, 8, 8192, 64, dtype=torch.float16, device="cuda") for _ in range(3))
    stream_cqsa.attention(q, k, v, is_causal=True, verbose=True)
    s = capsys.readouterr().out
    assert "Stream-CQSA plan: monolithic" in s and "Q/K/V on device" in s
    stream_cqsa.attention(q, k, v, is_causal=True, verbose=True, itr=1, accumulate_on_gpu=False, low_memory=True)
    s = capsys.readouterr().out
    assert "Stream-CQSA plan: decomposed: itr=1, c=7, 7 subproblems" in s and "accumulator on host (CPU)" in s
    assert "Stream-CQSA ran: decomposed" in s and "engine classic" in s
