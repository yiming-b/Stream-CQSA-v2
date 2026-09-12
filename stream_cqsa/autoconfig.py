"""
Automatic configuration of Stream-CQSA from a hardware description.

    from stream_cqsa.autoconfig import detect_hardware, plan, autotune, auto_attention

    hw  = detect_hardware()                        # every CUDA device + host RAM
    hw  = hardware_from_dict({"cuda:0": "40GiB", "cuda:1": "40GiB", "host": "200GiB"})
    cfg = plan(N=2**20, B=1, H=8, D=64, dtype=torch.float16, causal=True, hardware=hw)
    out = auto_attention(q, k, v, causal=True, hardware=hw)   # plan + run

The two knobs the user otherwise has to set are the decomposition depth
(``itr``: 0 = monolithic call, 1 = 7 subproblems, 2 = 49, ...) and where the
accumulator lives (``acc``: gpu or cpu), plus whether Q/K/V stay on the host,
how many subproblems run concurrently, and over how many devices. The planner
enumerates every combination, predicts peak device memory with the engine's
own estimators and wall time with a small cost model, discards what does not
fit the budget, and applies one rule:

    use as much of the memory budget as helps, unless a configuration is both
    faster and smaller -- i.e. take the Pareto frontier of (time, memory) among
    feasible configurations and choose its fastest point.

For a small N with an ample budget that always yields the monolithic call:
Stream-CQSA at itr=1 does 9/7 of the monolithic pair work (of which 22% is then
skipped as fully masked tiles) plus gather/merge, and measured end to end it is
1.4-1.6x the monolithic time at equal residency -- so "all 7 subproblems in
parallel with a device accumulator" is chosen only when the monolithic call
does not fit, which is exactly what the measurements say.

The cost model is deliberately simple (pair throughput + per-token host stage
costs + a per-depth factor + distributed balance/merge terms) and its constants
come from the A100 measurements in next/LOG.md. ``calibrate(hardware)`` refits
them on the machine at hand by timing a monolithic call and a few Stream-CQSA
configurations at a small N (a minute), and ``autotune`` replaces the model by
direct measurement of the feasible candidates when the shape is small enough
to afford it.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import torch

from .stable_stream import (estimate_monolithic_bytes, estimate_peak_bytes, estimate_peak_bytes_bwd, per_el_floor, effective_free_bytes)

GIB = float(1 << 30)

# Perfect (Singer) difference sets: c = l^2 - l + 1 chunks, |interest_set| = l.
# A subproblem gathers l of the c chunks; the pair work per depth is (l^2/c)^itr
# of the monolithic call, the masked fraction inside a subproblem is (l-1)/l^2,
# the gathered tokens per depth are l^itr * N and there are c^itr tasks.
# The engine's int64 group bits allow itr*(l-1) <= 63.
QUORUM_SETS: dict[int, tuple[int, ...]] = {
    3: (0, 1), 7: (0, 1, 3), 13: (0, 1, 3, 9), 21: (0, 1, 4, 14, 16),
    31: (0, 1, 3, 8, 12, 18), 57: (0, 1, 3, 13, 32, 36, 43, 52),
    73: (0, 1, 3, 7, 15, 31, 36, 54, 63),
}


def is_perfect_difference_set(c: int, s: Sequence[int]) -> bool:
    d = sorted((a - b) % c for a in s for b in s if a != b)
    return d == list(range(1, c))


# ---------------------------------------------------------------------------
# hardware description
# ---------------------------------------------------------------------------

def _parse_bytes(x) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([kmgt]i?b?)?\s*", str(x), re.I)
    if not m:
        raise ValueError(f"cannot parse memory size {x!r}")
    val = float(m.group(1)); unit = (m.group(2) or "b").lower().rstrip("b").rstrip("i")
    return int(val * {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}[unit])


@dataclass
class DeviceSpec:
    name: str                 # torch device string, e.g. 'cuda:0'
    model: str = "unknown"
    budget_bytes: int = 0     # memory this call may use on the device
    total_bytes: int = 0
    n_sm: int = 108
    identical_group: int = 0  # devices with the same model share a group


@dataclass
class HardwareSpec:
    devices: list[DeviceSpec] = field(default_factory=list)
    host_budget_bytes: int = 0        # host RAM this call may use (pinned Q/K/V, accumulator)
    n_cpu: int = 8
    interconnect_gbs: float = 25.0    # per-direction device<->device bandwidth used by the merge (NVLink ~200, PCIe ~25)

    def homogeneous(self) -> bool:
        return len({d.model for d in self.devices}) <= 1

    def summary(self) -> str:
        devs = ", ".join(f"{d.name}={d.model} {d.budget_bytes / GIB:.1f}/{d.total_bytes / GIB:.1f} GiB" for d in self.devices)
        return f"devices[{devs}] host {self.host_budget_bytes / GIB:.0f} GiB, {self.n_cpu} cpus, link {self.interconnect_gbs:.0f} GB/s"


def _host_free_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 64 * (1 << 30)


def detect_hardware(*, budget_fraction: float = 0.9, host_fraction: float = 0.8,
                    devices: Sequence[int] | None = None) -> HardwareSpec:
    """Describe the visible CUDA devices (free memory x budget_fraction) and the host."""
    hw = HardwareSpec(host_budget_bytes=int(_host_free_bytes() * host_fraction),
                      n_cpu=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8))
    if not torch.cuda.is_available():
        return hw
    idx = list(devices) if devices is not None else list(range(torch.cuda.device_count()))
    models = {}
    for i in idx:
        p = torch.cuda.get_device_properties(i)
        free, total = effective_free_bytes(i), p.total_memory     # honours a set_per_process_memory_fraction cap
        grp = models.setdefault(p.name, len(models))
        hw.devices.append(DeviceSpec(name=f"cuda:{i}", model=p.name, budget_bytes=int(free * budget_fraction),
                                     total_bytes=int(total), n_sm=p.multi_processor_count, identical_group=grp))
    if len(hw.devices) > 1:
        nv = _nvlink_present()
        hw.interconnect_gbs = 200.0 if nv else 25.0
    return hw


def _nvlink_present() -> bool:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10).stdout
        return bool(re.search(r"\bNV\d+\b", out))
    except Exception:
        return False


def hardware_from_dict(spec: dict[str, Any], *, interconnect_gbs: float | None = None) -> HardwareSpec:
    """
    {"cuda:0": "40GiB", "cuda:1": 40e9, "host": "256GiB", "cpus": 32, "link_gbs": 200}
    Values are memory *budgets* (what the call may use). Devices default to the
    detected model/SM count when the process can see them.
    """
    hw = HardwareSpec()
    models = {}
    for key, val in spec.items():
        k = str(key).lower()
        if k in ("host", "cpu", "ram"):
            hw.host_budget_bytes = _parse_bytes(val)
        elif k in ("cpus", "n_cpu", "ncpu"):
            hw.n_cpu = int(val)
        elif k in ("link_gbs", "interconnect_gbs"):
            hw.interconnect_gbs = float(val)
        else:
            m = re.fullmatch(r"(?:cuda:)?(\d+)", k)
            if not m:
                raise ValueError(f"unknown hardware key {key!r} (use 'cuda:N', 'host', 'cpus', 'link_gbs')")
            i = int(m.group(1))
            # value: a size, or {"mem": size, "model": "A100-SXM4-80GB"}; devices are
            # assumed identical unless a model string says otherwise.
            model, n_sm = "same", 108
            if isinstance(val, dict):
                model = str(val.get("model", "same")); mem = _parse_bytes(val.get("mem", val.get("budget", 0)))
            else:
                mem = _parse_bytes(val)
            total = mem
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                p = torch.cuda.get_device_properties(i); n_sm, total = p.multi_processor_count, p.total_memory
            grp = models.setdefault(model, len(models))
            hw.devices.append(DeviceSpec(name=f"cuda:{i}", model=model, budget_bytes=mem,
                                         total_bytes=total, n_sm=n_sm, identical_group=grp))
    if not hw.host_budget_bytes:
        hw.host_budget_bytes = int(_host_free_bytes() * 0.8)
    if interconnect_gbs is not None:
        hw.interconnect_gbs = float(interconnect_gbs)
    return hw


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Wall-time model. Defaults measured on A100-SXM4-80GB with the next/ engine
    (v11 kernel), fp16, H=8 D=64 (next/logs/e2e_compare_v11.json,
    pipeline_opts.json, dist_test_*.json). `calibrate` refits them.
    """
    pair_rate: float = 7.4e11        # monolithic FA-2: score pairs per second over all heads (causal 1M, H=8: 8*N^2/2 / 5.96 s)
    kernel_ratio: float = 1.40       # Stream-CQSA kernel time per live pair / FA-2 time per pair (v11 at L=56K)
    masked_frac: float = 0.22        # kept for calibrate(); the planner uses (l-1)/l^2 per quorum set
    depth_factor: float = 1.0        # residual per-depth factor (per-task overhead now explicit below)
    task_overhead_s: float = 3.0e-3  # fixed cost per subproblem (launch, scheduling, small-L kernel inefficiency); fitted from c=7..73 at N=131K
    bwd_ratio: float = 3.0           # backward (fwd+bwd of a subproblem) / forward time per pair
    # Host-stage costs per gathered token of H*D = 512 fp16 elements (the
    # measured shape); scaled by (B*H*D)/512 for other shapes.
    gather_s_per_tok: float = 0.5 / 3.1e6     # host gather (shared_chunks) per gathered token
    gather_dev_s_per_tok: float = 0.06 / 3.1e6  # device-side gather per token (inputs resident)
    merge_cpu_s_per_tok: float = 2.25 / 3.1e6  # host accumulator merge per gathered token, 8 threads
    merge_gpu_s_per_tok: float = 0.10 / 3.1e6  # device accumulator merge per gathered token
    d2h_s_per_tok: float = 0.25 / 3.1e6        # output d2h per gathered token (host accumulator)
    overlap: float = 0.85            # fraction of the host stages hidden under compute (pipelined engine, n_par=1)
    overlap_npar2: float = 0.92      # with two subproblems in flight
    dist_merge_bytes_per_tok: int = 16 * 8 * 64  # fp32 out + weights, per token (H*D*4 + ...), all_reduce volume
    dist_imbalance: float = 1.35     # per-rank straggling + host-stage contention (4 GPUs sharing one host; measured 62 s vs 40 s ideal at 4M)
    dist_fixed_s: float = 0.4        # per-call cost of the collective path (barriers, NCCL warm-up, task-list build)

    def mono_time(self, N: int, B: int, H: int, causal: bool, direction: str = "fwd") -> float:
        pairs = B * H * N * N * (0.5 if causal else 1.0)
        return pairs / self.pair_rate * (self.bwd_ratio if direction == "bwd" else 1.0)

    def cqsa_time(self, N: int, B: int, H: int, D: int, itr: int, causal: bool, acc: str,
                  stream_from_host: bool, n_par: int, world: int = 1, hw: HardwareSpec | None = None,
                  c: int = 7, l: int = 3, direction: str = "fwd") -> float:
        L = N * (l / c) ** itr
        n_tasks = c ** itr
        tokens = n_tasks * L                       # gathered tokens over all subproblems
        masked = (l - 1) / (l * l)                 # non-owner diagonal blocks of a subproblem
        pairs = n_tasks * B * H * L * L * (0.5 if causal else 1.0) * (1 - masked)
        compute = pairs / self.pair_rate * self.kernel_ratio * (self.depth_factor ** (itr - 1)) + n_tasks * self.task_overhead_s
        if direction == "bwd":
            compute *= self.bwd_ratio
        ts = tokens * (B * H * D) / 512.0          # token-equivalents of the calibrated shape
        gather = ts * (self.gather_s_per_tok if stream_from_host else self.gather_dev_s_per_tok)
        merge = ts * (self.merge_cpu_s_per_tok if acc == "cpu" else self.merge_gpu_s_per_tok)
        d2h = ts * self.d2h_s_per_tok if acc == "cpu" else 0.0
        host = gather + merge + d2h
        ov = self.overlap_npar2 if n_par >= 2 else self.overlap
        t = compute + host * (1 - ov) + max(0.0, host * ov - compute)   # hidden part cannot exceed compute
        if world > 1:
            per_rank = math.ceil(n_tasks / world) / n_tasks
            comm = 2 * N * B * H * D * 4 / (max(1.0, (hw.interconnect_gbs if hw else 25.0)) * 1e9) * math.log2(world)
            t = t * per_rank * self.dist_imbalance + comm + self.dist_fixed_s
        return t

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


# ---------------------------------------------------------------------------
# candidates and the plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    mode: str                      # 'mono' | 'cqsa' | 'cqsa_dist'
    itr: int = 0
    c: int = 7
    interest_set: tuple = (0, 1, 3)
    direction: str = "fwd"
    acc: str = "gpu"
    stream_from_host: bool = False
    n_par: int = 1
    world: int = 1
    devices: list[str] = field(default_factory=list)
    est_time_s: float = 0.0
    est_peak_gib: float = 0.0      # per device
    est_host_gib: float = 0.0
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)

    def name(self) -> str:
        if self.mode == "mono":
            return "monolithic"
        s = f"cqsa c={self.c} itr={self.itr} acc={self.acc} n_par={self.n_par}" + (" host-resident Q/K/V" if self.stream_from_host else "")
        return s + (f" on {self.world} devices" if self.world > 1 else "")

    def engine_kwargs(self) -> dict:
        return dict(itr=int(self.itr), c=int(self.c), interest_set=tuple(self.interest_set),
                    low_memory=(self.acc == "cpu"), accumulate_on_gpu=(self.acc == "gpu"),
                    stream_from_host=bool(self.stream_from_host), max_parallel=int(self.n_par),
                    shared_chunks=bool(self.stream_from_host and self.itr == 1))


def _host_bytes(N, B, H, D, itemsize, stream_from_host, acc):
    b = 0
    if stream_from_host:
        b += 3 * N * B * H * D * itemsize          # pinned Q/K/V
    if acc == "cpu":
        b += N * B * H * D * 4 + N * B * H * 8     # fp32 accumulator (+ m, l)
    return b


def plan(*, N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
         hardware: HardwareSpec | None = None, model: CostModel | None = None,
         max_itr: int = 3, n_pars: Sequence[int] = (1, 2, 4), allow_distributed: bool = True,
         quorum_sets: dict[int, Sequence[int]] | None = None, direction: str = "fwd",
         safety: float = 0.85, tol: float = 0.02) -> Plan:
    """
    Choose the configuration for one attention call on `hardware` (detected if None).

    `quorum_sets` ({c: interest_set}, default QUORUM_SETS) is the fourth axis:
    every (c, itr) pair is a candidate. Measured at N=131K (A100-40GB): for the
    same subproblem size a larger c at a lower depth beats a smaller c at a
    higher depth (c=31 itr=1: 468 ms, 1.40 GiB vs c=7 itr=2: 531 ms, 1.40 GiB;
    c=73 itr=1: 519 ms vs c=13 itr=2: 891 ms), because the pair work is
    (l^2/c)^itr and per-task overhead is c^itr; the model has both terms.
    `direction="bwd"` plans the backward (its own memory estimate and cost) --
    the forward and backward depths are chosen independently.
    """
    hw = hardware or detect_hardware()
    cm = model or CostModel()
    itemsize = torch.empty((), dtype=dtype).element_size()
    if not hw.devices:
        raise RuntimeError("no CUDA device in the hardware description")
    # Heterogeneous machines: plan on the largest identical group (the engine
    # splits tasks evenly, so mixing device speeds would idle the fast ones).
    groups = {}
    for d in hw.devices:
        groups.setdefault(d.identical_group, []).append(d)
    devs = max(groups.values(), key=lambda g: (len(g), min(x.budget_bytes for x in g)))
    dev_budget = min(d.budget_bytes for d in devs) * safety
    cands: list[dict] = []

    bwd = direction == "bwd"
    mono = estimate_monolithic_bytes(N, B=B, H=H, D=D, itemsize=itemsize) * (2.2 if bwd else 1.0)
    cands.append(dict(mode="mono", itr=0, c=7, interest_set=(0, 1, 3), acc="gpu", stream_from_host=False, n_par=1, world=1,
                      time=cm.mono_time(N, B, H, causal, direction), peak=mono / GIB, host=0.0,
                      ok=mono <= dev_budget))
    worlds = [1] + ([w for w in range(2, len(devs) + 1)] if allow_distributed else [])
    qsets = dict(quorum_sets) if quorum_sets is not None else dict(QUORUM_SETS)
    for c, iset in qsets.items():
        l = len(iset)
        if not is_perfect_difference_set(c, iset):
            raise ValueError(f"(c={c}, interest_set={tuple(iset)}) is not a perfect difference set")
        for w in worlds:
            for itr in range(1, max_itr + 1):
                n_tasks = c ** itr
                if itr * (l - 1) > 63 or (w > 1 and n_tasks < w) or n_tasks > 4096:
                    continue
                for acc in ("gpu", "cpu"):
                    for host in (False, True):
                        if acc == "cpu" and not host:
                            continue
                        for n_par in n_pars:
                            if n_par > n_tasks:
                                continue
                            est = estimate_peak_bytes_bwd if bwd else estimate_peak_bytes
                            peak = est(N, itr, B=B, H=H, D=D, itemsize=itemsize, n_par=n_par, c=c, l=l,
                                       stream_from_host=host, accumulate_on_gpu=(acc == "gpu"))
                            hostb = _host_bytes(N, B, H, D, itemsize, host, acc) * (2.0 if bwd else 1.0)
                            t = cm.cqsa_time(N, B, H, D, itr, causal, acc, host, n_par, world=w, hw=hw, c=c, l=l, direction=direction)
                            cands.append(dict(mode="cqsa" if w == 1 else "cqsa_dist", itr=itr, c=c, interest_set=tuple(iset),
                                              acc=acc, stream_from_host=host,
                                              n_par=n_par, world=w, time=t, peak=peak / GIB, host=hostb / GIB,
                                              ok=(peak <= dev_budget and hostb <= hw.host_budget_bytes)))
    feas = [c for c in cands if c["ok"]]
    if not feas:
        floor = per_el_floor(itemsize, True, False) * N * B * H * D
        p = Plan(mode="cqsa", itr=max_itr, c=max(qsets), interest_set=tuple(qsets[max(qsets)]), direction=direction,
                 acc="cpu", stream_from_host=True, n_par=1, world=len(devs),
                 devices=[d.name for d in devs], candidates=cands,
                 reason=(f"nothing fits: device budget {dev_budget / GIB:.1f} GiB, host budget {hw.host_budget_bytes / GIB:.0f} GiB; "
                         f"the O(N.H.D) device floor with everything offloaded is {floor / GIB:.1f} GiB. "
                         f"Falling back to the deepest, most offloaded configuration on every device; expect OOM."))
        return p
    front = [c for c in feas if not any((o["time"] < c["time"]) and (o["peak"] < c["peak"]) for o in feas)]
    best_t = min(c["time"] for c in front)
    near = [c for c in front if c["time"] <= best_t * (1 + tol)]
    pick = min(near, key=lambda c: c["peak"])
    p = Plan(mode=pick["mode"], itr=pick["itr"], c=pick["c"], interest_set=tuple(pick["interest_set"]), direction=direction,
             acc=pick["acc"], stream_from_host=pick["stream_from_host"],
             n_par=pick["n_par"], world=pick["world"], devices=[d.name for d in devs[:pick["world"]]],
             est_time_s=pick["time"], est_peak_gib=pick["peak"], est_host_gib=pick["host"], candidates=cands)
    mono_c = cands[0]
    if pick["mode"] == "mono":
        p.reason = (f"monolithic fits ({mono_c['peak']:.2f} GiB <= {dev_budget / GIB:.1f} GiB budget) and is the fastest "
                    f"feasible configuration ({mono_c['time']:.2f} s predicted)")
    else:
        alt = sorted(front, key=lambda c: c["time"])[:3]
        why = (f"monolithic fits ({mono_c['peak']:.1f} GiB) at ~{mono_c['time']:.1f} s on one device, but" if mono_c["ok"]
               else f"monolithic needs {mono_c['peak']:.1f} GiB > {dev_budget / GIB:.1f} GiB budget;")
        p.reason = (f"{why} fastest feasible on the "
                    f"(time, memory) frontier: {p.name()} ~{p.est_time_s:.1f} s at {p.est_peak_gib:.1f} GiB/device"
                    + (f", host {p.est_host_gib:.0f} GiB" if p.est_host_gib else "")
                    + "; frontier: " + "; ".join(f"{_cname(c)} {c['time']:.1f}s/{c['peak']:.1f}GiB" for c in alt))
    return p


def _cname(c: dict) -> str:
    if c["mode"] == "mono":
        return "mono"
    return f"c{c.get('c', 7)}/itr{c['itr']}/{c['acc']}{'/host' if c['stream_from_host'] else ''}/np{c['n_par']}" + (f"x{c['world']}" if c["world"] > 1 else "")


# ---------------------------------------------------------------------------
# calibration and empirical autotune
# ---------------------------------------------------------------------------

def calibrate(hardware: HardwareSpec | None = None, *, N: int = 131072, B: int = 1, H: int = 8, D: int = 64,
              dtype=torch.float16, causal: bool = True, device="cuda", verbose: bool = True) -> CostModel:
    """
    Refit the cost model on this machine: time the monolithic kernel and three
    Stream-CQSA configurations (itr=1 acc=gpu; itr=1 acc=cpu host-resident;
    itr=2 acc=gpu) at a small N and solve for pair_rate, kernel_ratio,
    host-stage cost and depth_factor. About a minute on an A100 at N=131072.
    """
    from .devkit import Config, run_config, measure
    from .stable_stream import TraceRecorder, stream_cqsa_forward
    cm = CostModel()
    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(dtype).pin_memory() for _ in range(3))
    scale = float(D) ** -0.5
    _, pm = measure(lambda: run_config(q, k, v, Config(mode="mono"), causal=causal, device=dev), device=dev, reps=2)
    pairs = B * H * N * N * (0.5 if causal else 1.0)
    cm.pair_rate = pairs / pm["s"]
    # itr=1, device accumulator, device-resident inputs: compute-dominated
    tr = TraceRecorder(enabled=True, device=dev)
    qd, kd, vd = (t.to(dev) for t in (q, k, v))
    stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1)       # warm
    tr = TraceRecorder(enabled=True, device=dev)
    _, info = stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1, trace=tr)
    st = info.get("stage_totals_ms", {})
    L = N * 3 / 7; tokens = 7 * L * (B * H * D) / 512.0
    live_pairs = 7 * B * H * L * L * (0.5 if causal else 1.0) * (1 - (cm.masked_frac if causal else cm.masked_frac * 0.5))
    if st.get("compute"):
        cm.kernel_ratio = (st["compute"] / 1e3) / (live_pairs / cm.pair_rate)
    if st.get("gather"):
        cm.gather_dev_s_per_tok = st["gather"] / 1e3 / tokens
    if st.get("merge"):
        cm.merge_gpu_s_per_tok = st["merge"] / 1e3 / tokens
    del qd, kd, vd
    # itr=1 host accumulator, host-resident inputs
    tr = TraceRecorder(enabled=True, device=dev)
    _, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal, allow_escalation=False, max_parallel=1,
                                  stream_from_host=True, low_memory=True, shared_chunks=True, trace=tr)
    st = info.get("stage_totals_ms", {})
    if st.get("gather"):
        cm.gather_s_per_tok = st["gather"] / 1e3 / tokens
    if st.get("merge"):
        cm.merge_cpu_s_per_tok = st["merge"] / 1e3 / tokens
    if st.get("d2h"):
        cm.d2h_s_per_tok = st["d2h"] / 1e3 / tokens
    # depth factor from itr=2 vs itr=1 (device accumulator)
    qd, kd, vd = (t.to(dev) for t in (q, k, v))
    _, p1 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1)[0], device=dev, reps=2)
    _, p2 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=2, causal=causal, allow_escalation=False, max_parallel=1)[0], device=dev, reps=2)
    _, p31 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1,
                                                 c=31, interest_set=QUORUM_SETS[31])[0], device=dev, reps=2)
    # c=31 and c=7 at itr=1 do the same pair work to within 10% (36/31 vs 9/7, masked 5/36 vs 2/9);
    # the difference is 24 extra tasks' overhead plus 3N more gathered tokens.
    cm.task_overhead_s = 0.0
    base7 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=7, l=3)
    base31 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=31, l=6)
    cm.task_overhead_s = max(0.0, ((p31["s"] - p1["s"]) - (base31 - base7)) / 24.0)
    cm.depth_factor = 1.0
    pred2 = cm.cqsa_time(N, B, H, D, 2, causal, "gpu", False, 1, c=7, l=3)
    pred1 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=7, l=3)
    cm.depth_factor = max(1.0, (p2["s"] / p1["s"]) / (pred2 / max(pred1, 1e-9)))
    if verbose:
        print(f"calibrate: mono {pm['s']:.3f}s -> pair_rate {cm.pair_rate:.2e}/s; kernel_ratio {cm.kernel_ratio:.2f}; "
              f"gather {cm.gather_s_per_tok*1e9:.1f} ns/tok (host) {cm.gather_dev_s_per_tok*1e9:.1f} (dev); "
              f"merge {cm.merge_cpu_s_per_tok*1e9:.1f} ns/tok (cpu) {cm.merge_gpu_s_per_tok*1e9:.1f} (gpu); "
              f"d2h {cm.d2h_s_per_tok*1e9:.1f} ns/tok; task_overhead {cm.task_overhead_s*1e3:.2f} ms (c=31 vs 7); itr2/itr1 {p2['s']/p1['s']:.2f} -> depth_factor {cm.depth_factor:.2f}")
    return cm


def autotune(*, N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
             hardware: HardwareSpec | None = None, max_candidates: int = 6, device="cuda", verbose: bool = True) -> Plan:
    """
    Plan, then MEASURE the top feasible single-device candidates on the real
    shape and pick by the same rule from measurements. Use when one call is
    cheap enough to run a few times (say under a minute) and will be repeated.
    """
    from .devkit import Config, quick_bench
    p = plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hardware)
    feas = [c for c in p.candidates if c["ok"] and c["world"] == 1]
    feas.sort(key=lambda c: c["time"])
    cfgs = []
    for c in feas[:max_candidates]:
        cfgs.append(Config(mode="mono") if c["mode"] == "mono" else
                    Config(mode="cqsa", itr=c["itr"], c=c["c"], interest_set=tuple(c["interest_set"]), acc=c["acc"],
                           stream_from_host=c["stream_from_host"], n_par=c["n_par"]))
    hw = hardware or detect_hardware()
    budget = min(d.budget_bytes for d in hw.devices) / GIB
    res = quick_bench(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, configs=cfgs, budget_gib=budget,
                      device=device, verbose=verbose, acc_rows=64, reps=1)
    pick = res["pick"]
    if pick is None:
        return p
    cfg = pick["cfg"]
    return Plan(mode=cfg["mode"], itr=cfg["itr"], c=cfg.get("c", 7), interest_set=tuple(cfg.get("interest_set", (0, 1, 3))),
                acc=cfg["acc"], stream_from_host=cfg["stream_from_host"],
                n_par=cfg["n_par"], world=1, devices=[hw.devices[0].name], est_time_s=pick["s"],
                est_peak_gib=pick["peak_gib"], candidates=p.candidates,
                reason=f"measured: {pick['config']} {pick['s']:.3f} s at {pick['peak_gib']:.2f} GiB (rule applied to measurements)")


# ---------------------------------------------------------------------------
# one-call entry point
# ---------------------------------------------------------------------------

def auto_attention(q, k, v, *, causal: bool = False, scale: float | None = None,
                   hardware: HardwareSpec | dict | None = None, model: CostModel | None = None,
                   plan_only: bool = False, verbose: bool = False, **overrides):
    """
    Plan for this call's shape and hardware, then run it. Returns (out, plan).
    Distributed plans run only inside an initialised torch.distributed group of
    the planned size; otherwise the single-device frontier point is used.
    `overrides` are forwarded to the engine (e.g. allow_escalation=False).
    """
    from .stable_stream import stream_cqsa_forward
    if isinstance(hardware, dict):
        hardware = hardware_from_dict(hardware)
    B, H, N, D = q.shape
    p = plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=causal, hardware=hardware, model=model)
    if p.mode == "cqsa_dist":
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() == p.world):
            single = [c for c in p.candidates if c["ok"] and c["world"] == 1]
            if single:
                best = min(single, key=lambda c: c["time"])
                p.reason += f" | not in a {p.world}-rank process group: using the single-device point {_cname(best)}"
                p.mode, p.itr, p.c, p.interest_set, p.acc, p.stream_from_host, p.n_par, p.world = \
                    best["mode"], best["itr"], best["c"], tuple(best["interest_set"]), best["acc"], best["stream_from_host"], best["n_par"], 1
    if verbose:
        print(f"auto_attention: {p.name()} -- {p.reason}")
    if plan_only:
        return None, p
    if p.mode == "mono":
        from .interface import flash_attn_func
        dev = torch.device(p.devices[0])
        qq, kk, vv = (t.to(dev, non_blocking=True) for t in (q, k, v))
        out = flash_attn_func(qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                              softmax_scale=scale, causal=causal).transpose(1, 2)
        return out, p
    kw = p.engine_kwargs(); kw.update(overrides)
    if p.stream_from_host:
        qq, kk, vv = (t if t.device.type == "cpu" else t.cpu() for t in (q, k, v))
    else:
        dev = torch.device(p.devices[0]); qq, kk, vv = (t.to(dev, non_blocking=True) for t in (q, k, v))
    if p.mode == "cqsa_dist":
        from .distributed import dist_stream_cqsa_forward
        out, _ = dist_stream_cqsa_forward(qq, kk, vv, causal=causal, scale=scale, **kw)
    else:
        out, _ = stream_cqsa_forward(qq, kk, vv, causal=causal, scale=scale, **kw)
    return out, p
