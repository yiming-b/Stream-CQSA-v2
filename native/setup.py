import glob
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "stream-cqsa"


def get_version() -> str:
    cand = [THIS_DIR / "stream_cqsa" / "__init__.py", THIS_DIR.parent / "stream_cqsa" / "__init__.py"]
    init_py = next(p for p in cand if p.exists()).read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in stream_cqsa/__init__.py")
    return match.group(1)


def detect_cuda_arch() -> str:
    """
    Compute capability of the GPU in this machine, e.g. "80" for A100.

    Source installs are the supported path: the user builds for the GPU they
    actually have, so the default is to detect it rather than to guess. Set
    FLASH_ATTN_CUDA_ARCHS explicitly to cross-compile or to cover several
    generations, e.g. "80;86;89;90".
    """
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}{minor}"
    except Exception:
        # No visible GPU at build time (CI, container, login node). sm_80 is a
        # reasonable floor: FlashAttention's SM80 kernels need Ampere or newer.
        print("WARNING: no CUDA device visible at build time; defaulting to "
              "arch 80. Set FLASH_ATTN_CUDA_ARCHS to match your target GPU.")
        return "80"


def cuda_arch_list() -> list[str]:
    raw = os.getenv("FLASH_ATTN_CUDA_ARCHS", "") or detect_cuda_arch()
    archs = [a.strip() for a in raw.split(";") if a.strip()]
    if not archs:
        raise RuntimeError("FLASH_ATTN_CUDA_ARCHS resolved to an empty list")
    for a in archs:
        if not a.isdigit():
            raise RuntimeError(f"Invalid CUDA arch '{a}'. Use numeric values like 80;90")
    return archs


def nvcc_arch_flags(archs: list[str]) -> list[str]:
    flags: list[str] = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    newest = max(archs, key=int)
    flags.extend(["-gencode", f"arch=compute_{newest},code=compute_{newest}"])
    return flags


def cqsa_kernel_set() -> str:
    # Default to the A100-focused fast iteration path. Set CQSA_KERNEL_SET=full
    # to compile all forward kernels.
    # `common` is the shipping default: fp16 + bf16, head_dim 64 and 128,
    # causal and non-causal. Covers most transformers without `full`'s
    # multi-hour compile.
    return os.getenv("CQSA_KERNEL_SET", "common").strip().lower()


def cqsa_sources() -> list[str]:
    base = [
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/src/cqsa_kernel.cu",
        "csrc/flash_attn/src/wave_kernels.cu",
    ]

    kernel_set = cqsa_kernel_set()
    if kernel_set == "full":
        return (
            base
            + sorted(glob.glob(str(THIS_DIR / "csrc" / "flash_attn" / "src" / "flash_fwd_*.cu")))
            + sorted(glob.glob(str(THIS_DIR / "csrc" / "flash_attn" / "src" / "flash_bwd_*.cu")))
        )
    if kernel_set == "a100_fp16_hdim128":
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
        ]
    if kernel_set == "common":
        # Shipping target: both dtypes users actually train in, and the two head
        # dims that cover most transformers (Llama 128, GPT-2 64). Exotic head
        # dims need CQSA_KERNEL_SET=full.
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
        ]
    if kernel_set == "a100_fp16_hdim64_128":
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
        ]
    if kernel_set == "native_dev":
        # fast iteration: hdim64 fp16 only, forward (causal + non-causal) and backward
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
        ]
    if kernel_set == "native_fwd64":
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
        ]
    if kernel_set == "a100_fp16_hdim64_noncau":
        return base + [
            "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
        ]
    raise RuntimeError(
        f"Unknown CQSA_KERNEL_SET='{kernel_set}'. "
        "Expected one of: full, common, a100_fp16_hdim128, "
        "a100_fp16_hdim64_128, a100_fp16_hdim64_noncau."
    )


def build_extension() -> CUDAExtension:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set. Activate a CUDA-enabled environment before building CQSA.")
    if torch.version.hip is not None:
        raise RuntimeError("CQSA lite setup currently supports CUDA only (ROCm disabled).")

    cutlass_header = THIS_DIR / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists():
        raise RuntimeError(
            "csrc/cutlass is missing. Copy it from flash-attention or clone with submodules before build."
        )

    kernel_set = cqsa_kernel_set()
    sources = cqsa_sources()
    if not sources:
        raise RuntimeError("No CUDA sources found for CQSA build")

    nvcc_threads = os.getenv("NVCC_THREADS", "4")
    arch_flags = nvcc_arch_flags(cuda_arch_list())

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ] + ([
            # Measurement build: compiles CQS out of the backward kernel to
            # recover the stock-FlashAttention baseline. Not correct; A/B only.
            "-DCQSA_BWD_NO_CQS",
        ] if os.getenv("CQSA_BWD_NO_CQS") else []) + [
            "--threads",
            str(nvcc_threads),
            *arch_flags,
        ],
    }

    define_macros = []
    if kernel_set != "full":
        define_macros.append(("CQSA_MINIMAL_FWD_KERNELS", None))
    if kernel_set == "a100_fp16_hdim64_noncau":
        define_macros.append(("CQSA_MINIMAL_HDIM64_NONCAUSAL_ONLY", None))
    if kernel_set == "common":
        define_macros.append(("CQSA_MINIMAL_BOTH_DTYPES", None))
    if kernel_set in ("native_dev", "native_fwd64"):
        define_macros.append(("CQSA_NATIVE_HDIM64_ONLY", None))


    return CUDAExtension(
        name="cqsa_native",
        sources=sources,
        include_dirs=[
            str(THIS_DIR / "csrc" / "flash_attn"),
            str(THIS_DIR / "csrc" / "flash_attn" / "src"),
            str(THIS_DIR / "csrc" / "cutlass" / "include"),
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description="CQSA: lightweight CUDA extension for CQS/CQSA forward kernels",
    packages=[],
    ext_modules=[build_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
    install_requires=["torch"],
)
