__version__ = "2.0.0"

from .stable_stream import (
    stream_cqsa_forward,
    stream_cqsa_backward,
    TraceRecorder,
)
from .oom_fallback import (stream_cqsa_auto, attention_oom_safe,
                           ESCALATION, ESCALATION_FAST)
from .native_autograd import stream_cqsa_attn, StreamCQSAAttention

from .interface import (
    flash_attn_func,
    flash_attn_func_cqs,
    flash_attn_func_cqs_group_bits,
    flash_attn_bwd_cqs_group_bits,
    flash_attn_func_cqsa,
)
from .config import (
    DEFAULT_PROBING_HISTORY,
    DEFAULT_PROBING_HISTORY_DIR,
    GLOBAL_STREAM_SEED,
    MEM_BUDGET_COE,
)
from .cqsa_probe_capacity import CQSAProbe
from .cqs_attention_stream import (
    CQSAStreamRunner,
    SubsequenceAttentionFn,
    compare_O,
    run_scqsa_stream,
    stream_cqsa,
)
from .attention_kernel import (
    default_subsequence_attention,
    custom_attn,
)
from .cqs_mask import CQS_mask
from .memory_fitting import fitting_mem, pred_mem, best_seq_length, estimate_memory_time_from_model
from .autograd_op import stream_cqsa_autograd

__all__ = [
    # --- primary entry points -------------------------------------------
    "stream_cqsa_auto",        # "just run it": escalates until it fits
    "stream_cqsa_forward",     # explicit forward, returns (out, lse)
    "stream_cqsa_backward",    # explicit backward, needs the global lse
    "stream_cqsa_attn",        # autograd-aware; supports .backward()
    "StreamCQSAAttention",     # the same, as an nn.Module
    "attention_oom_safe",
    "ESCALATION",
    "ESCALATION_FAST",
    "TraceRecorder",
    # --- everything else --------------------------------------------------
    "flash_attn_func",
    "flash_attn_func_cqs",
    "flash_attn_func_cqs_group_bits",
    "flash_attn_bwd_cqs_group_bits",
    "flash_attn_func_cqsa",
    "CQSAProbe",
    "CQSAStreamRunner",
    "SubsequenceAttentionFn",
    "DEFAULT_PROBING_HISTORY",
    "DEFAULT_PROBING_HISTORY_DIR",
    "GLOBAL_STREAM_SEED",
    "MEM_BUDGET_COE",
    "run_scqsa_stream",
    "stream_cqsa",
    "compare_O",
    "default_subsequence_attention",
    "custom_attn",
    "CQS_mask",
    "fitting_mem",
    "pred_mem",
    "best_seq_length",
    "estimate_memory_time_from_model",
    "stream_cqsa_autograd",
]

# v2 additions
from .autoconfig import (detect_hardware, hardware_from_dict, plan, calibrate, autotune,  # noqa: E402
                         auto_attention, QUORUM_SETS, CostModel, Plan)
from .devkit import compare_kernels, quick_bench, Config, run_config  # noqa: E402
from .adapters import flex_inner, dense_inner  # noqa: E402
