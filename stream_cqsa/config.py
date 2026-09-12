from __future__ import annotations

from pathlib import Path

# Project paths
STREAM_CQSA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBING_HISTORY_DIR = STREAM_CQSA_ROOT / "Probing_history"
DEFAULT_PROBING_HISTORY = DEFAULT_PROBING_HISTORY_DIR / "probing_history.csv"

# Probe defaults
DEFAULT_PROBE_START_NUM_ITR = 1
DEFAULT_PROBE_MAX_NUM_ITR = 12

# Streaming defaults
GLOBAL_STREAM_SEED = 123

# Memory-budget coefficient used by analytic probing and parallel-subsequence estimation.
# 1.0 => use full effective probe budget (subject to runtime validation).
MEM_BUDGET_COE = 1.0
