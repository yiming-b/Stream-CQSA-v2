from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _to_num(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _value_match(row_value: str, target: Any) -> bool:
    # Try numeric match first.
    rv_num = _to_num(row_value)
    tv_num = _to_num(target)
    if rv_num is not None and tv_num is not None:
        return float(rv_num) == float(tv_num)
    return str(row_value) == str(target)


def _row_match(row: Dict[str, str], filter: Dict[str, Any]) -> bool:
    for k, target in filter.items():
        if k not in row:
            return False
        rv = row[k]
        if isinstance(target, (list, tuple, set)):
            ok = any(_value_match(rv, t) for t in target)
            if not ok:
                return False
        else:
            if not _value_match(rv, target):
                return False
    return True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_points_csv(path: Path, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "memory", "memory_pred"])
        for xi, yi, pi in zip(x.tolist(), y.tolist(), y_pred.tolist()):
            w.writerow([float(xi), float(yi), float(pi)])


def fitting_mem(
    profile_path: str | Path,
    filter: Dict[str, Any],
    save_model: str | Path,
    degree: int | str = "auto",
    max_degree: int = 3,
) -> Dict[str, Any]:
    """
    Fit memory usage from profiling CSV using np.polyfit on N -> memory.

    Args:
        profile_path: Path to profiling CSV (expects at least columns 'N' and 'memory').
        filter: Row filter, e.g. {'H': 1, 'attn_kernel': 'custom_attn'}.
        save_model: Directory (or file prefix path) where model artifacts are written.

    Args:
        degree:
            - "auto": choose degree by AIC over feasible degrees [0..max_degree]
            - int: force a specific polynomial degree
        max_degree: upper bound used only when degree="auto".

    Returns:
        Dictionary containing fitted coefficients, selected degree, R^2,
        number of points, and artifact paths.
    """
    p_in = Path(profile_path).expanduser().resolve()
    if (not p_in.exists()) or (not p_in.is_file()):
        raise FileNotFoundError(f"profile_path not found: {p_in}")

    with p_in.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"profile_path has no rows: {p_in}")

    headers = set(rows[0].keys())
    for need_col in ("N", "memory"):
        if need_col not in headers:
            raise ValueError(f"profile_path missing required column '{need_col}': {p_in}")

    for k in filter.keys():
        if k not in headers:
            raise ValueError(f"filter key '{k}' not in CSV headers: {sorted(headers)}")

    x_vals: list[float] = []
    y_vals: list[float] = []
    used_rows = 0
    for row in rows:
        if not _row_match(row, filter):
            continue
        n = _to_num(row.get("N", ""))
        m = _to_num(row.get("memory", ""))
        if n is None or m is None:
            continue
        x_vals.append(float(n))
        y_vals.append(float(m))
        used_rows += 1

    if len(x_vals) == 0:
        raise ValueError(f"No usable rows after filtering with {filter}.")

    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Fit X=N and Y=memory.
    unique_n = np.unique(x)
    n_obs = int(len(x))
    max_feasible_degree = int(max(0, min(int(max_degree), int(unique_n.size) - 1, n_obs - 1)))

    if isinstance(degree, str):
        if degree != "auto":
            raise ValueError("degree must be an int or 'auto'.")
        candidates = list(range(0, max_feasible_degree + 1))
        if len(candidates) == 0:
            candidates = [0]
        best_degree = 0
        best_aic = float("inf")
        for d in candidates:
            c_try = np.polyfit(x, y, deg=int(d))
            y_try = np.polyval(c_try, x)
            rss = float(np.sum((y - y_try) ** 2))
            k = int(d) + 1
            # Stabilize near-zero RSS to avoid -inf in log.
            rss_use = max(rss, 1e-12)
            aic = float(n_obs) * math.log(rss_use / max(1, n_obs)) + 2.0 * float(k)
            if aic < best_aic:
                best_aic = aic
                best_degree = int(d)
        degree_sel = int(best_degree)
        degree_mode = "auto"
    else:
        degree_sel = int(degree)
        if degree_sel < 0:
            raise ValueError(f"degree must be >= 0, got {degree_sel}")
        if degree_sel > max_feasible_degree:
            raise ValueError(
                f"Requested degree={degree_sel} is not feasible for filtered data; "
                f"max feasible degree is {max_feasible_degree} "
                f"(num_unique_N={int(unique_n.size)}, num_points={n_obs})."
            )
        degree_mode = "manual"

    coef = np.polyfit(x, y, deg=degree_sel)
    y_pred = np.polyval(coef, x)

    if n_obs >= 2:
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    else:
        r2 = float("nan")

    p_save = Path(save_model).expanduser().resolve()
    if p_save.suffix == "":
        out_dir = _ensure_dir(p_save)
        json_path = out_dir / "memory_fit.json"
        npz_path = out_dir / "memory_fit.npz"
        pt_path = out_dir / "memory_fit.pt"
        points_csv = out_dir / "fit_points.csv"
    else:
        out_dir = _ensure_dir(p_save.parent)
        base = p_save.with_suffix("")
        json_path = base.with_suffix(".json")
        npz_path = base.with_suffix(".npz")
        pt_path = base.with_suffix(".pt")
        points_csv = base.with_name(base.name + "_points").with_suffix(".csv")

    np.savez(
        npz_path,
        coef=coef,
        x=x,
        y=y,
        y_pred=y_pred,
        degree=np.asarray([degree_sel], dtype=np.int32),
    )
    _save_points_csv(points_csv, x, y, y_pred)

    meta = {
        "profile_path": str(p_in),
        "filter": dict(filter),
        "num_points": int(len(x)),
        "num_unique_N": int(unique_n.size),
        "num_rows_used": int(used_rows),
        "degree_mode": str(degree_mode),
        "requested_degree": degree,
        "max_degree": int(max_degree),
        "r2": (None if not math.isfinite(float(r2)) else float(r2)),
    }
    torch.save(
        {"coef": [float(c) for c in np.asarray(coef).tolist()], "degree": int(degree_sel), "meta": meta},
        pt_path,
    )

    result: Dict[str, Any] = {
        "profile_path": str(p_in),
        "filter": dict(filter),
        "num_points": int(len(x)),
        "num_unique_N": int(unique_n.size),
        "num_rows_used": int(used_rows),
        "degree_mode": str(degree_mode),
        "requested_degree": degree,
        "max_degree": int(max_degree),
        "degree": int(degree_sel),
        "coef": [float(c) for c in np.asarray(coef).tolist()],
        "r2": (None if not math.isfinite(float(r2)) else float(r2)),
        "json_path": str(json_path),
        "npz_path": str(npz_path),
        "pt_path": str(pt_path),
        "points_csv": str(points_csv),
    }

    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    return result


def pred_mem(
    model_path: str | Path,
    N: float | int | np.ndarray,
) -> float | np.ndarray:
    """
    Predict memory (GiB) from N using a saved .pt model from fitting_mem().

    Args:
        model_path: path to `memory_fit.pt`, or a directory containing it.
        N: scalar or ndarray sequence length input.

    Returns:
        Predicted memory value(s), same scalar/array style as np.polyval.
    """
    p = Path(model_path).expanduser().resolve()
    if p.is_dir():
        p = p / "memory_fit.pt"
    if (not p.exists()) or (not p.is_file()):
        raise FileNotFoundError(f"model_path not found: {p}")

    model = torch.load(p, map_location="cpu")
    if not isinstance(model, dict):
        raise ValueError(f"Invalid model format at {p}: expected dict.")
    if "coef" not in model:
        raise ValueError(f"Invalid model format at {p}: missing key 'coef'.")

    coef = np.asarray(model["coef"], dtype=np.float64)
    pred = np.polyval(coef, N)
    if np.ndim(pred) == 0:
        return float(pred)  # type: ignore[return-value]
    return pred  # type: ignore[return-value]


def estimate_memory_time_from_model(
    profile_path: str | Path,
    filter: Dict[str, Any],
    save_model: str | Path | None = None,
    N: float | int | None = None,
    *,
    memory_degree: int | str = "auto",
    max_memory_degree: int = 3,
    time_degree: int = 2,
) -> Dict[str, Any]:
    """
    Estimate memory/time at sequence length N using:
      - memory model from fitting_mem(..., save_model=...) when save_model is provided
      - otherwise, fit memory directly from filtered profile rows in-memory
      - time fit from filtered rows in profile_path
    """
    if N is None:
        raise ValueError("N must be provided.")
    p_in = Path(profile_path).expanduser().resolve()
    if (not p_in.exists()) or (not p_in.is_file()):
        raise FileNotFoundError(f"profile_path not found: {p_in}")

    with p_in.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"profile_path has no rows: {p_in}")

    headers = set(rows[0].keys())
    for need_col in ("N", "time"):
        if need_col not in headers:
            raise ValueError(f"profile_path missing required column '{need_col}': {p_in}")
    for k in filter.keys():
        if k not in headers:
            raise ValueError(f"filter key '{k}' not in CSV headers: {sorted(headers)}")

    x_vals: list[float] = []
    t_vals: list[float] = []
    x_mem_vals: list[float] = []
    m_vals: list[float] = []
    for row in rows:
        if not _row_match(row, filter):
            continue
        n = _to_num(row.get("N", ""))
        t = _to_num(row.get("time", ""))
        m = _to_num(row.get("memory", ""))
        if n is None or t is None:
            continue
        x_vals.append(float(n))
        t_vals.append(float(t))
        if m is not None:
            x_mem_vals.append(float(n))
            m_vals.append(float(m))

    if len(x_vals) < 2:
        raise ValueError(
            f"Need at least 2 filtered points for time fit, got {len(x_vals)} (filter={filter})."
        )

    # Memory prediction.
    if save_model is not None:
        p_save = Path(save_model).expanduser().resolve()
        if p_save.suffix == "":
            pt_path = p_save / "memory_fit.pt"
        else:
            pt_path = p_save.with_suffix(".pt")

        if not pt_path.exists():
            fitting_mem(
                profile_path=profile_path,
                filter=filter,
                save_model=save_model,
                degree=memory_degree,
                max_degree=max_memory_degree,
            )
        model = torch.load(pt_path, map_location="cpu")
        if not isinstance(model, dict) or ("coef" not in model):
            raise ValueError(f"Invalid model format at {pt_path}: expected dict with key 'coef'.")
        coef_mem = np.asarray(model["coef"], dtype=np.float64)
        deg_mem = int(model.get("degree", len(coef_mem) - 1))
        mem_est = float(np.polyval(coef_mem, float(N)))
        model_pt_path: str | None = str(pt_path)
    else:
        if len(x_mem_vals) < 2:
            raise ValueError(
                f"Need at least 2 filtered points with valid 'memory' values to fit memory in-memory, "
                f"got {len(x_mem_vals)} (filter={filter})."
            )
        xm = np.asarray(x_mem_vals, dtype=np.float64)
        ym = np.asarray(m_vals, dtype=np.float64)
        order_m = np.argsort(xm)
        xm = xm[order_m]
        ym = ym[order_m]

        uniq_m = np.unique(xm)
        n_obs_m = int(len(xm))
        max_feasible_degree = int(max(0, min(int(max_memory_degree), int(uniq_m.size) - 1, n_obs_m - 1)))
        if isinstance(memory_degree, str):
            if memory_degree != "auto":
                raise ValueError("memory_degree must be an int or 'auto'.")
            candidates = list(range(0, max_feasible_degree + 1))
            if len(candidates) == 0:
                candidates = [0]
            best_degree = 0
            best_aic = float("inf")
            for d in candidates:
                c_try = np.polyfit(xm, ym, deg=int(d))
                y_try = np.polyval(c_try, xm)
                rss = float(np.sum((ym - y_try) ** 2))
                k = int(d) + 1
                rss_use = max(rss, 1e-12)
                aic = float(n_obs_m) * math.log(rss_use / max(1, n_obs_m)) + 2.0 * float(k)
                if aic < best_aic:
                    best_aic = aic
                    best_degree = int(d)
            deg_mem = int(best_degree)
        else:
            deg_mem = int(memory_degree)
            if deg_mem < 0:
                raise ValueError(f"memory_degree must be >= 0, got {deg_mem}")
            if deg_mem > max_feasible_degree:
                raise ValueError(
                    f"Requested memory_degree={deg_mem} is not feasible for filtered data; "
                    f"max feasible degree is {max_feasible_degree} "
                    f"(num_unique_N={int(uniq_m.size)}, num_points={n_obs_m})."
                )
        coef_mem = np.polyfit(xm, ym, deg=deg_mem)
        mem_est = float(np.polyval(coef_mem, float(N)))
        model_pt_path = None

    x = np.asarray(x_vals, dtype=np.float64)
    t = np.asarray(t_vals, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    t = t[order]

    uniq_n = np.unique(x)
    deg_t = int(max(1, min(int(time_degree), int(uniq_n.size) - 1, int(len(x)) - 1)))
    coef_time = np.polyfit(x, t, deg=deg_t)
    time_est = float(np.polyval(coef_time, float(N)))

    return {
        "N": float(N),
        "memory_gib_est": float(mem_est),
        "memory_degree": int(deg_mem),
        "memory_coef": [float(c) for c in np.asarray(coef_mem).tolist()],
        "time_s_est": float(time_est),
        "time_degree": int(deg_t),
        "time_coef": [float(c) for c in np.asarray(coef_time).tolist()],
        "model_pt_path": model_pt_path,
        "profile_path": str(p_in),
        "filter": dict(filter),
    }


def best_seq_length(
    profile_path: str | Path,
    filter: Dict[str, Any],
) -> float:
    """
    Fit time model y(x) = a*x^2 + b*x + c from CSV rows (after filter),
    then return x>0 that maximizes x / y(x).

    For y(x)=a*x^2+b*x+c, derivative of r(x)=x/y(x):
        r'(x) = (c - a*x^2) / y(x)^2
    so the positive critical point is x*=sqrt(c/a) (when a>0, c>0).
    """
    p_in = Path(profile_path).expanduser().resolve()
    if (not p_in.exists()) or (not p_in.is_file()):
        raise FileNotFoundError(f"profile_path not found: {p_in}")

    with p_in.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"profile_path has no rows: {p_in}")

    headers = set(rows[0].keys())
    for need_col in ("N", "time"):
        if need_col not in headers:
            raise ValueError(f"profile_path missing required column '{need_col}': {p_in}")
    for k in filter.keys():
        if k not in headers:
            raise ValueError(f"filter key '{k}' not in CSV headers: {sorted(headers)}")

    x_vals: list[float] = []
    t_vals: list[float] = []
    for row in rows:
        if not _row_match(row, filter):
            continue
        n = _to_num(row.get("N", ""))
        t = _to_num(row.get("time", ""))
        if n is None or t is None:
            continue
        x_vals.append(float(n))
        t_vals.append(float(t))

    if len(x_vals) < 3:
        raise ValueError(
            f"Need at least 3 filtered points for quadratic time fit, got {len(x_vals)}."
        )

    x = np.asarray(x_vals, dtype=np.float64)
    t = np.asarray(t_vals, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    t = t[order]

    coef_time = np.polyfit(x, t, deg=2)
    a, b, c = (float(coef_time[0]), float(coef_time[1]), float(coef_time[2]))
    print(a, b, c)


    if (not math.isfinite(a)) or (not math.isfinite(c)):
        raise ValueError(f"Invalid quadratic coefficients from time fit: {coef_time.tolist()}")
    if a <= 0.0 or c <= 0.0:
        raise ValueError(
            "No finite positive maximizer guaranteed for x/time_fit(x) "
            f"with fitted coefficients a={a}, c={c}. Require a>0 and c>0."
        )

    x_star = float(math.sqrt(c / a))
    y_star = float(a * x_star * x_star + b * x_star + c)
    if (not math.isfinite(y_star)) or y_star <= 0.0:
        raise ValueError(
            f"Invalid fitted time at candidate optimum x={x_star}: y={y_star} "
            f"(coef={coef_time.tolist()})."
        )
    return x_star
