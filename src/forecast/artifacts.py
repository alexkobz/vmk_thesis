from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.forecast.config import ForecastConfig


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_forecaster(forecaster, artifacts_dir: str | Path, filename: str = "forecaster.joblib") -> Path:
    out_dir = _ensure_dir(artifacts_dir)
    path = out_dir / filename
    joblib.dump(forecaster, path)
    return path


def save_run_artifacts(
    result: dict[str, Any],
    cfg: ForecastConfig,
    artifacts_dir: str | Path,
) -> dict[str, Path]:
    out_dir = _ensure_dir(artifacts_dir)
    saved: dict[str, Path] = {}

    metrics_path = out_dir / "metrics.json"
    cfg_dict = asdict(cfg)
    cfg_dict["estimator"] = _serialize_estimator(cfg.estimator)

    payload = {
        "scores": result.get("scores", []),
        "mean_score": result.get("mean_score"),
        "std_score": result.get("std_score"),
        "config": cfg_dict,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    saved["metrics"] = metrics_path

    mape = result.get("last_mape_by_secid")
    if isinstance(mape, pd.Series):
        mape_path = out_dir / "mape_by_secid.csv"
        mape.to_csv(mape_path, header=["mape"])
        saved["mape_by_secid"] = mape_path

    return saved


def _serialize_estimator(estimator) -> dict[str, Any] | None:
    if estimator is None:
        return None
    info: dict[str, Any] = {
        "class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
    }
    if hasattr(estimator, "get_params"):
        try:
            info["params"] = estimator.get_params(deep=True)
        except TypeError:
            info["params"] = estimator.get_params()
    else:
        info["repr"] = repr(estimator)
    return info
