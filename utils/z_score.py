"""
utils/z_score.py — Rolling Z-Score Normalization for Agent-B

This module is the SINGLE SOURCE OF TRUTH for all normalization logic.
It is imported by:
  - scripts/generate_training_data.py  (normalize the historical CSV)
  - The Kaggle training notebook        (normalize data during training)
  - The Modal daily pipeline            (normalize live data + reverse predictions)

The spec (Section 3.1, Quant Upgrade 2) says:
  "A Python script must calculate the Mean (μ) and Standard Deviation (σ)
   over a rolling 30-day window for both DXY and Brent. Each value in the
   90-day sequence is converted to a Z-Score (Z = (Price - μ) / σ)."
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 30,
    min_periods: int = 1,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute rolling Z-Score normalization.

    Z = (price - rolling_mean) / rolling_std

    Args:
        series: Raw price series (e.g. BZ=F Close or DXY Close)
        window: Rolling window size in trading days (default 30 per spec)
        min_periods: Minimum observations needed for the first values

    Returns:
        Tuple of (z_scores, rolling_means, rolling_stds)
        - z_scores: The normalized series
        - rolling_means: μ values (needed for reverse transform)
        - rolling_stds: σ values (needed for reverse transform)
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Prevent division by zero — if std is 0 or NaN, set to 1.0
    # This happens when all values in the window are identical (unlikely but safe)
    rolling_std = rolling_std.replace(0, 1.0).fillna(1.0)

    z_scores = (series - rolling_mean) / rolling_std

    return z_scores, rolling_mean, rolling_std


def reverse_zscore(z: float, mu: float, sigma: float) -> float:
    """
    Convert a Z-score prediction back to a dollar price.

    price = z * σ + μ

    This is used in post-processing after model inference to convert
    the model's Z-score output back to real dollar amounts.

    Args:
        z: The predicted Z-score from the model
        mu: The rolling mean (μ) from the last 30-day window
        sigma: The rolling std (σ) from the last 30-day window

    Returns:
        The real dollar price
    """
    return z * sigma + mu


def reverse_zscore_series(
    z_series: pd.Series,
    rolling_means: pd.Series,
    rolling_stds: pd.Series,
) -> pd.Series:
    """
    Batch reverse Z-score for an entire series.

    Args:
        z_series: Series of Z-scores
        rolling_means: Corresponding μ values
        rolling_stds: Corresponding σ values

    Returns:
        Series of real dollar prices
    """
    return z_series * rolling_stds + rolling_means


def save_scaler_params(
    rolling_means: pd.Series,
    rolling_stds: pd.Series,
    output_path: str | Path,
    feature_name: str = "brent",
) -> None:
    """
    Save the most recent rolling statistics for live inference.

    During live inference, we need the LAST μ and σ to normalize
    today's incoming price before feeding it to the model.
    This function saves those values to a JSON file.

    Args:
        rolling_means: Full series of rolling means
        rolling_stds: Full series of rolling stds
        output_path: Where to save the JSON file
        feature_name: Label for the feature (e.g. "brent", "dxy")
    """
    params = {
        f"{feature_name}_last_mu": float(rolling_means.iloc[-1]),
        f"{feature_name}_last_sigma": float(rolling_stds.iloc[-1]),
        f"{feature_name}_window": 30,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists, merge new params into it
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        existing.update(params)
        params = existing

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"  Saved {feature_name} scaler params to {output_path}")


def load_scaler_params(path: str | Path) -> dict:
    """
    Load saved scaler parameters for live inference.

    Args:
        path: Path to the JSON file saved by save_scaler_params()

    Returns:
        Dictionary with mu, sigma, and window values
    """
    with open(path, "r") as f:
        return json.load(f)
