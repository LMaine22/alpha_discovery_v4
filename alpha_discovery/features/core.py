# alpha_discovery/features/core.py

import numpy as np
import pandas as pd
from typing import Tuple

# A small constant to prevent division by zero
EPSILON = 1e-9


# ===================================
# Section 1: Normalization Functions
# ===================================

def zscore_rolling(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates the rolling z-score of a series.
    The z-score measures how many standard deviations an element is from the mean.
    """
    if min_periods is None:
        min_periods = window // 2

    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()

    return (series - mean) / (std + EPSILON)


def mad_z_rolling(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates a z-score using the Median Absolute Deviation (MAD).
    This is more robust to outliers than a standard z-score.
    """
    if min_periods is None:
        min_periods = window // 2

    # The constant 1.4826 scales the MAD to be comparable to the standard deviation
    # for a normal distribution.
    c = 1.4826

    median = series.rolling(window, min_periods=min_periods).median()
    mad = (series - median).abs().rolling(window, min_periods=min_periods).median()

    return (series - median) / (c * mad + EPSILON)


# ===================================
# Section 2: Relational Functions
# ===================================

def align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Aligns two series by their index and drops any rows with NaNs in either.
    Ensures that calculations like correlation or beta are performed on a common set of dates.
    """
    df = pd.concat([s1, s2], axis=1, join='inner').dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def rolling_corr_fisher(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> Tuple[
    pd.Series, pd.Series]:
    """
    Calculates the rolling correlation and its Fisher transformation.
    The Fisher transform helps normalize the correlation's distribution, making it
    more suitable for subsequent z-scoring.
    """
    if min_periods is None:
        min_periods = window // 2

    s1_aligned, s2_aligned = align_series(s1, s2)

    # Calculate rolling correlation, clipped to avoid issues with log(0)
    corr = s1_aligned.rolling(window, min_periods=min_periods).corr(s2_aligned).clip(-0.9999, 0.9999)

    # Apply the Fisher transformation
    fisher_transform = 0.5 * np.log((1 + corr) / (1 - corr))

    return corr, fisher_transform


def rolling_beta(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates the rolling beta of series s1 with respect to series s2.
    Beta = Cov(s1, s2) / Var(s2)
    """
    if min_periods is None:
        min_periods = window // 2

    s1_aligned, s2_aligned = align_series(s1, s2)

    covariance = s1_aligned.rolling(window, min_periods=min_periods).cov(s2_aligned)
    variance = s2_aligned.rolling(window, min_periods=min_periods).var()

    return covariance / (variance + EPSILON)


# ===================================
# Section 3: Time Series Functions
# ===================================

def get_realized_vol(price_series: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculates the annualized realized volatility over a rolling window.
    Based on the standard deviation of daily log returns.
    """
    log_returns = np.log(price_series / price_series.shift(1))

    # Multiply by sqrt(252) to annualize the daily standard deviation
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
    return realized_vol


def frac_diff(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """
    Computes fractional differentiation of a time series.
    This helps to make the series stationary while preserving more memory
    than traditional integer differencing.

    Args:
        series (pd.Series): The input time series.
        d (float): The order of differentiation, typically between [0, 1].
        window (int): The lookback window to compute weights. A larger window
                      provides a more accurate but slower calculation.
    """
    # Calculate weights
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])  # Reverse weights for dot product

    # Apply weights
    output = series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)
    return output