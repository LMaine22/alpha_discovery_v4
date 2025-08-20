# alpha_discovery/features/registry.py

import pandas as pd
import itertools
from typing import Dict, Callable

from ..config import settings
from . import core as fcore  # fcore for "feature core"


def _get_series(df: pd.DataFrame, ticker: str, column: str) -> pd.Series:
    """A robust helper to extract a single ticker/column series from the main dataframe."""
    col_name = f"{ticker}_{column}"
    if col_name in df.columns:
        return df[col_name]
    # Return an empty series with the same index if the column is not found
    return pd.Series(index=df.index, dtype=float)


# ===================================================================
# FEATURE DEFINITION REGISTRY
# This dictionary is the "DAG" your professor mentioned.
# Each entry defines HOW to build a feature.
# - key: A descriptive name for the feature calculation.
# - value: A lambda function that takes the full dataframe and relevant
#          ticker(s) and returns the calculated feature Series.
# ===================================================================

FEATURE_SPECS: Dict[str, Callable] = {
    # --- Single-Asset Momentum & Volatility Features ---
    "px_zscore_30d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PX_LAST"), window=30
    ),
    "px_zscore_90d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PX_LAST"), window=90
    ),
    "vol_turnover_zscore_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "TURNOVER"), window=60
    ),
    "realized_vol_21d": lambda df, t: fcore.get_realized_vol(
        _get_series(df, t, "PX_LAST"), window=21
    ),
    "frac_diff_px": lambda df, t: fcore.frac_diff(
        _get_series(df, t, "PX_LAST"), d=0.5, window=100
    ),

    # --- Single-Asset Options-Based Features ---
    "put_call_ratio_zscore_20d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PUT_CALL_VOLUME_RATIO_CUR_DAY"), window=20
    ),
    "ivol_zscore_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "IVOL"), window=60
    ),

    # --- Cross-Asset Relational Features ---
    # Note: These specs take two tickers, t1 and t2
    "corr_px_20d": lambda df, t1, t2: fcore.rolling_corr_fisher(
        _get_series(df, t1, "PX_LAST").pct_change(),
        _get_series(df, t2, "PX_LAST").pct_change(),
        window=20
    )[0],  # [0] returns the correlation, [1] would be the fisher transform

    "beta_px_60d": lambda df, t1, t2: fcore.rolling_beta(
        _get_series(df, t1, "PX_LAST").pct_change(),
        _get_series(df, t2, "PX_LAST").pct_change(),
        window=60
    ),
}


# ===================================================================
# THE FEATURE BUILDER
# This is the main function that orchestrates the feature creation.
# ===================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a complete matrix of lagged features for all tickers.

    This function iterates through all defined feature specs and tickers,
    calculates each feature, and applies the crucial shift(1) to prevent
    lookahead bias before adding it to the final dataframe.
    """
    print("Starting feature matrix construction...")
    all_features = {}
    all_tickers = settings.data.tradable_tickers + settings.data.macro_tickers

    # --- Build Single-Asset Features ---
    for ticker in all_tickers:
        for spec_name, spec_func in FEATURE_SPECS.items():
            # Check if the function is for single-asset features (takes df, t)
            if spec_func.__code__.co_argcount == 2:
                feature_name = f"{ticker}_{spec_name}"
                try:
                    # Calculate the raw feature
                    raw_feature = spec_func(df, ticker)
                    # ️ CRITICAL: Apply lag to prevent lookahead bias
                    all_features[feature_name] = raw_feature.shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(all_features)} single-asset features.")

    # --- Build Cross-Asset Features ---
    # Use SPY as the benchmark for beta calculations
    benchmark_ticker = 'SPY US Equity'
    cross_asset_features = {}

    for ticker in all_tickers:
        if ticker == benchmark_ticker:
            continue

        for spec_name, spec_func in FEATURE_SPECS.items():
            # Check if the function is for cross-asset features (takes df, t1, t2)
            if spec_func.__code__.co_argcount == 3:
                feature_name = f"{ticker}_vs_{benchmark_ticker}_{spec_name}"
                try:
                    # Calculate the raw feature
                    raw_feature = spec_func(df, ticker, benchmark_ticker)
                    #  CRITICAL: Apply lag to prevent lookahead bias
                    cross_asset_features[feature_name] = raw_feature.shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(cross_asset_features)} cross-asset features.")

    # Combine all features into a single dataframe
    all_features.update(cross_asset_features)
    feature_matrix = pd.DataFrame(all_features)

    print(f" Feature matrix construction complete. Shape: {feature_matrix.shape}")
    return feature_matrix