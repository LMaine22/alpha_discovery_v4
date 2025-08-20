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
# Each entry defines HOW to build a feature.
# - key: A descriptive feature name.
# - value: lambda(df, t) or lambda(df, t1, t2) -> pd.Series
# All features are lagged by 1 in build_feature_matrix to avoid lookahead.
# ===================================================================

FEATURE_SPECS: Dict[str, Callable] = {
    # =========================
    # A) CORE (Kept)
    # =========================
    "px_zscore_90d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PX_LAST"), window=90
    ),
    "realized_vol_21d": lambda df, t: fcore.get_realized_vol(
        _get_series(df, t, "PX_LAST"), window=21
    ),
    "news_pub_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "NEWS_PUBLICATION_COUNT"), 30
    ),
    "news_heat_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "NEWS_HEAT_READ_DMAX"), 30
    ),
    "tw_pub_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "TWITTER_PUBLICATION_COUNT"), 30
    ),
    "tw_sent_avg_dev_7d": lambda df, t: fcore.deviation_from_mean(
        _get_series(df, t, "TWITTER_SENTIMENT_DAILY_AVG"), 7
    ),
    # We keep turnover z as a liquidity state variable (your original)
    "vol_turnover_zscore_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "TURNOVER"), window=60
    ),

    # =========================
    # B) PRICE / MICROSTRUCTURE (New)
    # =========================
    "px_trend_5_20_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.momentum_trend(_get_series(df, t, "PX_LAST"), 5, 20), window=60
    ),
    "px_range_polarity_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.range_polarity(
            _get_series(df, t, "PX_LAST"),
            _get_series(df, t, "PX_LOW"),
            _get_series(df, t, "PX_HIGH"),
        ),
        window=60,
    ),
    "turnover_minus_volume_z_60d": lambda df, t: (
        fcore.zscore_rolling(_get_series(df, t, "TURNOVER"), 60)
        - fcore.zscore_rolling(_get_series(df, t, "PX_VOLUME"), 60)
    ),

    # =========================
    # C) DERIVATIVES / VOL (IVOL-FREE)
    # =========================
    # 3M call IV shock
    "iv_call3m_shock_1d_z_30d": lambda df, t: fcore.zscore_rolling(
        fcore.diff_n(_get_series(df, t, "3MO_CALL_IMP_VOL"), 1), window=30
    ),
    # Term premium proxy: 3M call IV - realized vol(21d)
    "iv_term_call3m_minus_rv21_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_CALL_IMP_VOL") - fcore.get_realized_vol(_get_series(df, t, "PX_LAST"), 21),
        window=60,
    ),
    # Skew: 3M put IV - 3M call IV
    "iv_skew_put_minus_call_3mo_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_PUT_IMP_VOL") - _get_series(df, t, "3MO_CALL_IMP_VOL"),
        window=60,
    ),
    # Moneyness tilt: 3M call IV vs IVOL_MONEYNESS
    "moneyness_tilt_call3m_minus_moneyness_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_CALL_IMP_VOL") - _get_series(df, t, "IVOL_MONEYNESS"),
        window=60,
    ),
    # IV vs News heat divergence (using 3M call IV as IV proxy)
    "iv_vs_newsheat_divergence_call3m_z_60d": lambda df, t: (
        fcore.zscore_rolling(_get_series(df, t, "3MO_CALL_IMP_VOL"), 60)
        - fcore.zscore_rolling(_get_series(df, t, "NEWS_HEAT_READ_DMAX"), 60)
    ),
    # Options total activity spike
    "options_total_volume_spike_z_30d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "TOT_OPT_VOLUME_CUR_DAY"), window=30
    ),
    # Options vs equity flow ratio
    "options_vs_equity_flow_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            _get_series(df, t, "TOT_OPT_VOLUME_CUR_DAY"),
            _get_series(df, t, "PX_VOLUME"),
        ),
        window=60,
    ),
    # Call/Put OI skew (positioning tilt)
    "oi_callput_skew_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            (_get_series(df, t, "OPEN_INT_TOTAL_CALL") - _get_series(df, t, "OPEN_INT_TOTAL_PUT")),
            (_get_series(df, t, "OPEN_INT_TOTAL_CALL") + _get_series(df, t, "OPEN_INT_TOTAL_PUT")),
        ),
        window=60,
    ),
    # Call OI 3d pct change
    "open_interest_call_pctchg_3d_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.pct_change_n(_get_series(df, t, "OPEN_INT_TOTAL_CALL"), 3), window=60
    ),

    # =========================
    # D) SENTIMENT / NEWS COMPOSITES (New)
    # =========================
    # Pos vs Neg balance on Twitter counts
    "tw_posneg_balance_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            (_get_series(df, t, "TWITTER_POS_SENTIMENT_COUNT") - _get_series(df, t, "TWITTER_NEG_SENTIMENT_COUNT")),
            (_get_series(df, t, "TWITTER_POS_SENTIMENT_COUNT") + _get_series(df, t, "TWITTER_NEG_SENTIMENT_COUNT")),
        ),
        window=60,
    ),
    # Price-vs-Sentiment divergence (7d momentum vs 7d sentiment change)
    "px_vs_sentiment_divergence_z_60d": lambda df, t: fcore.zscore_rolling(
        (
            fcore.pct_change_n(_get_series(df, t, "PX_LAST"), 7)
            - fcore.diff_n(_get_series(df, t, "TWITTER_SENTIMENT_DAILY_AVG"), 7)
        ),
        window=60,
    ),

    # =========================
    # E) FLOW / PCR (Replacement)
    # =========================
    "pcr_ema5_z_30d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PUT_CALL_VOLUME_RATIO_CUR_DAY").ewm(span=5, min_periods=2).mean(),
        window=30,
    ),

    # =========================
    # F) REGIME
    # =========================
    "vol90d_regime_z_120d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "VOLATILITY_90D"), window=120
    ),

    # =========================
    # G) CROSS-ASSET (vs SPY): Fisher-z corr + deltas, and z-scored beta
    # =========================
    "corr_px_fisher20_z_60d": lambda df, t1, t2: fcore.zscore_rolling(
        fcore.rolling_corr_fisher(
            _get_series(df, t1, "PX_LAST").pct_change(),
            _get_series(df, t2, "PX_LAST").pct_change(),
            window=20,
        )[1],
        window=60,
    ),
    "corr_delta_fisher_20_60_z_60d": lambda df, t1, t2: fcore.zscore_rolling(
        (
            fcore.rolling_corr_fisher(
                _get_series(df, t1, "PX_LAST").pct_change(),
                _get_series(df, t2, "PX_LAST").pct_change(),
                window=20,
            )[1]
            - fcore.rolling_corr_fisher(
                _get_series(df, t1, "PX_LAST").pct_change(),
                _get_series(df, t2, "PX_LAST").pct_change(),
                window=60,
            )[1]
        ),
        window=60,
    ),
    "beta_px_60d_z_120d": lambda df, t1, t2: fcore.zscore_rolling(
        fcore.rolling_beta(
            _get_series(df, t1, "PX_LAST").pct_change(),
            _get_series(df, t2, "PX_LAST").pct_change(),
            window=60,
        ),
        window=120,
    ),
}


# ===================================================================
# THE FEATURE BUILDER
# ===================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a complete matrix of lagged features for all tickers.

    Iterates through all defined feature specs and tickers, calculates each
    feature, and applies shift(1) to prevent lookahead bias.
    """
    print("Starting feature matrix construction...")
    all_features = {}
    all_tickers = settings.data.tradable_tickers + settings.data.macro_tickers

    # --- Build Single-Asset Features ---
    for ticker in all_tickers:
        for spec_name, spec_func in FEATURE_SPECS.items():
            if spec_func.__code__.co_argcount == 2:  # (df, t)
                feature_name = f"{ticker}_{spec_name}"
                try:
                    raw_feature = spec_func(df, ticker)
                    all_features[feature_name] = raw_feature.shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(all_features)} single-asset features.")

    # --- Build Cross-Asset Features (vs benchmark) ---
    benchmark_ticker = 'SPY US Equity'
    cross_asset_features = {}

    for ticker in all_tickers:
        if ticker == benchmark_ticker:
            continue
        for spec_name, spec_func in FEATURE_SPECS.items():
            if spec_func.__code__.co_argcount == 3:  # (df, t1, t2)
                feature_name = f"{ticker}_vs_{benchmark_ticker}_{spec_name}"
                try:
                    raw_feature = spec_func(df, ticker, benchmark_ticker)
                    cross_asset_features[feature_name] = raw_feature.shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(cross_asset_features)} cross-asset features.")

    # Combine, prune all-NaN columns
    all_features.update(cross_asset_features)
    feature_matrix = pd.DataFrame(all_features).dropna(how="all", axis=1)

    print(f" Feature matrix construction complete. Shape: {feature_matrix.shape}")
    return feature_matrix
