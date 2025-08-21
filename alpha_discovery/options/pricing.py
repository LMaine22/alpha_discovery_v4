# alpha_discovery/options/pricing.py
"""
Black–Scholes pricing helpers and IV retrieval for the options simulation.

This module provides:
- Closed-form Black–Scholes (European) call/put pricing with robust edge handling.
- Utilities to fetch entry/exit implied vol for a given ticker/date/direction
  from the master dataframe using the 3M IV series you maintain:
    * {ticker}_3MO_CALL_IMP_VOL
    * {ticker}_3MO_PUT_IMP_VOL
- A convenience function to compute entry/exit premiums for a trade given:
  S0, S1, K, T0, horizon (days), r, direction, and entry/exit IVs.

Design choices (matching the project plan):
- Tenor: default 0.25y (~63 bd) is set at the callsite via config.
- Exit: T_exit = max(T0 - h/252, EPS_T).
- IV: sticky tenor. Exit IV attempts to read the same 3M series on the exit date;
       if missing, fallback to the entry IV.
- Rate: default r = 0.0 (config-controlled). If options.risk_free_rate_mode == "macro",
        we attempt to read a macro short-rate series; else we fall back to constant_r.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd

from ..config import settings


# =========================
# Numeric helpers & constants
# =========================

EPS_SIGMA = 1e-9   # floor for volatility
EPS_T = 1e-6       # floor for time to expiry (in years)
CLAMP_SIGMA_MIN = 0.0002  # 2% annualized vol lower clamp for safety
CLAMP_SIGMA_MAX = 5.0     # 500% annualized vol upper clamp for safety


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    # 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ensure_float(x: Optional[float]) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if np.isnan(xf):
            return None
        return xf
    except Exception:
        return None


def _clamp_sigma(sig: Optional[float]) -> Optional[float]:
    if sig is None:
        return None
    return float(min(max(sig, CLAMP_SIGMA_MIN), CLAMP_SIGMA_MAX))


def _intrinsic_value(S: float, K: float, option_type: Literal["call", "put"]) -> float:
    if option_type == "call":
        return max(S - K, 0.0)
    else:
        return max(K - S, 0.0)


# =========================
# Black–Scholes (European)
# =========================

def bs_price_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    European call option price under Black–Scholes with continuous dividend yield q.
    Robust to tiny sigma/T by returning (discounted) intrinsic in the limit.
    """
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)

    if S <= 0.0 or K <= 0.0:
        return 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    return S * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2


def bs_price_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    European put option price under Black–Scholes with continuous dividend yield q.
    Robust to tiny sigma/T by returning (discounted) intrinsic in the limit.
    """
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)

    if S <= 0.0 or K <= 0.0:
        return 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nnd1 = _norm_cdf(-d1)
    Nnd2 = _norm_cdf(-d2)

    return K * math.exp(-r * T) * Nnd2 - S * math.exp(-q * T) * Nnd1


# =========================
# IV retrieval utilities
# =========================

def _iv_column_name(direction: Literal["long", "short"]) -> str:
    """
    Map trade direction to the appropriate 3M IV column suffix.
      long  -> calls
      short -> puts
    """
    return "3MO_CALL_IMP_VOL" if direction == "long" else "3MO_PUT_IMP_VOL"


def _col(df: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    """
    Build '{ticker}_{field}' and return the Series if present.
    """
    name = f"{ticker}_{field}"
    if name in df.columns:
        return df[name]
    return None


def _value_at_or_pad(series: pd.Series, when: pd.Timestamp) -> Optional[float]:
    """
    Read value at index==when; if missing, pad with last available prior value (no lookahead).
    Returns None if no prior value exists.
    """
    if series is None or series.empty:
        return None
    try:
        # Exact hit
        if when in series.index:
            v = series.loc[when]
            return _ensure_float(v)
        # Pad (asof): last known prior (no lookahead)
        s = series.dropna()
        if s.empty:
            return None
        # Using asof requires sorted index
        try:
            v = s.asof(when)  # type: ignore[attr-defined]
            return _ensure_float(v)
        except Exception:
            # Fallback manual pad
            s = s.loc[s.index <= when]
            if s.empty:
                return None
            return _ensure_float(s.iloc[-1])
    except Exception:
        return None


def get_entry_iv(ticker: str, date: pd.Timestamp, direction: Literal["long", "short"], df: pd.DataFrame) -> Optional[float]:
    """
    Retrieve entry IV for the ticker on 'date' from the appropriate 3M IV series.
    Returns a clamped float or None if unavailable.
    """
    iv_suffix = _iv_column_name(direction)
    series = _col(df, ticker, iv_suffix)
    iv = _value_at_or_pad(series, date)
    return _clamp_sigma(iv)


def get_exit_iv(
    ticker: str,
    exit_date: pd.Timestamp,
    direction: Literal["long", "short"],
    df: pd.DataFrame,
    fallback_sigma: Optional[float]
) -> Optional[float]:
    """
    Retrieve exit IV for the ticker on 'exit_date'.
    If missing, use fallback_sigma (typically entry IV). Returns clamped value or None.
    """
    iv_suffix = _iv_column_name(direction)
    series = _col(df, ticker, iv_suffix)
    iv = _value_at_or_pad(series, exit_date)
    if iv is None:
        iv = fallback_sigma
    return _clamp_sigma(iv)


# =========================
# Risk-free rate utilities
# =========================

def get_risk_free_rate(date: pd.Timestamp, df: Optional[pd.DataFrame] = None) -> float:
    """
    Return the risk-free rate 'r' to use in pricing.

    Modes:
      - "constant": always returns settings.options.constant_r
      - "macro": attempts to read a short-tenor rate from df, else falls back to constant.

    Macro heuristic (lightweight, optional):
      Try in order (annualized decimals expected, e.g., 0.05 for 5%):
        * 'USGG3M Index'  -> 3M UST
        * 'USGG6M Index'  -> 6M UST
        * 'USGG1YR Index' -> 1Y UST
        * 'USGG2YR Index' -> 2Y UST
    """
    mode = settings.options.risk_free_rate_mode
    if mode == "constant" or df is None:
        return float(settings.options.constant_r)

    candidates = [
        "USGG3M Index", "USGG6M Index", "USGG1YR Index", "USGG2YR Index"
    ]
    for tk in candidates:
        s = _col(df, tk, "PX_LAST")
        if s is None:
            # Try the convention where macro indices are stored without suffix
            if df is not None and tk in df.columns:
                s = df[tk]
        if s is None:
            continue
        val = _value_at_or_pad(s, date)
        if val is not None:
            # Convert percentage to decimal if it looks > 1 (e.g., 5.0 -> 0.05)
            r = float(val)
            if r > 1.0:
                r = r / 100.0
            return r

    return float(settings.options.constant_r)


# =========================
# Entry/Exit pricing wrapper
# =========================

@dataclass
class PricedLeg:
    """Container for entry/exit option pricing results."""
    entry_price: float
    exit_price: float
    T_exit: float
    entry_iv: float
    exit_iv: float
    option_type: Literal["call", "put"]


def price_entry_exit(
    S0: float,
    S1: float,
    K: float,
    T0: float,
    h_days: int,
    r: float,
    direction: Literal["long", "short"],
    entry_sigma: Optional[float],
    exit_sigma: Optional[float],
    q: float = 0.0
) -> Optional[PricedLeg]:
    """
    Price entry and exit premiums for a trade.

    Parameters
    ----------
    S0 : float
        Underlying at entry (trigger date).
    S1 : float
        Underlying at exit (trigger date + horizon h).
    K : float
        Strike. For ATM, K = S0.
    T0 : float
        Initial time to expiry (years). e.g., 63/252.
    h_days : int
        Holding period in business days.
    r : float
        Risk-free rate (annualized, decimal).
    direction : "long" | "short"
        Long -> buy call, Short -> buy put.
    entry_sigma : Optional[float]
        Entry IV to use (clamped). If None -> cannot price robustly.
    exit_sigma : Optional[float]
        Exit IV to use (clamped). If None -> fallback to entry_sigma.
    q : float
        Continuous dividend yield (optional, default 0).

    Returns
    -------
    PricedLeg | None
    """
    S0 = _ensure_float(S0)
    S1 = _ensure_float(S1)
    K = _ensure_float(K)
    T0 = _ensure_float(T0)
    if None in (S0, S1, K, T0):
        return None

    T1 = max(T0 - float(h_days) / 252.0, EPS_T)

    entry_sigma = _clamp_sigma(entry_sigma)
    exit_sigma = _clamp_sigma(exit_sigma if exit_sigma is not None else entry_sigma)

    if entry_sigma is None or exit_sigma is None:
        return None

    option_type: Literal["call", "put"] = "call" if direction == "long" else "put"

    if option_type == "call":
        p0 = bs_price_call(S0, K, T0, r, entry_sigma, q=q)
        p1 = bs_price_call(S1, K, T1, r, exit_sigma, q=q)
    else:
        p0 = bs_price_put(S0, K, T0, r, entry_sigma, q=q)
        p1 = bs_price_put(S1, K, T1, r, exit_sigma, q=q)

    # Basic sanity clamps: negative numeric noise should not appear.
    p0 = max(0.0, float(p0))
    p1 = max(0.0, float(p1))

    return PricedLeg(
        entry_price=p0,
        exit_price=p1,
        T_exit=T1,
        entry_iv=float(entry_sigma),
        exit_iv=float(exit_sigma),
        option_type=option_type,
    )


# =========================
# Convenience checks
# =========================

def has_required_iv_series(ticker: str, df: pd.DataFrame) -> bool:
    """
    Check whether both 3M call/put IV series exist for the given ticker.
    This helps the caller decide to skip non-optionable/insufficient IV names.
    """
    call_iv = f"{ticker}_3MO_CALL_IMP_VOL"
    put_iv = f"{ticker}_3MO_PUT_IMP_VOL"
    return (call_iv in df.columns) and (put_iv in df.columns)


def get_underlying_price(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    """
    Retrieve underlying close (PX_LAST) for ticker at 'date' (pad back if needed).
    """
    s = _col(df, ticker, "PX_LAST")
    if s is None:
        return None
    return _value_at_or_pad(s, date)


def get_price_on_exit(ticker: str, exit_date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    """
    Retrieve underlying close (PX_LAST) for ticker at exit_date (pad back if needed).
    """
    return get_underlying_price(ticker, exit_date, df)
