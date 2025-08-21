# alpha_discovery/eval/selection.py
"""
Selection helpers for options-based evaluation.

This module implements the "best-of" selection logic:
  1) For each ticker, evaluate all horizons on the options ledger.
  2) Choose the best horizon per ticker using a primary metric (Sharpe LB by default),
     with tie-breakers (Omega, then Support).
  3) Rank tickers by that best-horizon score and select up to top-N tickers,
     enforcing minimum support and minimum distinct ticker count.
  4) Filter the ledger to the selected (ticker, best_horizon) pairs,
     construct a daily portfolio return series (by trigger_date), and compute metrics.

Inputs
------
- The "options ledger" is produced by engine.backtester.run_setup_backtest_options(...)
  and must include at least:
    ['trigger_date', 'ticker', 'horizon_days', 'pnl_pct']

Outputs
-------
- Best-per-ticker mapping: {ticker: {'horizon': int, 'metrics': {...}}}
- Selection list: List[{'ticker': str, 'horizon': int}]
- Filtered ledger: only rows for the selected (ticker, horizon) pairs
- Daily returns series: pd.Series indexed by trigger_date
- Portfolio metrics dict: aligned with eval.metrics keys
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd

from ..config import settings
from . import metrics as M


# =========================
# Internal utilities
# =========================

def _daily_returns_for_pair(ledger: pd.DataFrame, ticker: str, horizon: int) -> pd.Series:
    """
    Build a daily return series (by trigger_date) for a (ticker, horizon) pair.
    We aggregate by mean of pnl_pct across concurrent trades on the same date.
    """
    sub = ledger[(ledger["ticker"] == ticker) & (ledger["horizon_days"] == horizon)]
    if sub.empty:
        return pd.Series(dtype=float)

    # Group by trigger_date and average pnl_pct (consistent with project convention)
    s = sub.groupby("trigger_date")["pnl_pct"].mean().sort_index()
    # Ensure a clean float Series
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _compute_metrics_from_returns(daily_returns: pd.Series) -> Dict[str, float]:
    """
    Compute metrics consistent with eval.metrics.calculate_all_metrics, but starting
    from a daily returns series rather than a trade_ledger.
    """
    if daily_returns is None or daily_returns.empty:
        return {}

    cleaned = M.winsorize(daily_returns)
    if cleaned.empty:
        return {}

    support = float(len(cleaned))  # align with prior convention: support = # of non-empty days
    sharpe_stats = M.block_bootstrap_sharpe(cleaned)
    omega = M.calculate_omega_ratio(cleaned)
    max_dd = M.calculate_max_drawdown(cleaned)

    # Annualized return (if enough data), matching prior convention
    min_days_for_annualization = 126
    if len(cleaned) >= min_days_for_annualization:
        num_years = len(cleaned) / 252.0
        total_return = (1.0 + cleaned).prod() - 1.0
        annualized_return = (1.0 + total_return) ** (1.0 / num_years) - 1.0
    else:
        annualized_return = 0.0

    out = {
        "support": support,
        "annualized_return": float(annualized_return),
        "volatility": float(cleaned.std() * np.sqrt(252.0)),
        "max_drawdown": float(max_dd),
        "omega_ratio": float(omega),
        "mean_return": float(cleaned.mean()),
    }
    # Merge Sharpe stats (sharpe_median, sharpe_lb, sharpe_ub)
    out.update({k: float(v) for k, v in sharpe_stats.items()})
    # Replace NaNs/infs defensively
    return {k: (0.0 if (not np.isfinite(v)) else float(v)) for k, v in out.items()}


def _metric_key(metrics: Dict[str, float], name: str) -> float:
    """
    Safely pull a metric from dict; default to 0.0 if missing.
    """
    v = metrics.get(name, 0.0)
    if v is None or not np.isfinite(v):
        return 0.0
    return float(v)


def _compare_metric_tuple(metrics: Dict[str, float],
                          primary: str,
                          tiebreakers: List[str]) -> Tuple[float, ...]:
    """
    Build a sorting tuple (descending) for robust ranking:
      (primary, tie1, tie2, ...)
    """
    vals = [_metric_key(metrics, primary)]
    for tb in tiebreakers:
        vals.append(_metric_key(metrics, tb))
    return tuple(vals)


# =========================
# Public data structures
# =========================

@dataclass
class TickerBest:
    ticker: str
    horizon: int
    metrics: Dict[str, float]


# =========================
# Public API
# =========================

def score_ticker_horizon(ledger: pd.DataFrame, ticker: str, horizon: int) -> Dict[str, float]:
    """
    Score a single (ticker, horizon) pair using daily mean pnl_pct by trigger_date.
    Returns a metrics dict (Sharpe, Omega, Support, etc.). Empty dict if insufficient data.
    """
    returns = _daily_returns_for_pair(ledger, ticker, horizon)
    return _compute_metrics_from_returns(returns)


def select_best_horizon_per_ticker(
    ledger: pd.DataFrame,
    metric_primary: Optional[str] = None,
    metric_tiebreakers: Optional[List[str]] = None,
    min_support_per_ticker: Optional[int] = None,
) -> Dict[str, TickerBest]:
    """
    For each ticker, evaluate all horizons and select the best horizon by (primary + tiebreakers).
    Enforces minimum support per ticker (based on # of daily return points).
    """
    if metric_primary is None:
        metric_primary = settings.selection.metric_primary
    if metric_tiebreakers is None:
        metric_tiebreakers = settings.selection.metric_tiebreakers
    if min_support_per_ticker is None:
        min_support_per_ticker = settings.selection.min_support_per_ticker

    if ledger.empty:
        return {}

    best_by_ticker: Dict[str, TickerBest] = {}

    tickers = sorted(ledger["ticker"].dropna().unique().tolist())
    horizons = sorted(ledger["horizon_days"].dropna().unique().astype(int).tolist())

    for tk in tickers:
        best_tuple: Optional[Tuple[float, ...]] = None
        best_choice: Optional[TickerBest] = None

        for h in horizons:
            m = score_ticker_horizon(ledger, tk, h)
            if not m:
                continue

            # Enforce per-ticker support threshold
            if m.get("support", 0.0) < float(min_support_per_ticker):
                continue

            tup = _compare_metric_tuple(m, metric_primary, metric_tiebreakers)

            # We sort descending later; here just keep the max tuple
            if (best_tuple is None) or (tup > best_tuple):
                best_tuple = tup
                best_choice = TickerBest(ticker=tk, horizon=int(h), metrics=m)

        if best_choice is not None:
            best_by_ticker[tk] = best_choice

    return best_by_ticker


def rank_and_select_tickers(
    best_by_ticker: Dict[str, TickerBest],
    top_n: Optional[int] = None,
    min_distinct_tickers: Optional[int] = None,
    metric_primary: Optional[str] = None,
    metric_tiebreakers: Optional[List[str]] = None,
) -> List[TickerBest]:
    """
    Rank tickers by their best-horizon metrics and pick up to top_n.
    Returns a (possibly shorter) list if few tickers meet thresholds.
    """
    if top_n is None:
        top_n = settings.selection.top_n_tickers
    if min_distinct_tickers is None:
        min_distinct_tickers = settings.selection.min_distinct_tickers
    if metric_primary is None:
        metric_primary = settings.selection.metric_primary
    if metric_tiebreakers is None:
        metric_tiebreakers = settings.selection.metric_tiebreakers

    candidates = list(best_by_ticker.values())
    if not candidates:
        return []

    # Rank DESC by (primary, tiebreakers)
    candidates.sort(key=lambda x: _compare_metric_tuple(x.metrics, metric_primary, metric_tiebreakers), reverse=True)

    selected = candidates[: int(top_n)]
    # Enforce minimum distinct tickers (best effort): if fewer exist, return what we have.
    if len(selected) < int(min_distinct_tickers):
        # Not enough distinct eligible tickers; return as many as we have
        return selected

    return selected


def filter_ledger_to_selection(
    ledger: pd.DataFrame,
    selection: List[TickerBest]
) -> pd.DataFrame:
    """
    Keep only rows for (ticker, horizon) pairs present in selection.
    """
    if ledger.empty or not selection:
        return pd.DataFrame(columns=ledger.columns)

    pairs = {(sel.ticker, sel.horizon) for sel in selection}
    mask = ledger.apply(lambda r: (r["ticker"], int(r["horizon_days"])) in pairs, axis=1)
    out = ledger[mask].copy()
    out.sort_values(by=["trigger_date", "ticker", "horizon_days"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def portfolio_daily_returns(filtered_ledger: pd.DataFrame) -> pd.Series:
    """
    Build the daily portfolio return series by averaging pnl_pct by trigger_date
    across all selected trades.
    """
    if filtered_ledger.empty:
        return pd.Series(dtype=float)

    s = filtered_ledger.groupby("trigger_date")["pnl_pct"].mean().sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def portfolio_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """
    Compute metrics on the portfolio daily return series.
    """
    return _compute_metrics_from_returns(daily_returns)


def selection_summary(selection: List[TickerBest]) -> Dict[str, object]:
    """
    Produce compact metadata for reporting.
    """
    if not selection:
        return {"chosen_tickers": "", "chosen_horizons": {}}

    chosen_tickers = [x.ticker for x in selection]
    chosen_horizons = {x.ticker: x.horizon for x in selection}
    return {
        "chosen_tickers": ", ".join(chosen_tickers),
        "chosen_horizons": chosen_horizons,
    }
