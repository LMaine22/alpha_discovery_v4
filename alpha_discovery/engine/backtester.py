# alpha_discovery/engine/backtester.py

import pandas as pd
from typing import List, Dict, Optional

from ..config import settings
from ..options import pricing


# Define the forward-looking time horizons (in business days) to test
TRADE_HORIZONS_DAYS = [1, 3, 5, 10, 21]


def prepare_forward_returns(master_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Calculates forward returns for all tradable tickers over all horizons.
    This is an expensive calculation that should be run only once.
    """
    print("Pre-calculating forward returns for all tickers...")
    all_fwd_returns: Dict[int, pd.DataFrame] = {}

    price_cols = {
        ticker: f"{ticker}_PX_LAST"
        for ticker in settings.data.tradable_tickers
        if f"{ticker}_PX_LAST" in master_df.columns
    }

    if not price_cols:
        print(" Warning: No price columns found for tradable tickers.")
        return all_fwd_returns

    prices_df = master_df[list(price_cols.values())]

    for h in TRADE_HORIZONS_DAYS:
        fwd_returns = prices_df.pct_change(h).shift(-h)
        all_fwd_returns[h] = fwd_returns.rename(
            columns={v: k for k, v in price_cols.items()}
        )

    return all_fwd_returns


def run_setup_backtest(
        setup_signals: List[str],
        signals_df: pd.DataFrame,
        fwd_returns_by_horizon: Dict[int, pd.DataFrame]  # We now receive the returns directly
) -> pd.DataFrame:
    """
    Runs a backtest for a single setup using pre-calculated forward returns.
    """
    if not setup_signals:
        return pd.DataFrame()

    # --- Step 1: Find Trigger Dates ---
    trigger_mask = signals_df[setup_signals].all(axis=1)
    trigger_dates = trigger_mask[trigger_mask].index

    # --- Step 2: Check for Minimum Support ---
    if len(trigger_dates) < settings.validation.min_initial_support:
        return pd.DataFrame()

    # --- Step 3: Build the Trade Ledger ---
    trade_ledger_rows = []

    for horizon in TRADE_HORIZONS_DAYS:
        fwd_returns_df = fwd_returns_by_horizon.get(horizon)
        if fwd_returns_df is None or fwd_returns_df.empty:
            continue

        # Slice returns at trigger dates (inner join on dates)
        triggered_returns = fwd_returns_df.loc[fwd_returns_df.index.isin(trigger_dates)]
        if triggered_returns.empty:
            continue

        melted_returns = triggered_returns.melt(
            var_name='ticker',
            value_name='forward_return',
            ignore_index=False
        )
        melted_returns['horizon_days'] = horizon
        trade_ledger_rows.append(melted_returns.reset_index())

    if not trade_ledger_rows:
        return pd.DataFrame()

    trade_ledger = pd.concat(trade_ledger_rows, ignore_index=True)
    trade_ledger = trade_ledger.rename(columns={'Date': 'trigger_date'})
    trade_ledger = trade_ledger.dropna(subset=['forward_return'])

    return trade_ledger


# ======================================================================
# New: Options-Backed Trade Ledger (Black–Scholes, 3M tenor, sticky IV)
# ======================================================================

def _add_bdays(ts: pd.Timestamp, n_bdays: int) -> pd.Timestamp:
    """Add business days to a timestamp using pandas BDay."""
    return pd.Timestamp(ts) + pd.tseries.offsets.BDay(n_bdays)


def _has_iv_for_ticker(ticker: str, df: pd.DataFrame) -> bool:
    """Check whether both 3M call/put IV series exist (unless allow_nonoptionable=True)."""
    if settings.options.allow_nonoptionable:
        # If allowed, we don't enforce both series; we will still try to fetch what we can.
        return True
    return pricing.has_required_iv_series(ticker, df)


def _contracts_for_capital(entry_option_price: float) -> int:
    """Compute #contracts for fixed capital per trade."""
    cap = float(settings.options.capital_per_trade)
    mult = int(settings.options.contract_multiplier)
    if entry_option_price <= 0.0:
        return 0
    return int(cap // (entry_option_price * mult))


def run_setup_backtest_options(
    setup_signals: List[str],
    signals_df: pd.DataFrame,
    master_df: pd.DataFrame,
    direction: str  # "long" or "short"
) -> pd.DataFrame:
    """
    Build an options-based trade ledger for a given setup over all (ticker, horizon).

    For each trigger date (where all setup signals are True):
      - For each tradable ticker (with required IV series unless allowed):
        - For each horizon h in TRADE_HORIZONS_DAYS:
            * Entry at trigger date: price ATM option (K=S0) with T0=tenor_days/252
              using 3M IV (call for long / put for short).
            * Exit at trigger+h business days: reprice with T1=max(T0-h/252, ε),
              IV from same 3M series on exit date; fallback to entry IV if missing.
            * Contracts = floor( capital / (entry_price * multiplier) ), skip if <1.
            * Record full ledger row with strike, underlying entry/exit, IVs,
              entry/exit option prices, contracts, PnL$ and PnL%.

    Returns
    -------
    pd.DataFrame with columns:
      trigger_date, exit_date, ticker, horizon_days, direction, option_type,
      strike, entry_underlying, exit_underlying,
      entry_iv, exit_iv, entry_option_price, exit_option_price,
      contracts, capital_allocated, pnl_dollars, pnl_pct
    """
    if not setup_signals:
        return pd.DataFrame()

    # 1) Trigger dates where all signals in the setup fire
    try:
        trigger_mask = signals_df[setup_signals].all(axis=1)
    except KeyError as e:
        print(f" Error: Missing signals in signals_df: {e}")
        return pd.DataFrame()

    trigger_dates = trigger_mask[trigger_mask].index
    if len(trigger_dates) < settings.validation.min_initial_support:
        return pd.DataFrame()

    # 2) Prepare constants
    tenor_years = float(settings.options.tenor_days) / 252.0
    capital_alloc = float(settings.options.capital_per_trade)
    r_mode = settings.options.risk_free_rate_mode

    # 3) Iterate and build rows
    rows: List[Dict] = []

    for trigger_date in trigger_dates:
        # Risk-free rate evaluated at entry (constant mode returns constant_r)
        r = pricing.get_risk_free_rate(trigger_date, df=master_df if r_mode == "macro" else None)

        for ticker in settings.data.tradable_tickers:
            # Skip tickers without underlying price or IVs (unless allowed)
            if not _has_iv_for_ticker(ticker, master_df):
                continue

            # Entry underlying
            S0 = pricing.get_underlying_price(ticker, trigger_date, master_df)
            if S0 is None:
                continue

            # Strike is ATM at entry
            K = float(S0)

            # Entry IV
            entry_iv = pricing.get_entry_iv(ticker, trigger_date, direction, master_df)
            if entry_iv is None:
                # If strictly required IV missing and not allowed, skip
                if not settings.options.allow_nonoptionable:
                    continue

            for h in TRADE_HORIZONS_DAYS:
                # Exit date is trigger_date + h business days (target)
                exit_date_target = _add_bdays(trigger_date, h)

                # Exit underlying (asof pad back if exact index missing)
                S1 = pricing.get_price_on_exit(ticker, exit_date_target, master_df)
                if S1 is None:
                    continue

                # Exit IV (fallback to entry IV if missing)
                exit_iv = pricing.get_exit_iv(ticker, exit_date_target, direction, master_df, fallback_sigma=entry_iv)
                if exit_iv is None:
                    # If still None, skip
                    continue

                # Price entry/exit
                priced = pricing.price_entry_exit(
                    S0=S0,
                    S1=S1,
                    K=K,
                    T0=tenor_years,
                    h_days=h,
                    r=r,
                    direction="long" if direction == "long" else "short",
                    entry_sigma=entry_iv,
                    exit_sigma=exit_iv,
                    q=0.0
                )
                if priced is None:
                    continue

                # Sizing
                contracts = _contracts_for_capital(priced.entry_price)
                if contracts < 1:
                    continue

                pnl_dollars = contracts * (priced.exit_price - priced.entry_price) * settings.options.contract_multiplier
                pnl_pct = pnl_dollars / capital_alloc if capital_alloc > 0 else 0.0

                rows.append({
                    "trigger_date": pd.Timestamp(trigger_date),
                    "exit_date": pd.Timestamp(exit_date_target),
                    "ticker": ticker,
                    "horizon_days": h,
                    "direction": direction,
                    "option_type": priced.option_type,
                    "strike": float(K),
                    "entry_underlying": float(S0),
                    "exit_underlying": float(S1),
                    "entry_iv": float(priced.entry_iv),
                    "exit_iv": float(priced.exit_iv),
                    "entry_option_price": float(priced.entry_price),
                    "exit_option_price": float(priced.exit_price),
                    "contracts": int(contracts),
                    "capital_allocated": float(capital_alloc),
                    "pnl_dollars": float(pnl_dollars),
                    "pnl_pct": float(pnl_pct),
                })

    if not rows:
        return pd.DataFrame()

    ledger = pd.DataFrame(rows)
    # Sort for readability
    ledger = ledger.sort_values(by=["trigger_date", "ticker", "horizon_days"]).reset_index(drop=True)
    return ledger
