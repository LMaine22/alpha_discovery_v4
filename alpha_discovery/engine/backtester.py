# alpha_discovery/engine/backtester.py

import pandas as pd
from typing import List, Dict

from ..config import settings

# Define the forward-looking time horizons (in business days) to test
TRADE_HORIZONS_DAYS = [1, 3, 5, 10, 21]


def prepare_forward_returns(master_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Calculates forward returns for all tradable tickers over all horizons.
    This is an expensive calculation that should be run only once.
    """
    print("Pre-calculating forward returns for all tickers...")
    all_fwd_returns = {}

    price_cols = {
        ticker: f"{ticker}_PX_LAST"
        for ticker in settings.data.tradable_tickers
        if f"{ticker}_PX_LAST" in master_df.columns
    }

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
        fwd_returns_by_horizon: Dict[int, pd.DataFrame]  # MODIFIED: We now receive the returns directly
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
        fwd_returns_df = fwd_returns_by_horizon[horizon]
        triggered_returns = fwd_returns_df.loc[fwd_returns_df.index.isin(trigger_dates)]
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