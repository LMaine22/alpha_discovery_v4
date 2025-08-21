# alpha_discovery/reporting/artifacts.py

import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from ..config import Settings


def _safe_get(d: Dict[str, Any], key: str, default=None):
    """Tiny helper to avoid repeated dict.get chains."""
    return d.get(key, default) if isinstance(d, dict) else default


def _format_date(dt) -> str:
    """Format pandas/py datetime-like to YYYY-MM-DD safely."""
    try:
        return pd.Timestamp(dt).strftime('%Y-%m-%d')
    except Exception:
        return ""


def _settings_to_json(settings_model: Settings) -> str:
    """
    Pydantic v2+ uses model_dump_json; v1 used json().
    This helper keeps us compatible with both.
    """
    # Pydantic v2
    if hasattr(settings_model, "model_dump_json"):
        return settings_model.model_dump_json(indent=4)  # type: ignore[attr-defined]
    # Pydantic v1 fallback
    return settings_model.json(indent=4)  # type: ignore[call-arg]


def _get_base_portfolio_capital(settings_model: Settings) -> float:
    """
    Read reporting.base_portfolio_capital if present; else default to 100_000.
    """
    try:
        reporting = getattr(settings_model, "reporting", None)
        if reporting is not None and hasattr(reporting, "base_portfolio_capital"):
            return float(reporting.base_portfolio_capital)
    except Exception:
        pass
    return 100_000.0


def _portfolio_daily_returns_from_ledger(filtered_ledger: pd.DataFrame) -> pd.Series:
    """
    Build the daily portfolio return series by averaging pnl_pct by trigger_date
    across all selected trades of a setup.
    """
    if not isinstance(filtered_ledger, pd.DataFrame) or filtered_ledger.empty:
        return pd.Series(dtype=float)
    s = (
        filtered_ledger
        .groupby("trigger_date")["pnl_pct"]
        .mean()
        .sort_index()
    )
    return pd.to_numeric(s, errors="coerce").dropna()


def _compound_total_return(daily_returns: pd.Series) -> float:
    """
    Compound a daily return series into a total return over the whole period.
    """
    if daily_returns is None or daily_returns.empty:
        return 0.0
    try:
        return float((1.0 + daily_returns).prod() - 1.0)
    except Exception:
        return 0.0


def save_results(
        all_fold_results: List[Dict],
        signals_metadata: List[Dict],
        settings: Settings
):
    """
    Saves the final results from a walk-forward validation run.
    - Writes a timestamped run folder under ./runs
    - Dumps the JSON config used
    - Builds a summary CSV of winning setups across folds
    - Saves the full options trade ledger (combined) with setup_id & fold

    NEW in this version:
    - Adds portfolio-level P&L/return columns to the summary for each setup:
        * trades_count
        * sum_capital_allocated
        * sum_pnl_dollars
        * avg_pnl_per_trade_dollars
        * avg_pnl_per_trade_pct
        * portfolio_total_return_pct
        * portfolio_total_pnl_dollars_on_base  (base=$100,000 by default or settings.reporting.base_portfolio_capital)
    """
    if not all_fold_results:
        print("No solutions were found in any fold. Nothing to save.")
        return

    print("\n--- Saving Final Results ---")

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"run_seed{settings.ga.seed}_{run_timestamp}"
    output_dir = os.path.join('runs', folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Persist the full configuration (Pydantic v1/v2 compatible)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(_settings_to_json(settings))
    print(f"Configuration saved to: {config_path}")

    # Build a map for signal metadata (if needed for descriptions)
    signal_meta_map = {m['signal_id']: m for m in signals_metadata} if signals_metadata else {}

    base_capital_for_reporting = _get_base_portfolio_capital(settings)

    summary_rows: List[Dict[str, Any]] = []
    all_trade_ledgers: List[pd.DataFrame] = []

    print("Generating final reports from winning setups across all folds...")

    for i, solution in enumerate(all_fold_results):
        # ---------------------------------------------------------------------
        # Setup-level identifiers and metadata
        # ---------------------------------------------------------------------
        setup_id = f"SETUP_{i:04d}"
        fold_num = int(_safe_get(solution, 'fold', 0))
        rank = _safe_get(solution, 'rank', None)
        direction = _safe_get(solution, 'direction', 'N/A')

        # Selection metadata (chosen tickers & horizons)
        selection_info = _safe_get(solution, 'selection', {}) or {}
        chosen_tickers_str = _safe_get(selection_info, 'chosen_tickers', '')
        chosen_horizons_map = _safe_get(selection_info, 'chosen_horizons', {}) or {}

        # ---------------------------------------------------------------------
        # Ledger (filtered options ledger for this setup)
        # ---------------------------------------------------------------------
        trade_ledger = solution.get('trade_ledger', None)
        if isinstance(trade_ledger, pd.DataFrame) and not trade_ledger.empty:
            ledger_with_id = trade_ledger.copy()
            ledger_with_id['setup_id'] = setup_id
            ledger_with_id['fold'] = fold_num
            all_trade_ledgers.append(ledger_with_id)

            first_dt = _format_date(ledger_with_id['trigger_date'].min())
            last_dt = _format_date(ledger_with_id['trigger_date'].max())

            # "Best performing ticker" — by mean pnl_pct
            try:
                best_tkr_series = ledger_with_id.groupby('ticker')['pnl_pct'].mean()
                best_performing_ticker = best_tkr_series.idxmax() if not best_tkr_series.empty else ''
            except Exception:
                best_performing_ticker = ''

            # ===== NEW portfolio/trade aggregates for summary =====
            trades_count = int(len(ledger_with_id))
            # Sum the actual per-trade capital column (robust even if you change capital later)
            sum_capital_allocated = float(pd.to_numeric(ledger_with_id.get('capital_allocated', pd.Series([0]*trades_count)), errors="coerce").fillna(0).sum())
            sum_pnl_dollars = float(pd.to_numeric(ledger_with_id.get('pnl_dollars', pd.Series([0]*trades_count)), errors="coerce").fillna(0).sum())
            avg_pnl_per_trade_dollars = float(sum_pnl_dollars / trades_count) if trades_count > 0 else 0.0
            avg_pnl_per_trade_pct = float(pd.to_numeric(ledger_with_id.get('pnl_pct', pd.Series(dtype=float)), errors="coerce").dropna().mean()) if trades_count > 0 else 0.0

            daily_returns = _portfolio_daily_returns_from_ledger(ledger_with_id)
            portfolio_total_return_pct = _compound_total_return(daily_returns)
            portfolio_total_pnl_dollars_on_base = float(base_capital_for_reporting * portfolio_total_return_pct)
        else:
            first_dt = ""
            last_dt = ""
            best_performing_ticker = ''
            trades_count = 0
            sum_capital_allocated = 0.0
            sum_pnl_dollars = 0.0
            avg_pnl_per_trade_dollars = 0.0
            avg_pnl_per_trade_pct = 0.0
            portfolio_total_return_pct = 0.0
            portfolio_total_pnl_dollars_on_base = 0.0

        # ---------------------------------------------------------------------
        # Metrics & objectives (already computed during GA evaluation)
        # ---------------------------------------------------------------------
        perf_metrics = _safe_get(solution, 'metrics', {}) or {}
        sharpe_lb = float(perf_metrics.get('sharpe_lb', 0.0) or 0.0)
        sharpe_median = float(perf_metrics.get('sharpe_median', 0.0) or 0.0)
        omega_ratio = float(perf_metrics.get('omega_ratio', 0.0) or 0.0)
        support = float(perf_metrics.get('support', 0.0) or 0.0)
        annualized_return = float(perf_metrics.get('annualized_return', 0.0) or 0.0)
        volatility = float(perf_metrics.get('volatility', 0.0) or 0.0)
        max_drawdown = float(perf_metrics.get('max_drawdown', 0.0) or 0.0)

        # ---------------------------------------------------------------------
        # Description from primitive signals (human-readable)
        # ---------------------------------------------------------------------
        setup_signal_ids = _safe_get(solution, 'setup', []) or []
        description_parts: List[str] = []
        for s in setup_signal_ids:
            meta = signal_meta_map.get(s, {})
            description_parts.append(meta.get('description', s))
        description_text = " AND ".join(description_parts) if description_parts else ""

        # ---------------------------------------------------------------------
        # Flatten record for summary row
        # ---------------------------------------------------------------------
        flat_record = {
            'setup_id': setup_id,
            'fold': fold_num,
            'rank': rank,
            'best_performing_ticker': best_performing_ticker,
            'entry_direction': direction,
            'first_trigger_date': first_dt,
            'last_trigger_date': last_dt,
            'sharpe_lb': sharpe_lb,
            'sharpe_median': sharpe_median,
            'omega_ratio': omega_ratio,
            'support': support,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'description': description_text,
            'signal_ids': ", ".join(setup_signal_ids),
            # Selection metadata
            'chosen_tickers': chosen_tickers_str,
            'chosen_horizons': str(chosen_horizons_map),
            'num_selected_tickers': len(chosen_horizons_map) if isinstance(chosen_horizons_map, dict) else 0,
            # ===== NEW portfolio/trade aggregates =====
            'trades_count': trades_count,
            'sum_capital_allocated': sum_capital_allocated,
            'sum_pnl_dollars': sum_pnl_dollars,
            'avg_pnl_per_trade_dollars': avg_pnl_per_trade_dollars,
            'avg_pnl_per_trade_pct': avg_pnl_per_trade_pct,
            'portfolio_total_return_pct': portfolio_total_return_pct,
            'portfolio_total_pnl_dollars_on_base': portfolio_total_pnl_dollars_on_base,
        }
        summary_rows.append(flat_record)

    # -------------------------------------------------------------------------
    # Save Summary File
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    # Column order emphasizing selection + new portfolio aggregates alongside core metrics
    ordered_cols = [
        'setup_id', 'fold', 'rank',
        'sharpe_lb', 'sharpe_median', 'omega_ratio', 'support',
        'annualized_return', 'volatility', 'max_drawdown',
        'best_performing_ticker', 'entry_direction',
        'first_trigger_date', 'last_trigger_date',
        'chosen_tickers', 'chosen_horizons', 'num_selected_tickers',
        'trades_count', 'sum_capital_allocated', 'sum_pnl_dollars',
        'avg_pnl_per_trade_dollars', 'avg_pnl_per_trade_pct',
        'portfolio_total_return_pct', 'portfolio_total_pnl_dollars_on_base',
        'description', 'signal_ids'
    ]
    existing_cols = [c for c in ordered_cols if c in summary_df.columns]
    summary_df = summary_df.reindex(columns=existing_cols).sort_values(
        by=['fold', 'sharpe_lb'], ascending=[True, False]
    )

    summary_path = os.path.join(output_dir, 'pareto_front_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"Enriched summary saved to: {summary_path}")

    # -------------------------------------------------------------------------
    # Save Trade Ledger File (combined from all solutions)
    # -------------------------------------------------------------------------
    if all_trade_ledgers:
        full_trade_ledger_df = pd.concat(all_trade_ledgers, ignore_index=True)

        # Prefer a stable column ordering if present
        ledger_order = [
            "setup_id", "fold",
            "trigger_date", "exit_date",
            "ticker", "horizon_days", "direction", "option_type",
            "strike", "entry_underlying", "exit_underlying",
            "entry_iv", "exit_iv",
            "entry_option_price", "exit_option_price",
            "contracts", "capital_allocated",
            "pnl_dollars", "pnl_pct",
        ]
        existing_ledger_cols = [c for c in ledger_order if c in full_trade_ledger_df.columns]
        remaining_cols = [c for c in full_trade_ledger_df.columns if c not in existing_ledger_cols]
        full_trade_ledger_df = full_trade_ledger_df.reindex(columns=existing_ledger_cols + remaining_cols)

        ledger_path = os.path.join(output_dir, 'pareto_front_trade_ledger.csv')
        full_trade_ledger_df.to_csv(ledger_path, index=False, float_format='%.6f')
        print(f"Full options trade ledger saved to: {ledger_path}")
    else:
        print("No trade ledgers found to save.")
