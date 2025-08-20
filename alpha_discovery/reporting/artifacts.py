# alpha_discovery/reporting/artifacts.py

import os
import pandas as pd
from typing import List, Dict
from datetime import datetime

from ..config import Settings


def save_results(
        all_fold_results: List[Dict],  # MODIFIED: Takes a list of results from all folds
        signals_metadata: List[Dict],
        settings: Settings
):
    """Saves the final results from a walk-forward validation run."""
    if not all_fold_results:
        print("No solutions were found in any fold. Nothing to save.")
        return

    print("\n--- Saving Final Results ---")

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"run_seed{settings.ga.seed}_{run_timestamp}"
    output_dir = os.path.join('runs', folder_name)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(settings.json(indent=4))
    print(f"Configuration saved to: {config_path}")

    signal_meta_map = {m['signal_id']: m for m in signals_metadata}
    summary_rows = []
    all_trade_ledgers = []

    print("Generating final reports from winning setups across all folds...")
    for i, solution in enumerate(all_fold_results):
        setup_id = f"SETUP_{i:04d}"
        trade_ledger = solution.get('trade_ledger')
        if trade_ledger is None or trade_ledger.empty:
            continue

        ledger_with_id = trade_ledger.copy()
        ledger_with_id['setup_id'] = setup_id
        ledger_with_id['fold'] = solution.get('fold', 0)
        all_trade_ledgers.append(ledger_with_id)

        best_ticker_perf = trade_ledger.groupby('ticker')['forward_return'].mean().idxmax()

        flat_record = {
            'setup_id': setup_id,
            'fold': solution.get('fold', 0),  # ADDED: Track the fold number
            'rank': solution['rank'],
            'best_performing_ticker': best_ticker_perf,
            'entry_direction': solution.get('direction', 'N/A'),
            'first_trigger_date': trade_ledger['trigger_date'].min().strftime('%Y-%m-%d'),
            'last_trigger_date': trade_ledger['trigger_date'].max().strftime('%Y-%m-%d'),
            **solution['metrics'],
            'description': " AND ".join([signal_meta_map.get(s, {}).get('description', s) for s in solution['setup']]),
            'signal_ids': ", ".join(solution['setup'])
        }
        summary_rows.append(flat_record)

    # Save Summary File
    summary_df = pd.DataFrame(summary_rows)
    ordered_cols = [
        'setup_id', 'fold', 'rank', 'sharpe_lb', 'sharpe_median', 'omega_ratio',
        'support', 'best_performing_ticker', 'entry_direction', 'annualized_return',
        'volatility', 'max_drawdown', 'first_trigger_date', 'last_trigger_date',
        'description', 'signal_ids'
    ]
    summary_df = summary_df.reindex(columns=ordered_cols).sort_values(by=['fold', 'sharpe_lb'], ascending=[True, False])
    summary_path = os.path.join(output_dir, 'pareto_front_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Enriched summary saved to: {summary_path}")

    # Save Trade Ledger File
    if all_trade_ledgers:
        full_trade_ledger_df = pd.concat(all_trade_ledgers, ignore_index=True)
        ledger_path = os.path.join(output_dir, 'pareto_front_trade_ledger.csv')
        full_trade_ledger_df.to_csv(ledger_path, index=False, float_format='%.4f')
        print(f"Full trade ledger saved to: {ledger_path}")