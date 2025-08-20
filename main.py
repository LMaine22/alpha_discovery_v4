# main.py

import pandas as pd
from alpha_discovery.config import settings
from alpha_discovery.data.loader import load_data_from_parquet
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals
from alpha_discovery.search.nsga import evolve
from alpha_discovery.reporting.artifacts import save_results
from alpha_discovery.eval.validation import create_walk_forward_splits


def run_pipeline():
    """
    Runs the full alpha discovery pipeline using walk-forward validation.
    """
    # --- Step 1: Load, Feature Engineer, and Compile Signals (Done Once on All Data) ---
    master_df = load_data_from_parquet()
    if master_df.empty: return

    feature_matrix = build_feature_matrix(master_df)
    if feature_matrix.empty: return

    signals_df, signals_metadata = compile_signals(feature_matrix)
    if signals_df.empty: return

    # --- Step 2: Create Walk-Forward Splits ---
    splits = create_walk_forward_splits(master_df.index)
    if not splits:
        print("Could not create any walk-forward splits. Check data date range and config.")
        return

    # --- Step 3: Loop Through Splits and Run GA on Each Training Set ---
    all_fold_results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        fold_num = i + 1
        print(f"\n{'=' * 20} RUNNING FOLD {fold_num}/{len(splits)} {'=' * 20}")
        print(f"Training Period: {train_idx.min().date()} to {train_idx.max().date()}")
        print(f"Testing Period:  {test_idx.min().date()} to {test_idx.max().date()}")

        # Slice all dataframes to only include the training data for this fold
        train_master_df = master_df.loc[train_idx]

        # MODIFIED: Use .reindex() for robust slicing.
        # This prevents KeyErrors if the start of train_idx is missing from signals_df
        # due to rolling window warm-up periods.
        train_signals_df = signals_df.reindex(train_idx).fillna(False)

        # The GA only ever sees the training data for this fold
        pareto_front_for_fold = evolve(train_signals_df, signals_metadata, train_master_df)

        # Add fold information to the results
        for solution in pareto_front_for_fold:
            solution['fold'] = fold_num

        all_fold_results.extend(pareto_front_for_fold)

    print(f"\n{'=' * 20} WALK-FORWARD VALIDATION COMPLETE {'=' * 20}")

    # --- Step 4: Save the Aggregated Results From All Folds ---
    save_results(all_fold_results, signals_metadata, settings)


if __name__ == '__main__':
    print("--- Starting Full Alpha Discovery Pipeline with Walk-Forward Validation ---")
    run_pipeline()
    print("\n--- Pipeline Finished ---")