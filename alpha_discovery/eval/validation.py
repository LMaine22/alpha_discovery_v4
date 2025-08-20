# alpha_discovery/eval/validation.py

import pandas as pd
from typing import List, Tuple

from ..config import settings


def create_walk_forward_splits(
        data_index: pd.DatetimeIndex,
        train_years: int = 3,
        test_years: int = 1,
        step_months: int = 6
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Creates a list of training and testing splits for walk-forward validation.

    This function generates rolling windows of data. For each window, it defines
    a training period and a subsequent, non-overlapping testing period. It also
    enforces an "embargo" period between train and test to prevent data leakage.

    Args:
        data_index: The complete DatetimeIndex of the dataset.
        train_years: The length of each training period in years.
        test_years: The length of each testing period in years.
        step_months: How many months to roll the window forward for each new split.

    Returns:
        A list of tuples, where each tuple contains (train_index, test_index).
    """
    splits = []

    start_date = data_index.min()
    end_date = data_index.max()

    train_period = pd.DateOffset(years=train_years)
    test_period = pd.DateOffset(years=test_years)
    step_period = pd.DateOffset(months=step_months)
    embargo_period = pd.DateOffset(days=settings.validation.embargo_days)

    current_start = start_date

    print("\n--- Creating Walk-Forward Splits ---")
    while True:
        train_end = current_start + train_period
        test_start = train_end + embargo_period
        test_end = test_start + test_period

        # Ensure the entire split is within the bounds of the data
        if test_end > end_date:
            break

        # Select the actual data indices for this split
        train_indices = data_index[(data_index >= current_start) & (data_index < train_end)]
        test_indices = data_index[(data_index >= test_start) & (data_index < test_end)]

        if not train_indices.empty and not test_indices.empty:
            splits.append((train_indices, test_indices))
            print(
                f"Created Split {len(splits)}: "
                f"Train ({train_indices.min().date()} to {train_indices.max().date()}), "
                f"Test ({test_indices.min().date()} to {test_indices.max().date()})"
            )

        # Move the window forward for the next split
        current_start += step_period

    print(f"Generated {len(splits)} walk-forward splits.")
    return splits