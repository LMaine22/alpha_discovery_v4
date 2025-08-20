# alpha_discovery/eval/metrics.py

import pandas as pd
import numpy as np
from typing import Dict


def winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clips a series to the specified lower and upper quantiles."""
    lower_bound = series.quantile(lower_q)
    upper_bound = series.quantile(upper_q)
    return series.clip(lower=lower_bound, upper=upper_bound)


def block_bootstrap_sharpe(
        returns_series: pd.Series,
        block_size: int = 5,
        num_iterations: int = 500,
        trading_days_per_year: int = 252
) -> Dict[str, float]:
    """Calculates the Sharpe ratio using a block bootstrap method."""
    if len(returns_series) < block_size * 2:
        return {'sharpe_median': 0.0, 'sharpe_lb': 0.0, 'sharpe_ub': 0.0}

    blocks = [
        returns_series.iloc[i: i + block_size]
        for i in range(0, len(returns_series) - block_size + 1)
    ]
    if not blocks:
        return {'sharpe_median': 0.0, 'sharpe_lb': 0.0, 'sharpe_ub': 0.0}

    num_blocks_to_sample = len(returns_series) // block_size
    sharpe_ratios = []

    for _ in range(num_iterations):
        resampled_blocks = np.random.choice(len(blocks), num_blocks_to_sample, replace=True)
        resampled_returns = pd.concat([blocks[i] for i in resampled_blocks])

        if resampled_returns.std() > 1e-9:
            mean_return = resampled_returns.mean()
            std_dev = resampled_returns.std()
            sharpe = (mean_return / std_dev) * np.sqrt(trading_days_per_year)
            sharpe_ratios.append(sharpe)

    if not sharpe_ratios:
        return {'sharpe_median': 0.0, 'sharpe_lb': 0.0, 'sharpe_ub': 0.0}

    return {
        'sharpe_median': np.nanmedian(sharpe_ratios),
        'sharpe_lb': np.nanpercentile(sharpe_ratios, 5),
        'sharpe_ub': np.nanpercentile(sharpe_ratios, 95)
    }


def calculate_omega_ratio(series: pd.Series, required_return: float = 0.0) -> float:
    """
    Calculates the Omega ratio, a more complete measure of risk/reward.
    It's the ratio of probability-weighted gains vs. probability-weighted losses.
    """
    returns_less_than_threshold = series - required_return

    gains = returns_less_than_threshold[returns_less_than_threshold > 0].sum()
    losses = returns_less_than_threshold[returns_less_than_threshold < 0].sum()

    if abs(losses) < 1e-9:
        return np.inf

    return gains / abs(losses)


def calculate_max_drawdown(series: pd.Series) -> float:
    if series.empty: return 0.0
    cumulative_returns = (1 + series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def calculate_all_metrics(
        trade_ledger: pd.DataFrame,
        setup_direction: str = 'long'
) -> Dict[str, float]:
    """Main function to calculate all performance metrics for a given setup."""
    if trade_ledger.empty:
        return {}

    daily_returns = trade_ledger.groupby('trigger_date')['forward_return'].mean()

    if setup_direction == 'short':
        daily_returns *= -1

    cleaned_returns = winsorize(daily_returns)
    if cleaned_returns.empty:
        return {}

    support = len(daily_returns)
    sharpe_stats = block_bootstrap_sharpe(cleaned_returns)
    omega = calculate_omega_ratio(cleaned_returns)
    max_dd = calculate_max_drawdown(cleaned_returns)

    min_days_for_annualization = 126
    if len(cleaned_returns) >= min_days_for_annualization:
        num_years = len(cleaned_returns) / 252
        total_return = (1 + cleaned_returns).prod() - 1
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
    else:
        annualized_return = 0.0

    final_metrics = {
        'support': float(support),
        'annualized_return': annualized_return,
        'volatility': cleaned_returns.std() * np.sqrt(252),
        'max_drawdown': max_dd,
        'omega_ratio': omega,
    }
    final_metrics.update(sharpe_stats)

    return {k: np.nan_to_num(v, nan=0.0, posinf=999, neginf=-999) for k, v in final_metrics.items()}