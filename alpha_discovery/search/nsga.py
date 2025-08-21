# alpha_discovery/search/nsga.py

import pandas as pd
import numpy as np
from typing import List, Dict
from joblib import Parallel, delayed
from tqdm import tqdm

from ..config import settings
from ..engine import backtester
from ..eval import selection


def _evaluate_one_setup(
        setup: List[str],
        signals_df: pd.DataFrame,
        signals_metadata: List[Dict],
        master_df: pd.DataFrame
) -> Dict:
    """
    Evaluate a single setup on training data using options-based backtesting and
    best-of selection:
      - Build full options ledger across all tickers & horizons for this setup.
      - For each ticker, choose best horizon by Sharpe LB (tie: Omega, Support).
      - Select up to top-N tickers (min-distinct & min-support enforced).
      - Build daily portfolio returns (mean of pnl_pct by trigger_date).
      - Compute metrics and map to GA objectives: [sharpe_lb, omega_ratio, support].
    """
    # Determine "direction" from the signal metadata (same heuristic you used):
    # count '<' as bearish; otherwise bullish.
    direction_score = 0
    for signal_id in setup:
        meta = next((m for m in signals_metadata if m['signal_id'] == signal_id), None)
        if meta and '<' in meta.get('condition', ''):
            direction_score -= 1
        else:
            direction_score += 1
    direction = 'long' if direction_score >= 0 else 'short'

    # Build the full options ledger for this setup
    options_ledger = backtester.run_setup_backtest_options(
        setup_signals=setup,
        signals_df=signals_df,
        master_df=master_df,
        direction=direction
    )

    if options_ledger is None or options_ledger.empty:
        # No trades → empty metrics/objectives
        empty_metrics = {}
        objectives = [
            empty_metrics.get('sharpe_lb', -99.0),
            empty_metrics.get('omega_ratio', 0.0),
            empty_metrics.get('support', 0.0),
        ]
        return {
            'setup': setup,
            'metrics': empty_metrics,
            'objectives': objectives,
            'rank': np.inf,
            'crowding_distance': 0.0,
            'trade_ledger': pd.DataFrame(),
            'direction': direction,
            'selection': {"chosen_tickers": "", "chosen_horizons": {}},
        }

    # Select best horizon per ticker, then rank and pick top-N tickers
    best_by_ticker = selection.select_best_horizon_per_ticker(
        options_ledger,
        metric_primary=settings.selection.metric_primary,
        metric_tiebreakers=settings.selection.metric_tiebreakers,
        min_support_per_ticker=settings.selection.min_support_per_ticker
    )
    chosen = selection.rank_and_select_tickers(
        best_by_ticker,
        top_n=settings.selection.top_n_tickers,
        min_distinct_tickers=settings.selection.min_distinct_tickers,
        metric_primary=settings.selection.metric_primary,
        metric_tiebreakers=settings.selection.metric_tiebreakers
    )

    # Filter ledger to selected (ticker, best_horizon) pairs
    filtered_ledger = selection.filter_ledger_to_selection(options_ledger, chosen)

    # Build daily portfolio returns (mean by trigger_date) and compute metrics
    daily_returns = selection.portfolio_daily_returns(filtered_ledger)
    perf_metrics = selection.portfolio_metrics(daily_returns)

    # Map metrics to GA objectives
    objectives = [
        perf_metrics.get('sharpe_lb', -99.0),
        perf_metrics.get('omega_ratio', 0.0),
        perf_metrics.get('support', 0.0),
    ]

    return {
        'setup': setup,
        'metrics': perf_metrics,
        'objectives': objectives,
        'rank': np.inf,
        'crowding_distance': 0.0,
        'trade_ledger': filtered_ledger,
        'direction': direction,
        'selection': selection.selection_summary(chosen),
    }


def _non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    """
    Perform non-dominated sorting on the population based on 'objectives'.
    Higher is better for all objectives.
    """
    fronts: List[List[Dict]] = []
    for ind1 in population:
        ind1['domination_count'] = 0
        ind1['dominated_solutions'] = []
        for ind2 in population:
            if ind1 is ind2:
                continue
            # ind1 dominates ind2 if ind1 >= ind2 in all objectives and > in at least one
            is_dominant = all(o1 >= o2 for o1, o2 in zip(ind1['objectives'], ind2['objectives'])) and \
                          any(o1 > o2 for o1, o2 in zip(ind1['objectives'], ind2['objectives']))
            if is_dominant:
                ind1['dominated_solutions'].append(ind2)
            elif all(o2 >= o1 for o1, o2 in zip(ind1['objectives'], ind2['objectives'])) and \
                    any(o2 > o1 for o1, o2 in zip(ind1['objectives'], ind2['objectives'])):
                ind1['domination_count'] += 1

    front1 = [ind for ind in population if ind['domination_count'] == 0]
    current_front = front1
    rank_num = 1

    while current_front:
        for ind in current_front:
            ind['rank'] = rank_num
        fronts.append(current_front)
        next_front: List[Dict] = []
        for ind1 in current_front:
            for ind2 in ind1['dominated_solutions']:
                ind2['domination_count'] -= 1
                if ind2['domination_count'] == 0:
                    next_front.append(ind2)
        rank_num += 1
        current_front = next_front

    return fronts


def _calculate_crowding_distance(front: List[Dict]):
    """
    Calculate crowding distance for a given front.
    """
    if not front:
        return
    num_objectives = len(front[0]['objectives'])
    for ind in front:
        ind['crowding_distance'] = 0.0

    for i in range(num_objectives):
        front.sort(key=lambda x: x['objectives'][i])
        min_val = front[0]['objectives'][i]
        max_val = front[-1]['objectives'][i]
        front[0]['crowding_distance'] = np.inf
        front[-1]['crowding_distance'] = np.inf
        if max_val == min_val:
            continue
        for j in range(1, len(front) - 1):
            front[j]['crowding_distance'] += (
                (front[j + 1]['objectives'][i] - front[j - 1]['objectives'][i]) / (max_val - min_val)
            )


def evolve(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """
    Run NSGA-II evolution with options-based evaluation and selection.

    Notes:
      - Uses only TRAINING data inside the fold (caller provides train slices).
      - Keeps objectives: [sharpe_lb, omega_ratio, support].
    """
    rng = np.random.default_rng(settings.ga.seed)
    all_signal_ids = list(signals_df.columns)

    # Initialize random parent population (setups are lists of signal IDs)
    from . import population as pop
    parent_population = pop.initialize_population(rng, all_signal_ids)

    print("\n--- Starting Genetic Algorithm Evolution (Options Mode) ---")
    pbar = tqdm(range(settings.ga.generations), desc="Evolving Generations")

    next_gen_parents: List[Dict] = []

    for gen in pbar:
        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Evaluating {len(parent_population)} parents...")
        evaluated_parents = Parallel(n_jobs=-1)(
            delayed(_evaluate_one_setup)(setup, signals_df, signals_metadata, master_df)
            for setup in parent_population
        )

        # Generate children via tournament selection + crossover + mutation
        children_setups = []
        while len(children_setups) < settings.ga.population_size:
            # Tournament selection prefers lower 'rank' and higher crowding distance
            p1 = min(rng.choice(evaluated_parents, 2, replace=False),
                     key=lambda x: (x['rank'], -x['crowding_distance']))
            p2 = min(rng.choice(evaluated_parents, 2, replace=False),
                     key=lambda x: (x['rank'], -x['crowding_distance']))

            child_setup = pop.crossover(p1['setup'], p2['setup'], rng)
            child_setup = pop.mutate(child_setup, all_signal_ids, rng)
            if child_setup:
                children_setups.append(child_setup)

        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Evaluating {len(children_setups)} children...")
        evaluated_children = Parallel(n_jobs=-1)(
            delayed(_evaluate_one_setup)(setup, signals_df, signals_metadata, master_df)
            for setup in children_setups
        )

        # Survivor selection: combine, sort into fronts, then fill next gen by fronts + crowding distance
        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Selecting survivors...")
        combined_population = evaluated_parents + evaluated_children
        fronts = _non_dominated_sort(combined_population)

        next_gen_parents = []
        for front in fronts:
            _calculate_crowding_distance(front)
            if len(next_gen_parents) + len(front) <= settings.ga.population_size:
                next_gen_parents.extend(front)
            else:
                # Partial take from this front by highest crowding distance
                front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                needed = settings.ga.population_size - len(next_gen_parents)
                next_gen_parents.extend(front[:needed])
                break

        # Next generation's setups
        parent_population = [ind['setup'] for ind in next_gen_parents]

        # Progress summary (front 1 stats)
        best_front = fronts[0] if fronts else []
        if best_front:
            best_objectives = [ind['objectives'] for ind in best_front]
            avg_sharpe_lb = np.mean([o[0] for o in best_objectives]) if best_objectives else 0.0
            avg_omega = np.mean([o[1] for o in best_objectives]) if best_objectives else 0.0
            pbar.set_postfix({"Sharpe LB": f"{avg_sharpe_lb:.2f}", "Omega": f"{avg_omega:.2f}", "Front Size": len(best_front)})
        else:
            pbar.set_postfix({"Sharpe LB": "0.00", "Omega": "0.00", "Front Size": 0})

    pbar.close()
    print("Evolution Complete.")

    # Return the final non-dominated front from the last selection step
    final_fronts = _non_dominated_sort(next_gen_parents) if next_gen_parents else []
    final_front = final_fronts[0] if final_fronts else []
    return final_front
