# alpha_discovery/search/nsga.py

import pandas as pd
import numpy as np
from typing import List, Dict
from joblib import Parallel, delayed
from tqdm import tqdm

from ..config import settings
from ..engine import backtester
from ..eval import metrics
from . import population as pop


def _evaluate_one_setup(
        setup: List[str],
        signals_df: pd.DataFrame,
        signals_metadata: List[Dict],
        fwd_returns_by_horizon: Dict[int, pd.DataFrame]
) -> Dict:
    """Private helper to run the full evaluation pipeline for a single setup."""
    direction_score = 0
    for signal_id in setup:
        meta = next((m for m in signals_metadata if m['signal_id'] == signal_id), None)
        if meta and '<' in meta['condition']:
            direction_score -= 1
        else:
            direction_score += 1

    direction = 'long' if direction_score >= 0 else 'short'

    trade_ledger = backtester.run_setup_backtest(setup, signals_df, fwd_returns_by_horizon)
    perf_metrics = metrics.calculate_all_metrics(trade_ledger, direction)

    # MODIFIED: Objectives now use Omega Ratio
    objectives = [
        perf_metrics.get('sharpe_lb', -99),
        perf_metrics.get('omega_ratio', 0),
        perf_metrics.get('support', 0)
    ]

    return {
        'setup': setup,
        'metrics': perf_metrics,
        'objectives': objectives,
        'rank': np.inf,
        'crowding_distance': 0.0,
        'trade_ledger': trade_ledger,
        'direction': direction  # ADDED: Pass direction to results
    }


def _non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    # ... (This function remains unchanged)
    fronts = []
    for ind1 in population:
        ind1['domination_count'] = 0
        ind1['dominated_solutions'] = []
        for ind2 in population:
            if ind1 is ind2: continue
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
        for ind in current_front: ind['rank'] = rank_num
        fronts.append(current_front)
        next_front = []
        for ind1 in current_front:
            for ind2 in ind1['dominated_solutions']:
                ind2['domination_count'] -= 1
                if ind2['domination_count'] == 0: next_front.append(ind2)
        rank_num += 1
        current_front = next_front
    return fronts


def _calculate_crowding_distance(front: List[Dict]):
    # ... (This function remains unchanged)
    if not front: return
    num_objectives = len(front[0]['objectives'])
    for ind in front: ind['crowding_distance'] = 0
    for i in range(num_objectives):
        front.sort(key=lambda x: x['objectives'][i])
        min_val, max_val = front[0]['objectives'][i], front[-1]['objectives'][i]
        front[0]['crowding_distance'] = np.inf
        front[-1]['crowding_distance'] = np.inf
        if max_val == min_val: continue
        for j in range(1, len(front) - 1):
            front[j]['crowding_distance'] += (front[j + 1]['objectives'][i] - front[j - 1]['objectives'][i]) / (
                        max_val - min_val)


def evolve(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """The main function to run the entire NSGA-II genetic algorithm."""
    rng = np.random.default_rng(settings.ga.seed)
    all_signal_ids = list(signals_df.columns)
    fwd_returns = backtester.prepare_forward_returns(master_df)
    parent_population = pop.initialize_population(rng, all_signal_ids)

    print("\n--- Starting Genetic Algorithm Evolution ---")
    pbar = tqdm(range(settings.ga.generations), desc="Evolving Generations")

    for gen in pbar:
        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Evaluating {len(parent_population)} parents...")
        evaluated_parents = Parallel(n_jobs=-1)(
            delayed(_evaluate_one_setup)(setup, signals_df, signals_metadata, fwd_returns) for setup in
            parent_population
        )
        children = []
        while len(children) < settings.ga.population_size:
            p1 = min(rng.choice(evaluated_parents, 2, replace=False),
                     key=lambda x: (x['rank'], -x['crowding_distance']))
            p2 = min(rng.choice(evaluated_parents, 2, replace=False),
                     key=lambda x: (x['rank'], -x['crowding_distance']))
            child_setup = pop.crossover(p1['setup'], p2['setup'], rng)
            child_setup = pop.mutate(child_setup, all_signal_ids, rng)
            if child_setup: children.append(child_setup)
        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Evaluating {len(children)} children...")
        evaluated_children = Parallel(n_jobs=-1)(
            delayed(_evaluate_one_setup)(setup, signals_df, signals_metadata, fwd_returns) for setup in children
        )
        tqdm.write(f"Gen {gen + 1}/{settings.ga.generations} | Selecting survivors...")
        combined_population = evaluated_parents + evaluated_children
        fronts = _non_dominated_sort(combined_population)
        next_gen_parents = []
        for front in fronts:
            _calculate_crowding_distance(front)
            if len(next_gen_parents) + len(front) <= settings.ga.population_size:
                next_gen_parents.extend(front)
            else:
                front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                needed = settings.ga.population_size - len(next_gen_parents)
                next_gen_parents.extend(front[:needed])
                break
        parent_population = [ind['setup'] for ind in next_gen_parents]
        best_front = fronts[0]
        best_objectives = [ind['objectives'] for ind in best_front]
        avg_sharpe_lb = np.mean([o[0] for o in best_objectives])
        avg_omega = np.mean([o[1] for o in best_objectives])  # MODIFIED
        pbar.set_postfix(
            {"Sharpe LB": f"{avg_sharpe_lb:.2f}", "Omega": f"{avg_omega:.2f}", "Front Size": len(best_front)})
    pbar.close()
    print("Evolution Complete.")
    final_front = _non_dominated_sort(next_gen_parents)[0]
    return final_front