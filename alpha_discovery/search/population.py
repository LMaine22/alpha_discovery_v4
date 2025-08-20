# alpha_discovery/search/population.py

import numpy as np
from typing import List

from ..config import settings


def initialize_population(
        rng: np.random.Generator,
        all_signal_ids: List[str],
) -> List[List[str]]:
    """
    Creates the initial random population of setups.

    Args:
        rng: The NumPy random number generator for deterministic results.
        all_signal_ids: The complete list of available primitive signal IDs.

    Returns:
        A list of setups, where each setup is a list of signal IDs.
    """
    print(f"Initializing population of size {settings.ga.population_size}...")
    population = []

    # Use a set to ensure we don't create duplicate setups in the first generation
    seen_dna = set()

    while len(population) < settings.ga.population_size:
        # Randomly choose a length for this setup (e.g., 2 or 3 signals)
        length = rng.choice(settings.ga.setup_lengths_to_explore)

        # Randomly choose signal IDs without replacement
        setup = list(rng.choice(all_signal_ids, size=length, replace=False))

        # Create a canonical representation (DNA) to check for uniqueness
        dna = tuple(sorted(setup))

        if dna not in seen_dna:
            seen_dna.add(dna)
            population.append(setup)

    return population


def crossover(
        parent1: List[str],
        parent2: List[str],
        rng: np.random.Generator
) -> List[str]:
    """
    Combines two parent setups to create a new child setup.

    This implementation uses a simple method: it pools all unique signals
    from both parents and randomly draws a new setup from that pool.
    """
    # Pool all unique signals from both parents
    combined_pool = list(set(parent1) | set(parent2))

    # Choose a length for the child, defaulting to the length of the first parent
    child_length = rng.choice([len(parent1), len(parent2)])

    # Ensure we don't try to sample more signals than are in the pool
    if len(combined_pool) < child_length:
        child_length = len(combined_pool)

    if child_length == 0:
        return []  # Return an empty child if there's nothing to cross over

    # Create the child by sampling from the combined pool
    child = list(rng.choice(combined_pool, size=child_length, replace=False))

    return child


def mutate(
        setup: List[str],
        all_signal_ids: List[str],
        rng: np.random.Generator
) -> List[str]:
    """
    Applies a random mutation to a setup.

    With a probability defined by the mutation rate, one signal in the setup
    is replaced with a new, random signal from the entire pool.
    """
    if rng.random() < settings.ga.mutation_rate and len(setup) > 0:
        # Choose a random signal in the setup to replace
        index_to_mutate = rng.integers(0, len(setup))

        # Find a new signal that is not already in the setup
        current_signals = set(setup)
        potential_new_signals = [sig for sig in all_signal_ids if sig not in current_signals]

        if potential_new_signals:
            new_signal = rng.choice(potential_new_signals)

            # Replace the old signal with the new one
            mutated_setup = setup.copy()
            mutated_setup[index_to_mutate] = new_signal
            return mutated_setup

    # If no mutation occurs, return the original setup
    return setup