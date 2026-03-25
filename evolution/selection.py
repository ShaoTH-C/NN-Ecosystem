# selection.py - evolutionary operators: tournament selection, crossover, etc.

import numpy as np
from typing import List
from evolution.genome import Genome
import config as cfg


def select_parent(population: List[Genome], tournament_size: int = 3) -> Genome:
    """Pick a few random individuals and return the fittest one."""
    if len(population) <= tournament_size:
        competitors = population
    else:
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        competitors = [population[i] for i in indices]

    best = max(competitors, key=lambda g: g.fitness)
    return best


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """
    Uniform crossover - each gene randomly comes from parent A or B.
    If the parents have different topologies, just clone the fitter one.
    Generation is set to max of parents (caller handles incrementing).
    """
    if parent_a.layer_sizes != parent_b.layer_sizes:
        better = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
        return better.copy()

    mask = np.random.rand(len(parent_a.genes)) < 0.5
    child_genes = np.where(mask, parent_a.genes, parent_b.genes)

    child = Genome(
        layer_sizes=list(parent_a.layer_sizes),
        genes=child_genes,
        generation=max(parent_a.generation, parent_b.generation),
    )
    return child


def blend_crossover(parent_a: Genome, parent_b: Genome, alpha: float = 0.5) -> Genome:
    """
    BLX-alpha crossover - interpolates between parents instead of picking
    one or the other. Tends to explore the search space more.
    """
    if parent_a.layer_sizes != parent_b.layer_sizes:
        better = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
        return better.copy()

    t = np.random.uniform(-alpha, 1.0 + alpha, size=len(parent_a.genes))
    child_genes = parent_a.genes * t + parent_b.genes * (1.0 - t)

    child = Genome(
        layer_sizes=list(parent_a.layer_sizes),
        genes=child_genes,
        generation=max(parent_a.generation, parent_b.generation),
    )
    return child


def create_next_generation(
    population: List[Genome],
    pop_size: int,
    elite_count: int = 2,
    tournament_size: int = 3,
) -> List[Genome]:
    """
    Build the next generation. Top individuals survive unchanged (elitism),
    the rest are bred via tournament selection + crossover + mutation.
    This is for the optional generational mode.
    """
    if not population:
        return [Genome() for _ in range(pop_size)]

    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)

    next_gen: List[Genome] = []

    # elitism - keep the best ones as-is
    for i in range(min(elite_count, len(sorted_pop))):
        elite = sorted_pop[i].copy()
        elite.fitness = 0.0
        next_gen.append(elite)

    # fill the rest with offspring
    while len(next_gen) < pop_size:
        parent_a = select_parent(sorted_pop, tournament_size)

        if np.random.rand() < cfg.CROSSOVER_RATE and len(sorted_pop) > 1:
            parent_b = select_parent(sorted_pop, tournament_size)
            attempts = 0
            while parent_b is parent_a and attempts < 5:
                parent_b = select_parent(sorted_pop, tournament_size)
                attempts += 1
            child = crossover(parent_a, parent_b)
        else:
            child = parent_a.copy()

        child = child.mutate()
        child.fitness = 0.0
        next_gen.append(child)

    return next_gen[:pop_size]
