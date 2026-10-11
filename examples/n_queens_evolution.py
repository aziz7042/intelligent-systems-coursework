"""Evolutionary search for N-Queens using permutation chromosomes."""

import numpy as np


def conflicts(chromosome: np.ndarray) -> int:
    total = 0
    for left in range(len(chromosome)):
        for right in range(left + 1, len(chromosome)):
            if abs(int(chromosome[left]) - int(chromosome[right])) == right - left:
                total += 1
    return total


def ordered_crossover(first: np.ndarray, second: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    start, end = sorted(rng.choice(len(first), size=2, replace=False))
    child = np.full(len(first), -1, dtype=int)
    child[start:end] = first[start:end]
    remaining = [gene for gene in second if gene not in child]
    positions = [index for index, value in enumerate(child) if value == -1]
    child[positions] = remaining
    return child


def mutate(chromosome: np.ndarray, rng: np.random.Generator, probability: float = 0.25) -> np.ndarray:
    result = chromosome.copy()
    if rng.random() < probability:
        left, right = rng.choice(len(result), size=2, replace=False)
        result[left], result[right] = result[right], result[left]
    return result


def solve(n: int = 8, *, population_size: int = 120, generations: int = 1_000, seed: int = 7) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    population = [rng.permutation(n) for _ in range(population_size)]

    def tournament() -> np.ndarray:
        contestants = rng.choice(population_size, size=4, replace=False)
        return min((population[index] for index in contestants), key=conflicts)

    for _ in range(generations):
        population.sort(key=conflicts)
        if conflicts(population[0]) == 0:
            return population[0], 0
        next_population = [population[0].copy(), population[1].copy()]
        while len(next_population) < population_size:
            child = ordered_crossover(tournament(), tournament(), rng)
            next_population.append(mutate(child, rng))
        population = next_population

    population.sort(key=conflicts)
    return population[0], conflicts(population[0])


if __name__ == "__main__":
    solution, score = solve()
    assert score == 0
    print("solution:", solution.tolist())
    print("diagonal conflicts:", score)
