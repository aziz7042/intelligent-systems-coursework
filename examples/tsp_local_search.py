"""Hill climbing and threshold acceptance for a symmetric TSP instance."""

from collections.abc import Callable

import numpy as np


DISTANCES = np.array(
    [
        [0, 5, 8, 11, 3, 7],
        [5, 0, 10, 4, 9, 12],
        [8, 10, 0, 6, 17, 8],
        [11, 4, 6, 0, 6, 5],
        [3, 9, 17, 6, 0, 11],
        [7, 12, 8, 5, 11, 0],
    ],
    dtype=float,
)


def route_length(route: np.ndarray, distances: np.ndarray) -> float:
    return float(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


def swap_neighbor(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    candidate = route.copy()
    left, right = rng.choice(len(route), size=2, replace=False)
    candidate[left], candidate[right] = candidate[right], candidate[left]
    return candidate


def search(
    distances: np.ndarray,
    accept: Callable[[float, float, int], bool],
    *,
    iterations: int = 5_000,
    seed: int = 7,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    current = rng.permutation(len(distances))
    current_cost = route_length(current, distances)
    best, best_cost = current.copy(), current_cost

    for step in range(iterations):
        candidate = swap_neighbor(current, rng)
        candidate_cost = route_length(candidate, distances)
        if accept(current_cost, candidate_cost, step):
            current, current_cost = candidate, candidate_cost
        if current_cost < best_cost:
            best, best_cost = current.copy(), current_cost

    return best, best_cost


def hill_climb(distances: np.ndarray) -> tuple[np.ndarray, float]:
    return search(distances, lambda current, candidate, _step: candidate < current)


def threshold_acceptance(distances: np.ndarray, initial_threshold: float = 8.0) -> tuple[np.ndarray, float]:
    def accept(current: float, candidate: float, step: int) -> bool:
        threshold = initial_threshold * (1.0 - step / 5_000)
        return candidate - current <= max(0.0, threshold)

    return search(distances, accept)


if __name__ == "__main__":
    for name, optimizer in (("hill climb", hill_climb), ("threshold acceptance", threshold_acceptance)):
        route, cost = optimizer(DISTANCES)
        assert sorted(route.tolist()) == list(range(len(DISTANCES)))
        print(f"{name}: route={route.tolist()} cost={cost:.0f}")
