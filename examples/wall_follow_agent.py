"""Small stateful rule-based wall-following example.

The eight sensor values are ordered clockwise from north-west through west.
Each value is truthy when that neighboring cell is blocked.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class WallFollowAgent:
    reverse_steps: int = 0

    @staticmethod
    def sense(sensors: Iterable[object]) -> tuple[int, ...]:
        features = tuple(1 if value else 0 for value in sensors)
        if len(features) != 8:
            raise ValueError("exactly eight sensor values are required")
        return features

    def action(self, sensors: Iterable[object]) -> str:
        nw, north, ne, east, se, south, sw, west = self.sense(sensors)

        if self.reverse_steps:
            self.reverse_steps -= 1
            return "N"

        if se and not south and sw:
            self.reverse_steps = 2
            return "S"

        if north and not east:
            return "E"
        if east and not south:
            return "S"
        if south and not west:
            return "W"
        if west and not north:
            return "N"

        return "N" if not any((nw, north, ne, east, se, south, sw, west)) else "W"


if __name__ == "__main__":
    agent = WallFollowAgent()
    assert agent.action([0] * 8) == "N"
    assert agent.action([0, 1, 0, 0, 0, 0, 0, 0]) == "E"
    assert agent.action([0, 0, 0, 0, 1, 0, 1, 0]) == "S"
    assert agent.action([0] * 8) == "N"
    print("wall-follow rules: checks passed")
