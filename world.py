"""A compact, fully synthetic three-junction practice course.

The image sensor returns a synthetic sign panel. It is not a rendered RGB scene
or an object detector. Ground-truth labels belong to this environment/evaluator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

JUNCTION_Y = (1.2, 4.4, 7.6)
BRANCH_X = 1.35
GOAL = (0.0, 10.85)
ROBOT_RADIUS = 0.18


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float
    kind: str = "pillar"

    def as_dict(self):
        return dict(x=self.x, y=self.y, width=self.width, height=self.height,
                    kind=self.kind)


class PracticeWorld:
    def __init__(self, seed: int, profile: str = "standard"):
        self.seed = int(seed)
        self.profile = profile
        rng = np.random.default_rng(self.seed)
        self._labels = rng.integers(0, 3, size=3).astype(int).tolist()
        self.start = (float(rng.uniform(-0.04, 0.04)), -0.35,
                      math.pi / 2 + float(rng.uniform(-0.04, 0.04)))
        self.wheel_gain = tuple(float(x) for x in rng.uniform(.98, 1.02, 2))
        self.obstacles = [Rectangle(x, y + 1.45, .34, .90)
                          for y in JUNCTION_Y for x in (-.65, .65)]
        self.obstacles += [Rectangle(-2.05, 5.25, .12, 12.3, "wall"),
                           Rectangle(2.05, 5.25, .12, 12.3, "wall"),
                           Rectangle(0, -0.88, 4.22, .12, "wall"),
                           Rectangle(0, 11.38, 4.22, .12, "wall")]

    def public_geometry(self):
        branches = []
        for y in JUNCTION_Y:
            branches.append({"junction": [0.0, y],
                             "left": self.branch_points(y, 0),
                             "straight": self.branch_points(y, 1),
                             "right": self.branch_points(y, 2)})
        return {"description": "synthetic sign-following practice course",
                "units": "metres and radians", "bounds": [-2, 2, -.82, 11.32],
                "start": list(self.start), "goal": list(GOAL),
                "junction_y": list(JUNCTION_Y), "branches": branches,
                "signs": [{"index": i, "x": 0., "y": y + .45,
                           "z": 1.05, "panel_width": .52,
                           "panel_height": .52, "facing": -math.pi / 2}
                          for i, y in enumerate(JUNCTION_Y)],
                "robot_radius": ROBOT_RADIUS,
                "obstacles": [ob.as_dict() for ob in self.obstacles]}

    @staticmethod
    def branch_points(y: float, chosen_label: int):
        if chosen_label == 1:
            return [[0.0, y + 2.75]]
        x = -BRANCH_X if chosen_label == 0 else BRANCH_X
        return [[x, y + .65], [x, y + 2.25], [0.0, y + 2.75]]

    def capture_sign(self, junction: int):
        from vision import render_sign
        # Per-junction seeds keep sensor data identical across policy variants.
        rng = np.random.default_rng(self.seed * 13 + 700_001 + junction)
        return render_sign(self._labels[junction], rng, profile=self.profile)

    def collides(self, x: float, y: float):
        for ob in self.obstacles:
            nx = np.clip(x, ob.x-ob.width/2, ob.x+ob.width/2)
            ny = np.clip(y, ob.y-ob.height/2, ob.y+ob.height/2)
            if (x-nx)**2 + (y-ny)**2 < ROBOT_RADIUS**2:
                return True
        return False

    def evaluate(self, choices):
        correct = [int(a == b) for a, b in zip(choices, self._labels)]
        return {"true_labels": self._labels.copy(), "correct_decisions": sum(correct),
                "decision_count": len(choices),
                "all_signs_followed": len(choices) == 3 and all(correct)}

