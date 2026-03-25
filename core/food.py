# food.py - plants that spawn randomly and give energy to herbivores

import numpy as np
import config as cfg


class Food:
    __slots__ = ("x", "y", "energy", "radius", "alive", "age")

    def __init__(self, x: float = None, y: float = None, energy: float = None):
        self.x = x if x is not None else np.random.uniform(20, cfg.WORLD_WIDTH - 20)
        self.y = y if y is not None else np.random.uniform(20, cfg.WORLD_HEIGHT - 20)
        self.energy = energy if energy is not None else cfg.FOOD_ENERGY
        self.radius = cfg.FOOD_RADIUS
        self.alive = True
        self.age = 0

    def update(self):
        self.age += 1

    def consume(self) -> float:
        """Eat this food, returns how much energy it had."""
        if not self.alive:
            return 0.0
        self.alive = False
        return self.energy

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
